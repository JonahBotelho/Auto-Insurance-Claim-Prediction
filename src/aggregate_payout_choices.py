"""Model scaled profile payouts as totals of underlying whole-cent claims."""

from collections import Counter

import numpy as np

from src.claim_payout_choices import PayoutChoices
from src.feature_schema import PAYOUT_FACTOR


def payout_bases(counts):
    whole = {amount for amount, cent in counts if cent}
    scaled = {amount for amount, cent in counts if not cent}
    return whole, scaled


class AggregatePayoutChoices(PayoutChoices):
    """Use observed totals, inferred residuals and recurring decompositions as choices."""

    evidence_size = 12

    def fit_candidate_support(self):
        self.decompositions_ = {}
        for key, counts in self.positive_.items():
            whole, scaled = payout_bases(counts)
            if whole and sum(whole) in scaled:
                by_total = self.decompositions_.setdefault(sum(whole), {})
                by_total.setdefault(frozenset(whole), set()).add(key)

    @staticmethod
    def _group_features(exposure, rating, zeros, counts):
        base = PayoutChoices._group_features(exposure, rating, zeros, counts)
        whole, scaled = payout_bases(counts)
        disclosed = sum(whole)
        covered = disclosed in scaled
        remainder = max(scaled, default=0) - disclosed
        return (*base, bool(scaled), covered, bool(scaled) and remainder > 0, bool(scaled) and remainder < 0,
                np.log1p(max(0, remainder) / 100), len(whole) >= 2)

    def _payout_candidates(self, key, counts):
        whole, scaled = payout_bases(counts)
        exposed = sum(whole)
        remaining = {total - exposed for total in scaled if total > exposed}
        learned = self._find_components(key, whole, scaled)
        scaled_choices = scaled or {exposed}
        whole_choices = whole | remaining | learned.keys()
        for cent, options, kind in ((True, whole_choices, "repeat_cent"), (False, scaled_choices, "repeat_scaled")):
            for base in sorted(options):
                support = counts.get((base, cent), 0)
                paired = counts.get((base, not cent), 0)
                profiles = self.profiles_by_payout_.get(base, ())
                others = len(profiles) - int(key in profiles)
                # Residual arithmetic uses only the cents already observed for the
                # queried profile. Cross-profile evidence excludes that entire profile.
                evidence = (
                    np.log1p(base / 100) - 7, support == 0, support >= 2, paired == 0, paired >= 2,
                    np.log1p(others), base % 100 == 0, cent and base in remaining,
                    not cent and not scaled, cent and (exposed in scaled), not cent and base == exposed,
                    np.log1p(learned.get(base, 0)) if cent else 0.,
                )
                yield kind, base / 100 if cent else PAYOUT_FACTOR * base / 100, evidence, (base, cent)

    def _find_components(self, key, observed, totals):
        counts = Counter()
        for total in totals:
            for components, profiles in self.decompositions_.get(total, {}).items():
                support = len(profiles) - int(key in profiles)
                if support > 0 and observed <= components:
                    for component in components:
                        counts[component] += support
        return dict(counts.most_common(4))
