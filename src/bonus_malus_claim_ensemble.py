"""Deployable pairwise and count-confirmed claim ensembles with Bonus-Malus timeline detection."""

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from src.bonus_malus_timeline import BonusMalusTimelineDetector
from src.claim_benchmark_pipeline import supported_admission
from src.claim_count_mixture import MAX_ROWS
from src.claim_history_leaf_retrieval import ClaimHistoryLeafRetrieval
from src.claim_probability import log_odds
from src.dashboard_model import profile_keys
from src.pairwise_payout_ensemble import PairwisePayoutEnsemble
from src.profile_count_hurdle import ProfileCountHurdle
from src.profile_count_survival import COUNT, ProfileCountSurvival


def _align_risk_with_flags(risk, flags, cutoff):
    """Keep decision_components() consistent with flags so threshold CV matches predict()."""
    if not np.isfinite(cutoff):
        return risk
    above = np.nextafter(cutoff, np.inf)
    below = np.nextafter(cutoff, -np.inf)
    aligned = np.where(flags & (risk < cutoff), np.maximum(risk, above), risk)
    return np.where(~flags & (aligned >= cutoff), np.minimum(aligned, below), aligned)


class PairwiseTimelineClaimEnsemble(PairwisePayoutEnsemble):
    """Pairwise payout reranker augmented with French Bonus-Malus reference-window cold detection."""

    def fit(self, features, amounts):
        with threadpool_limits(limits=2):
            super().fit(features, amounts)
            self.timeline_ = BonusMalusTimelineDetector(random_seed=self.random_seed).fit(features, amounts)
        return self

    def _score_batch(self, features):
        risk, prices, flags, cutoff = super()._score_batch(features)
        if not len(features):
            return risk, prices, flags, cutoff
        risk, prices, flags = self.timeline_.augment_predictions(features, risk, prices, flags, cutoff)
        return _align_risk_with_flags(risk, flags, cutoff), prices, flags, cutoff


class CountRetrievalTimelineClaimEnsemble(PairwiseTimelineClaimEnsemble):
    """Pairwise ensemble with count-hurdle agreement, leaf-retrieval admissions, and BM timeline detection."""

    def fit(self, features, amounts):
        with threadpool_limits(limits=2):
            super().fit(features, amounts)
            self.count_survival_ = ProfileCountSurvival().fit(features, amounts)
            self.count_hurdle_ = ProfileCountHurdle().fit(features, amounts)
            self.leaf_ranked_ = ClaimHistoryLeafRetrieval(ranking=True).fit(features, amounts)
            self.leaf_regular_ = ClaimHistoryLeafRetrieval().fit(features, amounts)
        return self

    def _refine_warm_flags(self, features, anchor, prices, flags, cutoff):
        """Apply count-survival/hurdle agreement and dual leaf-retrieval admissions on warm profiles."""
        _, inverse, peers = np.unique(profile_keys(features), return_inverse=True, return_counts=True)
        point, continuation, warm = self.count_survival_.predict(features, anchor)
        if not warm.any():
            return prices, flags
        _, hurdle, _ = self.count_hurdle_.predict(features, anchor)
        sizes = np.minimum(peers, MAX_ROWS)
        baseline, center = log_odds(point), log_odds(anchor)
        decisions = []
        for law in (continuation, hurdle):
            rate = ((law @ COUNT) / sizes)[inverse]
            shift = np.clip(log_odds(rate) - baseline, -0.6, 0.6)
            decisions.append(expit(center + shift / 2.0) >= cutoff)
        first, second = decisions
        near = warm & (np.abs(anchor - cutoff) <= 0.075)
        counted = np.where(near & (first == second), second, flags)
        ranked, _ = self.leaf_ranked_.predict(features)
        regular, _ = self.leaf_regular_.predict(features)
        admitted = ~counted & supported_admission(ranked, near) & supported_admission(regular, near)
        updated_flags = counted | admitted
        # Quote newly admitted warm profiles using fitted source-tail support and local tail caps.
        added_warm = updated_flags & ~flags
        if added_warm.any():
            addition_quotes = self.tail_.support(features)[1]
            tail_caps = self.timeline_.tail_.predict(features)
            warm_quotes = np.where(admitted, np.maximum(0.8 * addition_quotes, np.minimum(addition_quotes, tail_caps)), addition_quotes)
            prices = np.where(added_warm, warm_quotes, prices)
        return prices, updated_flags

    def _score_batch(self, features):
        # Run PairwisePayoutEnsemble first, then warm count+leaf refinement, then cold BM-window detection.
        risk, prices, flags, cutoff = PairwisePayoutEnsemble._score_batch(self, features)
        if not len(features):
            return risk, prices, flags, cutoff
        prices, flags = self._refine_warm_flags(features, risk, prices, flags, cutoff)
        risk, prices, flags = self.timeline_.augment_predictions(features, risk, prices, flags, cutoff)
        return _align_risk_with_flags(risk, flags, cutoff), prices, flags, cutoff
