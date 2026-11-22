"""Test whether payouts shared across profiles clarify ambiguous claim histories."""

from collections import Counter

import numpy as np
from scipy.special import expit

from src.masked_payout_history import signature_amount
from src.claim_probability import log_odds
from src.dashboard_model import profile_keys
from src.local_tail_refinement import history_risk_model, masked_risk_examples
from src.masked_payout_history import aggregate_history, describe_history, history_design


def fingerprint_index(paid_by_profile):
    return Counter(signature for paid in paid_by_profile.values() for signature in paid)


def fingerprints(observed, own, index):
    whole = [cents for cents, source in observed if source]
    scaled = [cents for cents, source in observed if not source]
    subtotal = sum(whole)
    missing = max(0, max(scaled, default=0) - subtotal) if whole else 0

    def matches(cents, source):
        key = cents, source
        # Exclude the entire profile, including all of its simulated hidden rows.
        return np.log1p(index[key] - (key in own)) if cents > 0 else 0.

    exact = [matches(*signature) for signature in observed]
    opposite = [matches(cents, not source) for cents, source in observed]
    return [max(exact, default=0.), max(opposite, default=0.),
            max((matches(cents, False) for cents in whole), default=0.),
            max((matches(cents, True) for cents in scaled), default=0.),
            matches(subtotal, False), matches(missing, True),
            np.mean([cents % 100 == 0 for cents, _ in observed]),
            np.mean([cents % 1000 == 0 for cents, _ in observed]),
            np.mean([cents % 5000 == 0 for cents, _ in observed]),
            np.log1p(min(signature_amount(signature) for signature in observed))]


class PayoutCollisionEvidence:
    """Compare matched masked models with and without empirical payout fingerprints."""

    def fit(self, features, target):
        keys, self.zeros_, self.paid_ = aggregate_history(features, target)
        self.index_ = fingerprint_index(self.paid_)
        positions, histories, extras, labels, weights = [], [], [], [], []
        for position, observed, history, label, weight in masked_risk_examples(keys, self.zeros_, self.paid_):
            positions.append(position)
            histories.append(history)
            extras.append(fingerprints(observed, self.paid_[keys[position]], self.index_))
            labels.append(label)
            weights.append(weight)
        control, full = self.design(features.iloc[positions], histories, extras)
        self.models_ = [history_risk_model().fit(frame, labels, sample_weight=weights) for frame in (control, full)]
        return self

    @staticmethod
    def design(features, histories, extras):
        controls = history_design(features, histories)
        return controls, controls.assign(**{f"payout_fingerprint_{i}": vector for i, vector in enumerate(np.asarray(extras).T)})

    def predict(self, query, anchor):
        keys, positions, inverse, counts = np.unique(profile_keys(query), return_index=True, return_inverse=True,
                                                     return_counts=True)
        selected = np.flatnonzero([key in self.paid_ for key in keys])
        risks = np.tile(np.asarray(anchor)[positions], (2, 1))
        histories, extras = [], []
        for index in selected:
            key = keys[index]
            zero, paid = self.zeros_[key], self.paid_[key]
            histories.append(describe_history(zero, paid, zero + sum(paid.values()) + counts[index]))
            extras.append(fingerprints(paid, paid, self.index_))
        if len(selected):
            designs = self.design(query.iloc[positions[selected]], histories, extras)
            for values, model, design in zip(risks, self.models_, designs, strict=True):
                values[selected] = model.predict(design)
        base, full = risks[:, inverse]
        shift = np.clip(log_odds(full) - log_odds(base), -.65, .65)
        return base, full, expit(log_odds(anchor) + .5 * shift)
