"""Refine unsupported low quotes using only observed payouts and fitted risk."""

import numpy as np

from src.dashboard_model import profile_keys
from src.feature_schema import RANDOM_SEED
from src.local_payout_tail import LocalPayoutTail
from src.masked_payout_history import aggregate_history, describe_history, history_design, masked_states, observed_support


def masked_risk_examples(keys, zeros, payments):
    for position in np.unique(keys, return_index=True)[1]:
        key = keys[position]
        if key not in payments:
            continue
        paid = payments[key]
        total = zeros[key] + sum(paid.values())
        for retention in (.78, .93):
            for zero, observed, negative, positive in masked_states(zeros[key], paid, retention):
                weight = negative + positive
                if observed and weight > 1e-8:
                    yield position, observed, describe_history(zero, observed, total), positive / weight, weight / 2


def history_risk_model():
    from lightgbm import LGBMRegressor
    return LGBMRegressor(objective="cross_entropy", n_estimators=165, num_leaves=10, min_child_samples=100,
                         learning_rate=.035, reg_lambda=48., random_state=RANDOM_SEED,
                         n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True)


def refine_low_quotes(tail, query, original, paid, low_risk):
    selected = (original > 0.) & (original <= 1.25 * tail.raw_floor_) & low_risk
    if not selected.any():
        return original
    keys = profile_keys(query)
    for index in np.flatnonzero(selected):
        history = paid.get(keys[index])
        selected[index] = bool(history) and not np.isclose(observed_support(history), original[index], rtol=0., atol=.005).any()
    candidate = np.maximum(.8 * original, np.minimum(tail.predict(query), original))
    return np.where(selected, candidate, original)


def cap_to_observed_payout_support(query, prices, flags, paid, floor):
    """Cap flagged quotes at the highest payout supported by their observed history."""
    selected = np.flatnonzero(flags & (prices > floor))
    if not len(selected):
        return prices
    result = prices.copy()
    limit = {}
    for index, key in zip(selected, profile_keys(query.iloc[selected]), strict=True):
        history = paid.get(key)
        if history:
            if key not in limit:
                limit[key] = max(floor, float(observed_support(history).max()))
            result[index] = min(result[index], limit[key])
    return result


class LocalTailRefinement:
    """Keep positive flags and lower only quotes with weak payout support."""

    def fit(self, features, target):
        keys, self.zeros_, self.paid_ = aggregate_history(features, target)
        self.tail_ = LocalPayoutTail().fit(features, target)
        panel = list(masked_risk_examples(keys, self.zeros_, self.paid_))
        self.risk_ = None
        if panel:
            positions, _, histories, labels, weights = zip(*panel, strict=True)
            design = history_design(features.iloc[list(positions)], histories)
            self.risk_ = history_risk_model().fit(design, np.asarray(labels), sample_weight=np.asarray(weights))
        return self

    def predict(self, query, prices):
        if self.risk_ is None or not np.any(prices > 0.):
            return prices
        keys, first, inverse, sizes = np.unique(profile_keys(query), return_index=True, return_inverse=True,
                                                return_counts=True)
        positions, histories = [], []
        for i, key in enumerate(keys):
            if key in self.paid_:
                zero, paid = self.zeros_[key], self.paid_[key]
                positions.append(i)
                histories.append(describe_history(zero, paid, zero + sum(paid.values()) + sizes[i]))
        if not positions:
            return prices
        low = np.zeros(len(keys), dtype=bool)
        low[positions] = self.risk_.predict(history_design(query.iloc[first[positions]], histories)) < .36
        return refine_low_quotes(self.tail_, query, prices, self.paid_, low[inverse])
