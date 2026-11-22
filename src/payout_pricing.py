"""Price flagged exact-profile matches using other observed payouts and zeros."""

from collections import defaultdict

import numpy as np
import pandas as pd

from src.feature_schema import CENT_TOLERANCE, PAYOUT_FACTOR, PROFILE_COLUMNS, RANDOM_SEED

HISTORY_COLUMNS = ("k", "z", "distinct", "cent", "anchor")
AMOUNT_FEATURES = [*HISTORY_COLUMNS, "feature1"]
MEDIAN_PRICING_CAP = 2250.
MEDIAN_PRICING_PARAMS = {
    "loss_function": "Quantile:alpha=0.5", "iterations": 300, "depth": 3, "learning_rate": .04,
    "l2_leaf_reg": 20, "random_seed": RANDOM_SEED, "thread_count": 2, "allow_writing_files": False, "verbose": False,
}


def whole_cent(amount):
    return abs(amount - np.round(amount, 2)) < CENT_TOLERANCE


def payout_signature(amount):
    cent = bool(whole_cent(amount))
    normalized = amount if cent else amount / PAYOUT_FACTOR
    return int(round(100 * normalized)), cent


def profile_tuples(rows):
    return rows.loc[:, PROFILE_COLUMNS].itertuples(index=False, name=None)


def payout_summary(positive, zero_count, default_amount):
    if not positive:
        return 0, min(zero_count, 4), 0, 0, default_amount
    cents = [abs(value - round(value, 2)) < CENT_TOLERANCE for value in positive]
    # The development data contains repeats with a consistent multiplier. Keep the
    # observed cents intact and use the inferred base amount only for history features.
    normalized = [value if cent else round(value / PAYOUT_FACTOR, 2) for value, cent in zip(positive, cents)]
    return (min(len(positive), 4), min(zero_count, 4), min(len(set(normalized)), 3),
            int(any(cents)) + int(all(cents)), float(np.median(normalized)))


class PayoutHistory:
    def __init__(self, fitting, amounts):
        self.positive = defaultdict(list)
        self.zero = defaultdict(int)
        amounts = np.asarray(amounts)
        for key, amount in zip(profile_tuples(fitting), amounts):
            if amount > 0:
                self.positive[key].append(float(amount))
            else:
                self.zero[key] += 1
        self.default_amount = float(np.median(amounts[amounts > 0]))

    def transform(self, rows, *, own_amounts=None):
        if own_amounts is None:
            own_amounts = np.full(len(rows), -1.)
        summaries = []
        positive_counts = np.empty(len(rows), dtype=int)
        zero_counts = np.empty(len(rows), dtype=int)
        for index, (key, amount) in enumerate(zip(profile_tuples(rows), own_amounts)):
            positive = self.positive.get(key, [])
            zeros = self.zero.get(key, 0) - int(amount == 0)
            if amount > 0:
                positive = positive.copy()
                positive.remove(amount)
            positive_counts[index] = len(positive)
            zero_counts[index] = zeros
            summaries.append(payout_summary(positive, zeros, self.default_amount))
        return pd.DataFrame(summaries, columns=HISTORY_COLUMNS, index=rows.index), positive_counts, zero_counts


class MedianPayoutPricer:
    def fit(self, features, amounts):
        from catboost import CatBoostRegressor

        amounts = np.asarray(amounts)
        self.history_ = PayoutHistory(features, amounts)
        history, positive, _ = self.history_.transform(features, own_amounts=amounts)
        known = positive > 0
        self.model_ = None
        self.floor_ = min(MEDIAN_PRICING_CAP, float(np.quantile(amounts[amounts > 0], .05)))
        if known.any():
            inputs = history.loc[known].assign(feature1=features.loc[known, "feature1"])
            self.model_ = CatBoostRegressor(**MEDIAN_PRICING_PARAMS).fit(inputs, amounts[known])
        return self

    def predict(self, features, fallback):
        prices = np.array(fallback, dtype=float, copy=True)
        if self.model_ is not None:
            history, positive, _ = self.history_.transform(features)
            known = (positive > 0) & (prices > 0)
            if known.any():
                inputs = history.loc[known].assign(feature1=features.loc[known, "feature1"])
                prices[known] = np.clip(self.model_.predict(inputs), self.floor_, MEDIAN_PRICING_CAP)
        return prices
