"""Estimate absolute-error savings for alternative quotes from masked payout histories."""

import numpy as np
from lightgbm import LGBMRegressor

from src.feature_schema import RANDOM_SEED
from src.masked_payout_history import observed_support, signature_amount

QUOTE_FRACTIONS = np.array([.25, .5, .75, 1.])


def masked_quote_targets(full_paid, observed_paid, rate):
    # Sample quote sizes around available payout evidence. Hidden rows contribute
    # min(actual, quote) to the score, including zero when no claim occurred.
    support = observed_support(observed_paid)
    quotes = np.median(support) * np.array([.125, .25, .5, 1., 2.])
    quotes = np.maximum(1., quotes)
    hidden = full_paid - observed_paid
    if not hidden:
        return quotes, np.zeros_like(quotes)
    amounts = np.array([signature_amount(mark) for mark in hidden])
    counts = np.array(list(hidden.values()))
    targets = np.average(np.minimum(amounts[:, None] / quotes, 1.), weights=counts, axis=0)
    return quotes, rate * targets


class MaskedQuotePricing:
    def fit(self, design, quotes, targets, weights, random_seed=RANDOM_SEED):
        quote_count = quotes.shape[1]
        inputs = np.column_stack([np.repeat(design, quote_count, axis=0), np.log(quotes.ravel())])
        self.model_ = LGBMRegressor(
            objective="cross_entropy", n_estimators=180, num_leaves=12, min_child_samples=120,
            learning_rate=.035, reg_lambda=40., random_state=random_seed, n_jobs=2,
            verbosity=-1, deterministic=True, force_col_wise=True,
        ).fit(inputs, targets.ravel(), sample_weight=np.repeat(weights / quote_count, quote_count))
        return self

    def predict(self, design, original, floor=1.):
        quotes = np.maximum(floor, original[:, None] * QUOTE_FRACTIONS)
        inputs = np.column_stack([np.repeat(design, len(QUOTE_FRACTIONS), axis=0), np.log(quotes.ravel())])
        utility = quotes * (2 * self.model_.predict(inputs).reshape(quotes.shape) - 1)
        chosen = utility.argmax(axis=1)
        best = np.take_along_axis(quotes, chosen[:, None], axis=1).ravel()
        gain = np.take_along_axis(utility, chosen[:, None], axis=1).ravel()
        return np.where(gain > 0, best, floor)
