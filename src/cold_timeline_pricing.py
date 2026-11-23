"""Use the estimated zero mass to price high-confidence timeline claims."""

import numpy as np


class ColdTimelinePricing:
    """Blend positive payout quantiles from linked profiles and all fitting claims."""

    def fit(self, keys, amounts, linked_profiles):
        amounts = np.asarray(amounts)
        positive = amounts > 0
        self.pooled_ = amounts[positive]
        linked = amounts[positive & np.isin(keys, linked_profiles)]
        self.linked_ = linked if len(linked) >= 20 else self.pooled_
        return self

    def predict(self, eligible_risk, tail):
        # At the unconditional median, the total mass below the quote is 0.5.
        # Remove the zero mass before looking up a positive payout quantile.
        quantile = np.clip(1. - .5 / eligible_risk, 0., .25)
        linked = np.quantile(self.linked_, quantile)
        pooled = np.quantile(self.pooled_, quantile)
        return np.maximum(tail, (linked + pooled) / 2)
