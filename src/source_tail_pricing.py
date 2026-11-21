"""Estimate positive fallback amounts from the lower tails of fitted payout sources."""

from collections import defaultdict

import numpy as np

from src.dashboard_model import profile_keys
from src.payout_pricing import whole_cent


class SourceTailPricing:
    def fit(self, features, amounts):
        amounts = np.asarray(amounts)
        positive = amounts > 0
        payouts = amounts[positive]
        self.floor_ = float(np.quantile(payouts, .05))
        pooled = self.pooled_tail_ = float(np.quantile(payouts, .02))
        cent = whole_cent(payouts)
        tails = {}
        for kind in (True, False):
            matching = payouts[cent == kind]
            tails[kind] = float(np.quantile(matching, .02)) if len(matching) else pooled

        payout_groups = defaultdict(list)
        for key, amount, kind in zip(profile_keys(features)[positive], payouts, cent):
            payout_groups[key].append((amount, tails[kind]))
        self.observed_ = {}
        self.source_ = {}
        for key, group in payout_groups.items():
            local, support = zip(*group)
            self.observed_[key] = min(self.floor_, min(local))
            self.source_[key] = min(self.floor_, (pooled + np.mean(support)) / 2)
        return self

    def support(self, features):
        keys = profile_keys(features)
        observed = np.fromiter((self.observed_.get(key, self.floor_) for key in keys), dtype=float)
        source = np.fromiter((self.source_.get(key, self.floor_) for key in keys), dtype=float)
        return observed, source

    def predict(self, features, prices):
        # Only revisit estimates resting on the existing global floor.
        # The original detector still decides which rows get a payout.
        at_floor = np.isclose(prices, self.floor_, rtol=0., atol=1e-8)
        if not at_floor.any():
            return prices
        result = prices.copy()
        _, source = self.support(features.loc[at_floor])
        result[at_floor] = source
        return result


class PayoutSourceQuantiles:
    """Estimate positive payout quantiles for a mixture of the two sources."""

    def fit(self, amounts, quantiles):
        amounts = np.asarray(amounts)
        positive = np.sort(amounts[amounts > 0.])
        scaled = ~whole_cent(positive)
        self.shares_ = np.linspace(0., 1., 101)
        self.pooled_ = float(scaled.mean())
        if not scaled.any() or scaled.all():
            self.quotes_ = [np.full(len(self.shares_), np.quantile(positive, q)) for q in quantiles]
            return self
        cent_cdf = np.cumsum(~scaled) / (~scaled).sum()
        scaled_cdf = np.cumsum(scaled) / scaled.sum()
        cdf = cent_cdf + self.shares_[:, None] * (scaled_cdf - cent_cdf)
        self.quotes_ = [positive[np.argmax(cdf >= q, axis=1)] for q in quantiles]
        return self

    def predict(self, scaled_weight):
        # Use pooled source shares to stabilize estimates at the extremes.
        weight = (np.asarray(scaled_weight) + self.pooled_) / 2
        return tuple(np.interp(weight, self.shares_, quote) for quote in self.quotes_)
