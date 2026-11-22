"""Estimate low payout quantiles from fitting-only cohorts and payout sources."""

import numpy as np
import pandas as pd

from src.dashboard_model import profile_keys
from src.masked_payout_history import aggregate_history
from src.payout_pricing import whole_cent


def quote_cells(features):
    # The ordinal fleet bands and exposure require no target-derived encodings.
    exposure = np.minimum(3, (4 * features.feature1.to_numpy()).astype(int))
    vehicle = np.searchsorted([3, 6], features.feature3.to_numpy())
    return exposure * 3 + vehicle


def weighted_quantile(values, weights, quantile):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, cumulative[-1] * quantile)), len(order) - 1)
    return float(values[order[index]])


class LocalPayoutTail:
    """Shrink each payout cohort toward its matching source's pooled lower tail."""

    def fit(self, features, amounts):
        y = np.asarray(amounts)
        positive = y > 0.
        self.payouts_, self.cells_ = y[positive], quote_cells(features)[positive]
        self.sources_ = whole_cent(self.payouts_)
        self.raw_floor_ = float(np.quantile(self.payouts_, .02))
        _, _, self.paid_ = aggregate_history(features, y)
        self.quotes_ = {}
        pooled = self.sources_.mean()
        for cell in np.unique(quote_cells(features)):
            local = self.cells_ == cell
            for count in range(5):
                whole = (pooled + count / 4) / 2
                weights = np.zeros(len(self.payouts_))
                for source, mass in ((True, whole), (False, 1 - whole)):
                    matching = self.sources_ == source
                    if not matching.any():
                        continue
                    region = matching & local
                    within = region.sum() / (region.sum() + 130.)
                    weights += mass * ((1 - within) * matching / matching.sum()
                                       + within * region / max(1, region.sum()))
                self.quotes_[cell, count] = weighted_quantile(self.payouts_, weights, .02)
        return self

    def predict(self, query):
        estimates = np.full(len(query), self.raw_floor_)
        for index, (key, cell) in enumerate(zip(profile_keys(query), quote_cells(query), strict=True)):
            past = self.paid_.get(key)
            if past:
                fraction = sum(n for (_, cent), n in past.items() if cent) / sum(past.values())
                estimates[index] = self.quotes_.get((cell, int(np.rint(4 * fraction))), self.raw_floor_)
        return estimates
