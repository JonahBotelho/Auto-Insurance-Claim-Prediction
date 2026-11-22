"""Training-only claim history and positive amount medians for exact feature profiles."""

import numpy as np
import pandas as pd

from src.feature_schema import (
    ALL_FEATURE_COLUMNS, PROFILE_HISTORY_COLUMNS, DEFAULT_PROFILE_HISTORY_SMOOTHING,
)


class ProfileHistory:
    def __init__(self, smoothing=DEFAULT_PROFILE_HISTORY_SMOOTHING):
        self.smoothing = smoothing

    def fit(self, features, amounts):
        work = features[ALL_FEATURE_COLUMNS].reset_index(drop=True).copy()
        work["_profile_claim"] = (np.asarray(amounts) > 0).astype(int)
        work["_profile_amount"] = np.asarray(amounts)
        grouped = work.groupby(ALL_FEATURE_COLUMNS, observed=True, dropna=False)
        self.counts_ = grouped["_profile_claim"].agg(["sum", "size"])
        self.claim_rate_ = work["_profile_claim"].mean()
        positive = work[work["_profile_amount"] > 0]
        self.medians_ = positive.groupby(
            ALL_FEATURE_COLUMNS, observed=True, dropna=False,
        )["_profile_amount"].median()
        return self

    def _counts_for(self, features):
        index = pd.MultiIndex.from_frame(features[ALL_FEATURE_COLUMNS])
        return self.counts_.reindex(index).fillna(0)

    def transform(self, features, own_claim=None):
        lookup = self._counts_for(features)
        count = lookup["size"].to_numpy(dtype=float)
        claims = lookup["sum"].to_numpy(dtype=float)
        if own_claim is not None:
            # During classifier fitting, remove each row's own label and support.
            # Validation/inference rows never supply labels.
            count = count - 1
            claims = claims - np.asarray(own_claim)
        denominator = count + self.smoothing
        rate = np.divide(
            claims + self.smoothing * self.claim_rate_, denominator,
            out=np.full(len(features), self.claim_rate_), where=denominator > 0,
        )
        result = features.copy()
        result[PROFILE_HISTORY_COLUMNS[0]] = count
        result[PROFILE_HISTORY_COLUMNS[1]] = claims
        result[PROFILE_HISTORY_COLUMNS[2]] = rate
        return result

    def predict_amount(self, features, fallback):
        index = pd.MultiIndex.from_frame(features[ALL_FEATURE_COLUMNS])
        median = self.medians_.reindex(index).to_numpy()
        return np.where(np.isfinite(median), median, fallback)
