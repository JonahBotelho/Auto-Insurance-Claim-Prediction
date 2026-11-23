"""Describe payout-copy capacity and independent support for an observed amount."""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from src.dashboard_model import profile_keys
from src.exact_em_deconvolution import EMReconciledVeto
from src.payout_pricing import payout_signature
from src.warm_source_context import CONTEXT_COLUMNS, WarmSourceContext

COLUMNS = (
    "whole_only", "scaled_only", "mixed_incomplete", "sum_attested", "copies_complete", "possible_copies_left",
    "whole_count", "scaled_count", "whole_families", "price_log", "price_small", "price_large", "whole_dollars",
    "other_whole", "other_scaled", "other_confirmed_sums", "other_split_sums", "log_source_ratio",
    "gap_fraction", "gap_other_whole", "copy_upper_rate",
)
STRUCTURAL_COLUMNS = (*CONTEXT_COLUMNS, "source_em_probability", "source_em_support")


class PayoutSourceEvidence:
    def __init__(self, structural=False):
        self.structural = structural

    def fit(self, features, amounts):
        self.profiles_ = defaultdict(Counter)
        keys = profile_keys(features)
        for key, amount in zip(keys, amounts):
            if amount > 0:
                self.profiles_[key][payout_signature(amount)] += 1
        self.whole_support_ = Counter()
        self.scaled_support_ = Counter()
        self.sum_support_ = Counter()
        self.split_support_ = Counter()
        for observed in self.profiles_.values():
            cent, scaled = self._by_source(observed)
            self.whole_support_.update(cent.keys())
            self.scaled_support_.update(scaled.keys())
            if cent and sum(cent) in scaled:
                self.sum_support_[sum(cent)] += 1
                self.split_support_[sum(cent)] += int(len(cent) > 1)
        if self.structural:
            ages = dict(zip(keys, features.feature3))
            self.source_context_ = WarmSourceContext().fit(self.profiles_, ages)
            self.em_ = EMReconciledVeto().fit(features, amounts)
        return self

    @staticmethod
    def _by_source(observed):
        return ({amount: count for (amount, whole), count in observed.items() if whole},
                {amount: count for (amount, whole), count in observed.items() if not whole})

    def _describe(self, key, peers):
        observed = self.profiles_.get(key, {})
        if not observed:
            return (0.,) * len(COLUMNS)
        whole, scaled = self._by_source(observed)
        subtotal = sum(whole)
        top = max(scaled, default=0)
        anchor = top or max(whole, default=0)
        paid = anchor / 100.
        reconciled = len(scaled) == 1 and top == subtotal
        gap = max(0, top - subtotal)
        upper = sum(max(0, 2 - count) for count in observed.values()) if reconciled else 0
        overlap = min(1., upper / peers) if reconciled else 1.
        other_whole = self.whole_support_[anchor] - int(anchor in whole)
        other_scaled = self.scaled_support_[anchor] - int(anchor in scaled)
        own_sum = anchor == subtotal and anchor in scaled
        confirmed = self.sum_support_[anchor] - int(own_sum)
        splits = self.split_support_[anchor] - int(own_sum and len(whole) > 1)
        gap_backing = self.whole_support_[gap] - int(gap in whole)
        return (
            not scaled, not whole, bool(whole and scaled and not reconciled), reconciled, reconciled and upper == 0,
            upper, sum(whole.values()), sum(scaled.values()), len(whole), np.log1p(paid), paid < 250, paid > 2500,
            anchor % 100 == 0, np.log1p(other_whole), np.log1p(other_scaled), np.log1p(confirmed), np.log1p(splits),
            np.log((1. + other_whole) / (1. + other_scaled)), gap / max(top, 1), np.log1p(gap_backing), overlap,
        )

    def transform(self, features):
        keys = profile_keys(features)
        repeated = pd.Series(keys).value_counts(sort=False)
        described = {key: self._describe(key, size) for key, size in repeated.items()}
        return pd.DataFrame.from_records([described[key] for key in keys], columns=COLUMNS, index=features.index)

    def score_inputs(self, features):
        result = self.transform(features).assign(
            exposure=features.feature1,
            vehicle_age_code=features.feature3.astype(float),
        )
        if not getattr(self, "structural", False):
            return result
        warm, _, probability, support = self.em_.predict(features)
        context = np.zeros((len(features), len(CONTEXT_COLUMNS)))
        for index, (key, age) in enumerate(zip(profile_keys(features), features.feature3)):
            if warm[index]:
                cent, scaled = self._by_source(self.profiles_[key])
                context[index] = self.source_context_.describe(cent, scaled, age)
        for column, values in zip(CONTEXT_COLUMNS, context.T):
            result[column] = values
        return result.assign(source_em_probability=probability, source_em_support=np.log1p(support * 1000))
