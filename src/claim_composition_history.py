"""Describe the payout composition of each exact profile without its focal claim."""

from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from src.payout_pricing import payout_signature


COMPOSITION_COLUMNS = (
    "history_distinct_whole", "history_distinct_scaled", "history_repeated_whole",
    "history_repeated_scaled", "history_payout_reconciled", "history_payout_gap",
)


def describe_composition(payments):
    whole = {amount: copies for (amount, cent), copies in payments.items() if cent}
    scaled = {amount: copies for (amount, cent), copies in payments.items() if not cent}
    scaled_max = max(scaled, default=0)
    whole_total = sum(whole)
    return (
        len(whole), len(scaled), sum(count - 1 for count in whole.values()),
        sum(count - 1 for count in scaled.values()), bool(scaled_max and scaled_max == whole_total),
        (scaled_max - whole_total) / max(1, scaled_max, whole_total),
    )


class ClaimCompositionHistory:
    def fit(self, keys, amounts):
        payments = defaultdict(Counter)
        for key, amount in zip(keys, amounts, strict=True):
            if amount > 0:
                payments[key][payout_signature(amount)] += 1
        self.payments_ = dict(payments)
        self.compact_ = {key: describe_composition(marks) for key, marks in payments.items()}
        return self

    def _without(self, key, amount):
        mark = payout_signature(amount)
        other = self.payments_[key].copy()
        other[mark] -= 1
        if not other[mark]:
            del other[mark]
        return describe_composition(other)

    def transform(self, keys, index, amounts=None):
        design = np.array([self.compact_.get(key, (0,) * len(COMPOSITION_COLUMNS)) for key in keys], dtype=float)
        if amounts is not None:
            for position in np.flatnonzero(amounts > 0):
                design[position] = self._without(keys[position], amounts[position])
        return pd.DataFrame(design.reshape(-1, len(COMPOSITION_COLUMNS)), columns=COMPOSITION_COLUMNS, index=index)
