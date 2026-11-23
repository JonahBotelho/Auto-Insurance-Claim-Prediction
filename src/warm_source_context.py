"""Fitting-only payout table frequencies and support for two-component settlements."""

from collections import Counter
import numpy as np


CONTEXT_COLUMNS = (
    "source_whole_share", "source_share_whole_only", "source_share_scaled_only",
    "source_cent_backing", "source_scaled_backing", "source_two_sum", "source_gap_two_sum",
    "source_gap_backing", "source_whole_only_families",
)


class WarmSourceContext:
    def fit(self, paid_profiles, vehicle_ages):
        self.cent_, self.scaled_ = Counter(), Counter()
        counts = {}
        for key, paid in paid_profiles.items():
            age = int(vehicle_ages[key])
            table_counts = counts.setdefault(age, np.zeros(2))
            for (amount, whole), count in paid.items():
                (self.cent_ if whole else self.scaled_)[amount] += 1
                table_counts[int(whole)] += count
        combined = np.sum(list(counts.values()), axis=0)
        prior = combined[1] / combined.sum() if counts else .5
        self.prior_ = prior
        self.age_share_ = {age: (values[1] + 40 * prior) / (values.sum() + 40) for age, values in counts.items()}
        self.common_ = {amount: count for amount, count in self.cent_.items() if count >= 2}
        self.pair_support_ = {}
        return self

    def pair_support(self, total):
        if total not in self.pair_support_:
            self.pair_support_[total] = sum(
                min(count, self.common_.get(total - first, 0))
                for first, count in self.common_.items() if first <= total - first
            )
        return self.pair_support_[total]

    def describe(self, whole, scaled, vehicle_age):
        top = max(scaled, default=0)
        subtotal = sum(whole)
        residual = top - subtotal
        share = self.age_share_.get(int(vehicle_age), self.prior_)
        return (
            share, share * (not scaled), share * (not whole),
            np.log1p(self.cent_[top]), np.log1p(self.scaled_[subtotal]),
            np.log1p(self.pair_support(top)), np.log1p(self.pair_support(residual)),
            np.log1p(self.cent_[residual]), len(whole) * (not scaled),
        )
