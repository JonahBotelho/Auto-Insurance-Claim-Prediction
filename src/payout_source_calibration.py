"""Calibrate separate probabilities for whole-cent and scaled claim outcomes."""

import numpy as np

from src.feature_schema import RANDOM_SEED
from src.payout_pricing import whole_cent


def payout_source_design(rows):
    return rows[[
        "whole_only", "scaled_only", "mixed_incomplete", "sum_attested", "copies_complete", "possible_copies_left",
        "whole_count", "scaled_count", "whole_families", "price_log", "price_small", "price_large", "whole_dollars",
        "other_whole", "other_scaled", "other_confirmed_sums", "other_split_sums", "log_source_ratio",
        "gap_fraction", "gap_other_whole", "copy_upper_rate", "exposure", "vehicle_age_code", "prior_zeros",
    ]].assign(
        scaled_source_balance=rows.scaled_only * rows.log_source_ratio,
        scaled_age=rows.scaled_only * rows.vehicle_age_code,
        scaled_common_payment=rows.scaled_only * rows.other_confirmed_sums,
        partial_gap_belief=rows.mixed_incomplete * rows.gap_other_whole,
    ).to_numpy(dtype=float)


class PayoutSourceCalibration:
    @staticmethod
    def _inputs(rows, probability):
        return np.column_stack([payout_source_design(rows), probability, rows.template_single,
                                rows.template_batch, rows.aggregate_choice])

    def fit(self, rows, probability):
        known = rows.prior_positives.gt(0).to_numpy()
        fitting = rows.loc[known]
        actual = fitting.actual.to_numpy()
        target = np.where(actual > 0, np.where(whole_cent(actual), 1, 2), 0)
        self.classifier_ = None
        if len(np.unique(target)) == 3:
            from catboost import CatBoostClassifier
            base = probability[known]
            weight = .5 + np.exp(-((base - .25) / .20) ** 2)
            self.classifier_ = CatBoostClassifier(
                iterations=200, depth=3, learning_rate=.04, l2_leaf_reg=35, random_seed=RANDOM_SEED,
                thread_count=2, verbose=False, allow_writing_files=False,
            ).fit(self._inputs(fitting, base), target, sample_weight=weight)
        return self

    def predict(self, rows, reference):
        adjusted = reference.copy()
        scaled_share = np.full(len(rows), np.nan)
        known = rows.prior_positives.gt(0).to_numpy()
        if self.classifier_ is None or not known.any():
            return adjusted, scaled_share
        held = rows.loc[known]
        classes = self.classifier_.predict_proba(self._inputs(held, reference[known]))
        # After two copies of the scaled amount, another has little support.
        classes[held.scaled_count.to_numpy() >= 2, 2] *= .02
        classes /= classes.sum(axis=1, keepdims=True)
        adjusted[known] = classes[:, 1:].sum(axis=1)
        scaled_share[known] = classes[:, 2] / np.maximum(adjusted[known], 1e-10)
        complete = rows.copies_complete.astype(bool).to_numpy()
        adjusted[complete] = np.minimum(adjusted[complete], .02)
        return adjusted, scaled_share
