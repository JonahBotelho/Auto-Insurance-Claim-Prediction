"""Prepare exposure and tenure descriptors from the anonymous policy columns."""

import numpy as np

from src.feature_schema import CATEGORICAL_FEATURES, PROFILE_COLUMNS

CATEGORIES = (*[column for column in CATEGORICAL_FEATURES if column != "feature3"], "feature6", "duration_form")


def actuarial_features(rows, zero_history):
    features = rows.loc[:, PROFILE_COLUMNS].copy()
    exposure = rows.feature1
    month_error = np.abs(exposure * 12 - np.round(exposure * 12))
    day_error = np.minimum(np.abs(exposure * 365 - np.round(exposure * 365)),
                           np.abs(exposure * 366 - np.round(exposure * 366)))
    features["duration_form"] = np.select([exposure.eq(1), month_error <= .008, day_error <= .19], [0, 1, 2], default=3)
    features["duration_form"] = features.duration_form.astype("category")
    features["feature3"] = features.feature3.astype(float)
    features["feature6"] = features.feature6.astype("category")
    # These are proxies for licence tenure and clean-discount progression
    # inferred from the anonymous values, so the base columns remain inputs.
    experience = rows.feature2 / 12
    reference_bonus = np.maximum(50, 100 * .95 ** experience)
    features["licence_delay"] = rows.feature8 - 18 - experience
    features["bonus_departure"] = np.log(rows.feature10 / reference_bonus) / np.log(1.25 / .95)
    features["bonus_floor"] = rows.feature10.eq(50).astype(float)
    features["market_segment"] = rows.feature12 - rows.feature3.astype(float)
    features["zero_history"] = zero_history
    return features
