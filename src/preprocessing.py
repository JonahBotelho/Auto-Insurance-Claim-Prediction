"""
Shared preprocessing for every claim model (Module 3 input transformations).

- Categorical features are one-hot encoded, so level 8 is not treated as
  "bigger" than level 1.
- Numeric features are standardized. Lasso, ridge and k-NN need this because
  their penalties or distances depend on each feature's scale.
- Claim amounts can be modelled as log(1 + ClaimAmount). Squared-error models
  fitted on raw amounts are dominated by the few $100k+ claims.
"""

import numpy as np
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Features with a few discrete levels that act as categories, chosen during
# data exploration (2-10 distinct values each). The binary features 4, 5 and 9
# need no encoding, so they are scaled with the numeric features.
CATEGORICAL_FEATURES = [
    "feature3", "feature7", "feature11", "feature13",
    "feature14", "feature15", "feature16", "feature18",
]
NUMERIC_FEATURES = [
    "feature1", "feature2", "feature4", "feature5", "feature6",
    "feature8", "feature9", "feature10", "feature12", "feature17",
]


RARE_LEVEL_MINIMUM_ROWS = 20


def build_preprocessor() -> ColumnTransformer:
    """One-hot encode categorical features and standardize numeric ones."""
    one_hot = OneHotEncoder(
        # Drop one level per feature: the full set of dummy columns always
        # sums to 1, which duplicates the intercept.
        drop="first",
        # Levels with fewer than 20 training rows (e.g. feature13 = 0 or 3,
        # with 4 and 11 rows) share one "infrequent" column instead of getting
        # their own, since a handful of rows is mostly noise.
        min_frequency=RARE_LEVEL_MINIMUM_ROWS,
        # A level unseen in training is put in the infrequent column instead
        # of crashing on the test or competition set.
        handle_unknown="infrequent_if_exist",
    )
    return ColumnTransformer([
        ("categorical", one_hot, CATEGORICAL_FEATURES),
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
    ])


def log_scale_to_claim_amount(log_predictions):
    """Convert log(1 + y) predictions back to dollars, floored at $0.

    A model can predict below zero on the log scale, which converts to a
    negative claim. Real claims are never negative, so a negative prediction
    is always worse than $0 under MAE.
    """
    return np.maximum(np.expm1(log_predictions), 0.0)


def fit_on_log_target(regressor) -> TransformedTargetRegressor:
    """Wrap a regressor so it trains on log(1 + y) and predicts in dollars."""
    return TransformedTargetRegressor(
        regressor=regressor, func=np.log1p, inverse_func=log_scale_to_claim_amount
    )
