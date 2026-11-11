"""
Builds the shared preprocessing step: scales continuous/discrete features
and one-hot encodes categorical features, bucketing rare categories
automatically so a handful of near-empty categories don't get their own
unstable column.
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES, RARE_CATEGORY_MIN_FREQUENCY


def build_preprocessing_pipeline(
    min_category_frequency: int = RARE_CATEGORY_MIN_FREQUENCY,
) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_category_frequency,
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
