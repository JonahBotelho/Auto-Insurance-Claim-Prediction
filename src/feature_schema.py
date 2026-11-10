"""
Shared constants describing the dataset's column names and feature groupings.
Every other module imports from here instead of re-declaring these lists,
so feature-type decisions (made during EDA) live in exactly one place.
"""

ROW_INDEX_COLUMN = "rowIndex"
TARGET_COLUMN = "ClaimAmount"
HAS_CLAIM_COLUMN = "has_claim"

# Continuous features: many distinct values, true numeric scale.
CONTINUOUS_FEATURES = ["feature1", "feature2"]

# Discrete counts: fewer distinct values but still ordered/numeric.
DISCRETE_COUNT_FEATURES = ["feature6", "feature8", "feature10", "feature12", "feature17"]

# Categorical/binary: low cardinality, codes rather than measurements.
CATEGORICAL_FEATURES = [
    "feature3", "feature4", "feature5", "feature7", "feature9",
    "feature11", "feature13", "feature14", "feature15", "feature16", "feature18",
]

# feature1 vs. claim rate is curved (EDA: 0.33% to 9% across deciles), not a
# straight line, so these let a linear log-odds model fit that curve.
FEATURE1_SQUARED_COLUMN = "feature1_squared"
FEATURE1_CUBED_COLUMN = "feature1_cubed"
ENGINEERED_FEATURES = [FEATURE1_SQUARED_COLUMN, FEATURE1_CUBED_COLUMN]

NUMERIC_FEATURES = CONTINUOUS_FEATURES + DISCRETE_COUNT_FEATURES + ENGINEERED_FEATURES
ALL_FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Categories with fewer rows than this get bucketed into "infrequent" by the encoder.
RARE_CATEGORY_MIN_FREQUENCY = 30

# Fixed seed so splits/CV folds are reproducible across runs and teammates.
RANDOM_SEED = 42
