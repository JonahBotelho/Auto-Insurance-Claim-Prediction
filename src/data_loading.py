"""
Loads the training/test CSVs and derives the binary has-claim label
used to train the classification stage of the two-stage model.
"""

from pathlib import Path

import pandas as pd

from src.feature_schema import (
    CATEGORICAL_FEATURES,
    FEATURE1_CUBED_COLUMN,
    FEATURE1_SQUARED_COLUMN,
    HAS_CLAIM_COLUMN,
    PROFILE_COLUMNS,
    ROW_INDEX_COLUMN,
    TARGET_COLUMN,
)

DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data"
DEFAULT_TRAINING_PATH = DATA_DIRECTORY / "trainingset.csv"
DEFAULT_TEST_PATH = DATA_DIRECTORY / "testset.csv"
REQUIRED_PREDICTION_COLUMNS = (ROW_INDEX_COLUMN, *PROFILE_COLUMNS)


def _add_feature1_polynomial_terms(data: pd.DataFrame) -> pd.DataFrame:
    data[FEATURE1_SQUARED_COLUMN] = data["feature1"] ** 2
    data[FEATURE1_CUBED_COLUMN] = data["feature1"] ** 3
    return data


def _prepare_typed_features(data: pd.DataFrame) -> pd.DataFrame:
    # Cast explicitly so categorical columns are never mistaken for plain
    # numbers downstream (e.g. accidentally scaled instead of one-hot encoded).
    for column in CATEGORICAL_FEATURES:
        data[column] = data[column].astype("category")
    return _add_feature1_polynomial_terms(data)


def _read_csv_with_typed_categories(path: Path) -> pd.DataFrame:
    return _prepare_typed_features(pd.read_csv(path))


def load_test_data_from_buffer(buffer) -> pd.DataFrame:
    """Load and validate an uploaded CSV stream using the exact inference preprocessing."""
    data = pd.read_csv(buffer)
    missing = [column for column in REQUIRED_PREDICTION_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")
    if data.empty:
        raise ValueError("The uploaded CSV contains no data rows.")
    return _prepare_typed_features(data)


def load_training_data(path: Path = DEFAULT_TRAINING_PATH) -> pd.DataFrame:
    return _read_csv_with_typed_categories(path)


def load_test_data(path: Path = DEFAULT_TEST_PATH) -> pd.DataFrame:
    return _read_csv_with_typed_categories(path)


def add_has_claim_label(training_data: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COLUMN not in training_data.columns:
        raise ValueError(f"'{TARGET_COLUMN}' column is required to derive '{HAS_CLAIM_COLUMN}'")
    labeled_data = training_data.copy()
    labeled_data[HAS_CLAIM_COLUMN] = (labeled_data[TARGET_COLUMN] > 0).astype(int)
    return labeled_data
