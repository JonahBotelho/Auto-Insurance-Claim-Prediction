"""
Read the project CSV files and write prediction files.

Every model in this project shares these helpers, so the input and output
formats stay the same as the course's data/submission.csv template:
two columns, rowIndex and ClaimAmount.
"""

from pathlib import Path

import pandas as pd

ROW_INDEX_COLUMN = "rowIndex"
TARGET_COLUMN = "ClaimAmount"


def load_training_set(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Return (features, claim_amounts) from the labelled training CSV."""
    training_data = pd.read_csv(csv_path)
    features = training_data.drop(columns=[ROW_INDEX_COLUMN, TARGET_COLUMN])
    return features, training_data[TARGET_COLUMN]


def load_feature_set(csv_path: Path) -> tuple[pd.Series, pd.DataFrame]:
    """Return (row_indexes, features) from an unlabelled CSV such as testset.csv.

    The competition set uses the same format as the test set, so this also
    reads competitionset.csv for the final deployed model.
    """
    feature_data = pd.read_csv(csv_path)
    features = feature_data.drop(columns=[ROW_INDEX_COLUMN])
    return feature_data[ROW_INDEX_COLUMN], features


def write_predictions(row_indexes: pd.Series, predictions, csv_path: Path) -> None:
    """Write predictions in the submission.csv format, creating folders if needed."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame(
        {ROW_INDEX_COLUMN: row_indexes.to_numpy(), TARGET_COLUMN: predictions}
    )
    submission.to_csv(csv_path, index=False)
