"""
Shared cross-validation helpers for the analysis scripts.

All model comparisons use the same grouped 5-fold split, so their MAE
numbers can be compared directly.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_validate

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAINING_PATH = REPOSITORY_ROOT / "data" / "trainingset.csv"
OUTPUT_FOLDER = REPOSITORY_ROOT / "analysis" / "output"
FOLD_COUNT = 5

# Let analysis scripts import the model code in src/.
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))


def duplicate_row_groups(features: pd.DataFrame) -> np.ndarray:
    """Give identical feature rows the same group id.

    Many training rows are exact copies. A plain K-fold split can put copies
    in both the training and validation folds, which makes the CV score look
    better than it really is. GroupKFold keeps each group in a single fold.
    """
    return pd.util.hash_pandas_object(features, index=False).to_numpy()


def cross_validate_mae(model, features, claims, groups) -> dict:
    """Return mean training MAE, mean cross-validated MAE and run time."""
    start_time = time.perf_counter()
    scores = cross_validate(
        model, features, claims, groups=groups, cv=GroupKFold(n_splits=FOLD_COUNT),
        scoring="neg_mean_absolute_error", return_train_score=True,
    )
    return {
        "training_mae": -scores["train_score"].mean(),
        "cv_mae": -scores["test_score"].mean(),
        "seconds": time.perf_counter() - start_time,
    }
