"""
Splitting and scoring utilities used to evaluate the two-stage model
before trusting it with real predictions. Stratifying on has_claim
keeps the ~95/5 zero-claim imbalance consistent across every split,
since a random split could otherwise leave a fold with almost no
real claims to validate against.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.feature_schema import DEFAULT_CV_FOLDS, DEFAULT_EVALUATION_PERCENT, RANDOM_SEED


def stratified_train_validation_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = DEFAULT_EVALUATION_PERCENT / 100
):
    has_claim = (y > 0).astype(int)
    return train_test_split(
        X, y, test_size=test_size, stratify=has_claim, random_state=RANDOM_SEED
    )


def validate_evaluation_partitions(y_train, y_val, n_splits):
    for label, target in (("training", y_train), ("evaluation", y_val)):
        counts = (target > 0).astype(int).value_counts()
        if len(counts) != 2:
            raise ValueError(f"The {label} partition needs both claim classes; "
                             "adjust the evaluation percentage or provide more examples.")
        if label == "training" and counts.min() < n_splits:
            raise ValueError(f"CV folds ({n_splits}) exceed the smallest training class "
                             f"({counts.min()} examples); reduce CV folds or the evaluation percentage.")


def cross_validate_mae(
    model, X: pd.DataFrame, y: pd.Series, n_splits: int = DEFAULT_CV_FOLDS
) -> np.ndarray:
    has_claim = (y > 0).astype(int)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(
        model, X, y, cv=folds.split(X, has_claim), scoring="neg_mean_absolute_error",
        error_score="raise",
    )
    return -scores
