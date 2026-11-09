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

from src.feature_schema import RANDOM_SEED


def stratified_train_validation_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    has_claim = (y > 0).astype(int)
    return train_test_split(
        X, y, test_size=test_size, stratify=has_claim, random_state=RANDOM_SEED
    )


def cross_validate_mae(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> np.ndarray:
    has_claim = (y > 0).astype(int)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_val_score(
        model, X, y, cv=folds.split(X, has_claim), scoring="neg_mean_absolute_error"
    )
    return -scores
