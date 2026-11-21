"""
Splitting and scoring utilities used to evaluate the two-stage model
before trusting it with real predictions. Stratifying on has_claim
keeps the ~95/5 zero-claim imbalance consistent across every split,
since a random split could otherwise leave a fold with almost no
real claims to validate against.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.feature_schema import DEFAULT_CV_FOLDS, DEFAULT_EVALUATION_PERCENT, RANDOM_SEED


def stratified_train_validation_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = DEFAULT_EVALUATION_PERCENT / 100,
    random_seed: int = RANDOM_SEED,
):
    has_claim = (y > 0).astype(int)
    return train_test_split(
        X, y, test_size=test_size, stratify=has_claim, random_state=random_seed
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


def cross_validate_thresholds(
    model, X: pd.DataFrame, y: pd.Series, thresholds, n_splits: int = DEFAULT_CV_FOLDS,
    random_seed: int = RANDOM_SEED,
):
    """Return mean fold MAE and positive-class F1 for each candidate cutoff."""
    thresholds = np.asarray(thresholds)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    mae_scores = np.empty((n_splits, len(thresholds)))
    f1_scores = np.empty_like(mae_scores)
    for fold, (train, validation) in enumerate(folds.split(X, y > 0)):
        fitted = clone(model).fit(X.iloc[train], y.iloc[train])
        probability, amount = fitted.predict_components(X.iloc[validation])
        if not np.isfinite(probability).all() or not np.isfinite(amount).all():
            raise ValueError("Cross-validation returned nonfinite predictions.")
        actual = y.iloc[validation].to_numpy()
        # The cutoff affects only the final gate. All candidates share one
        # fit and one set of out-of-fold predictions per fold.
        cutoff = thresholds
        adaptive = getattr(fitted, "adaptive_claim_threshold", None)
        if adaptive is not None:
            if len(thresholds) != 1:
                raise ValueError("Adaptive claim thresholds require a single operating policy.")
            cutoff = adaptive(probability)
        predicted = np.where(probability[:, None] >= cutoff, amount[:, None], 0)
        mae_scores[fold] = np.abs(actual[:, None] - predicted).mean(axis=0)
        f1_scores[fold] = [
            f1_score(actual > 0, candidate > 0, zero_division=0) for candidate in predicted.T
        ]
    return mae_scores.mean(axis=0), f1_scores.mean(axis=0)
