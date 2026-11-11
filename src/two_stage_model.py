"""
Classifier presets and log-target Ridge severity for the two-stage claim
model. Pipelines handle raw features, with native categorical support
instead of shared preprocessing for histogram gradient boosting.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from src.feature_schema import (
    DEFAULT_MODEL_ID, EXTRA_TREES_MODEL_ID, HIST_GRADIENT_BOOSTING_MODEL_ID,
    RANDOM_FOREST_MODEL_ID, RANDOM_SEED,
)
from src.preprocessing import build_preprocessing_pipeline


def _build_claim_classifier(model_id):
    if model_id == DEFAULT_MODEL_ID:
        return LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    if model_id in (RANDOM_FOREST_MODEL_ID, EXTRA_TREES_MODEL_ID):
        classifier_type = (
            RandomForestClassifier if model_id == RANDOM_FOREST_MODEL_ID else ExtraTreesClassifier
        )
        return classifier_type(
            n_estimators=200, min_samples_leaf=20, max_features="sqrt",
            n_jobs=-1, random_state=RANDOM_SEED,
        )
    if model_id == HIST_GRADIENT_BOOSTING_MODEL_ID:
        return HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.1, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=1.0, early_stopping="auto",
            categorical_features="from_dtype", random_state=RANDOM_SEED,
        )
    raise ValueError(f"Unknown claim model: {model_id}")


def build_claim_classifier_pipeline(model_id=DEFAULT_MODEL_ID) -> Pipeline:
    classifier = _build_claim_classifier(model_id)
    if model_id == HIST_GRADIENT_BOOSTING_MODEL_ID:
        # Preserve the loader's pandas category dtypes so native categorical
        # splits can be learned independently inside each training/CV fit.
        return Pipeline([("model", classifier)])
    return Pipeline([
        ("preprocessing", build_preprocessing_pipeline()),
        ("model", classifier),
    ])


def build_claim_severity_pipeline() -> Pipeline:
    # log1p/expm1 directly operationalizes the EDA finding that nonzero
    # claim amounts are heavily right-skewed (skew 16.44, fixed to -0.52
    # in log space), so Ridge is fit on a near-symmetric target.
    severity_model = TransformedTargetRegressor(
        regressor=Ridge(alpha=1.0),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    return Pipeline([
        ("preprocessing", build_preprocessing_pipeline()),
        ("model", severity_model),
    ])


def build_two_stage_model(model_id, claim_probability_threshold):
    """Build fresh estimators for each training run and threshold candidate."""
    return TwoStageClaimModel(
        classifier=build_claim_classifier_pipeline(model_id),
        claim_probability_threshold=claim_probability_threshold,
    )


class TwoStageClaimModel(BaseEstimator, RegressorMixin):
    """
    Predicts claim amount as P(claim) * severity-if-claim, learned as two
    separate models trained independently. Below
    claim_probability_threshold, predicts 0 directly instead of a
    probability-weighted amount.
    """

    def __init__(self, classifier=None, regressor=None, claim_probability_threshold=0.0):
        self.classifier = classifier
        self.regressor = regressor
        self.claim_probability_threshold = claim_probability_threshold

    def fit(self, X, y):
        self.classifier_ = self.classifier or build_claim_classifier_pipeline()
        self.regressor_ = self.regressor or build_claim_severity_pipeline()

        has_claim = (y > 0).astype(int)
        self.classifier_.fit(X, has_claim)

        claimed_rows = y > 0
        self.regressor_.fit(X[claimed_rows], y[claimed_rows])
        return self

    def predict(self, X):
        claim_probability = self.classifier_.predict_proba(X)[:, 1]
        severity = self.regressor_.predict(X)
        expected_amount = np.maximum(claim_probability * severity, 0)
        return np.where(
            claim_probability >= self.claim_probability_threshold,
            expected_amount,
            0.0,
        )
