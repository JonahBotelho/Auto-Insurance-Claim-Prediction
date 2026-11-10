"""
Factory functions that build the default classifier and regressor
pipelines for the two-stage claim model. Each one chains the shared
preprocessing step directly into a model, so callers get a single
object that handles raw features in and predictions out.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.feature_schema import RANDOM_SEED
from src.preprocessing import build_preprocessing_pipeline


def build_claim_classifier_pipeline() -> Pipeline:
    # max_depth/min_samples_leaf keep the tree from memorizing individual
    # training rows, while still letting it isolate rare high-risk feature
    # combinations that a linear model spreads too thinly to separate out.
    classifier_model = DecisionTreeClassifier(
        max_depth=5, min_samples_leaf=50, random_state=RANDOM_SEED
    )
    return Pipeline([
        ("preprocessing", build_preprocessing_pipeline()),
        ("model", classifier_model),
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
