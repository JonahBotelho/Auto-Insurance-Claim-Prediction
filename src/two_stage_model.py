"""
Classifier presets and configurable severity for the two-stage claim model. Pipelines handle raw features, with native categorical support
instead of shared preprocessing for histogram gradient boosting.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.feature_schema import (
    DECISION_TREE_MODEL_ID, DECISION_TREE_PARAMS, DEFAULT_MODEL_ID, DEFAULT_OVERSAMPLING_PERCENT,
    EXTRA_TREES_MODEL_ID, EXTRA_TREES_PARAMS,
    HIST_GRADIENT_BOOSTING_MODEL_ID, HIST_GRADIENT_BOOSTING_PARAMS,
    LOGISTIC_REGRESSION_PARAMS, RANDOM_FOREST_MODEL_ID, RANDOM_FOREST_PARAMS, RIDGE_PARAMS,
    DEFAULT_PROFILE_HISTORY_SMOOTHING, PROFILE_HISTORY_COLUMNS, RANDOM_SEED, SEVERITY_METHOD_LABELS,
)
from src.preprocessing import build_preprocessing_pipeline
from src.oversampling import correct_claim_probabilities, oversample_claims
from src.profile_history import ProfileHistory


def _build_claim_classifier(model_id):
    if model_id == DEFAULT_MODEL_ID:
        return LogisticRegression(**LOGISTIC_REGRESSION_PARAMS)
    if model_id == DECISION_TREE_MODEL_ID:
        return DecisionTreeClassifier(**DECISION_TREE_PARAMS)
    if model_id == RANDOM_FOREST_MODEL_ID:
        return RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    if model_id == EXTRA_TREES_MODEL_ID:
        return ExtraTreesClassifier(**EXTRA_TREES_PARAMS)
    if model_id == HIST_GRADIENT_BOOSTING_MODEL_ID:
        return HistGradientBoostingClassifier(**HIST_GRADIENT_BOOSTING_PARAMS)
    raise ValueError(f"Unknown claim model: {model_id}")


def build_claim_classifier_pipeline(
    model_id=DEFAULT_MODEL_ID, min_samples_leaf=None, classifier_params=None,
    use_profile_history=False, random_seed=RANDOM_SEED,
) -> Pipeline:
    classifier = _build_claim_classifier(model_id)
    classifier.set_params(random_state=random_seed)
    if min_samples_leaf is not None:
        classifier.set_params(min_samples_leaf=min_samples_leaf)
    if classifier_params:
        classifier.set_params(**classifier_params)
    if model_id == HIST_GRADIENT_BOOSTING_MODEL_ID:
        # Preserve the loader's pandas category dtypes so native categorical
        # splits can be learned independently inside each training/CV fit.
        return Pipeline([("model", classifier)])
    preprocessing = build_preprocessing_pipeline(
        include_engineered_features=model_id != DECISION_TREE_MODEL_ID,
    )
    if use_profile_history:
        preprocessing.transformers.append(("profile_history", "passthrough", PROFILE_HISTORY_COLUMNS))
    return Pipeline([("preprocessing", preprocessing), ("model", classifier)])


def build_claim_severity_pipeline(include_engineered_features=True, ridge_alpha=None) -> Pipeline:
    # log1p/expm1 directly operationalizes the EDA finding that nonzero
    # claim amounts are heavily right-skewed (skew 16.44, fixed to -0.52
    # in log space), so Ridge is fit on a near-symmetric target.
    severity_model = TransformedTargetRegressor(
        regressor=Ridge(**(RIDGE_PARAMS if ridge_alpha is None else {"alpha": ridge_alpha})),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    return Pipeline([
        ("preprocessing", build_preprocessing_pipeline(
            include_engineered_features=include_engineered_features,
        )),
        ("model", severity_model),
    ])


def build_two_stage_model(
    model_id, claim_probability_threshold, min_samples_leaf=None,
    oversampling_percent=DEFAULT_OVERSAMPLING_PERCENT, classifier_params=None,
    use_profile_history=False, profile_history_smoothing=DEFAULT_PROFILE_HISTORY_SMOOTHING,
    severity_method="ridge", ridge_alpha=None, severity_scale=1.0, random_seed=RANDOM_SEED,
):
    """Build fresh estimators for each training run and threshold candidate."""
    return TwoStageClaimModel(
        classifier=build_claim_classifier_pipeline(
            model_id, min_samples_leaf=min_samples_leaf, classifier_params=classifier_params,
            use_profile_history=use_profile_history, random_seed=random_seed,
        ),
        # Submission 1 used the original features in both stages.
        regressor=build_claim_severity_pipeline(
            include_engineered_features=model_id != DECISION_TREE_MODEL_ID, ridge_alpha=ridge_alpha,
        ),
        claim_probability_threshold=claim_probability_threshold,
        oversampling_percent=oversampling_percent,
        use_profile_history=use_profile_history, profile_history_smoothing=profile_history_smoothing,
        severity_method=severity_method, severity_scale=severity_scale,
    )


class TwoStageClaimModel(BaseEstimator, RegressorMixin):
    """
    Predicts claim amount as P(claim) * severity-if-claim, learned as two
    separate models trained independently. Below
    claim_probability_threshold, predicts 0 directly instead of a
    probability-weighted amount.
    """

    def __init__(
        self, classifier=None, regressor=None, claim_probability_threshold=0.0,
        oversampling_percent=DEFAULT_OVERSAMPLING_PERCENT, use_profile_history=False,
        profile_history_smoothing=DEFAULT_PROFILE_HISTORY_SMOOTHING,
        severity_method="ridge", severity_scale=1.0,
    ):
        self.classifier = classifier
        self.regressor = regressor
        self.claim_probability_threshold = claim_probability_threshold
        self.oversampling_percent = oversampling_percent
        self.use_profile_history = use_profile_history
        self.profile_history_smoothing = profile_history_smoothing
        self.severity_method = severity_method
        self.severity_scale = severity_scale

    def fit(self, X, y):
        self.classifier_ = clone(self.classifier) if self.classifier is not None else build_claim_classifier_pipeline(use_profile_history=self.use_profile_history)
        self.regressor_ = clone(self.regressor) if self.regressor is not None else build_claim_severity_pipeline()

        if self.severity_method not in SEVERITY_METHOD_LABELS:
            raise ValueError("Choose an available severity method.")
        if not np.isfinite(self.severity_scale) or self.severity_scale <= 0:
            raise ValueError("Severity scale must be finite and positive.")
        if not np.isfinite(self.profile_history_smoothing) or self.profile_history_smoothing < 0:
            raise ValueError("Profile smoothing must be finite and nonnegative.")
        has_claim = (y > 0).astype(int)
        self.profile_history_ = None
        classifier_features = X
        if self.use_profile_history or self.severity_method == "profile_median":
            self.profile_history_ = ProfileHistory(self.profile_history_smoothing).fit(X, y)
        if self.use_profile_history:
            classifier_features = self.profile_history_.transform(X, own_claim=has_claim)
        # This runs inside every CV fit, before fitting classifier preprocessing.
        sampled_X, sampled_labels, self.claim_prior_odds_ratio_ = oversample_claims(
            classifier_features, has_claim, self.oversampling_percent,
        )
        self.classifier_.fit(sampled_X, sampled_labels)

        claimed_rows = y > 0
        self.regressor_.fit(X[claimed_rows], y[claimed_rows])
        return self

    def predict_claim_probability(self, X):
        features = self.profile_history_.transform(X) if self.use_profile_history else X
        probabilities = self.classifier_.predict_proba(features)[:, 1]
        # Saved models from before oversampling retain their original predictions.
        return correct_claim_probabilities(
            probabilities, getattr(self, "claim_prior_odds_ratio_", 1.0),
        )

    def predict_components(self, X):
        """Return claim probabilities and weighted amounts before the cutoff."""
        claim_probability = self.predict_claim_probability(X)
        severity = self.regressor_.predict(X)
        if self.severity_method == "profile_median":
            severity = self.profile_history_.predict_amount(X, severity)
        severity = np.maximum(severity, 0) * self.severity_scale
        expected_amount = np.maximum(claim_probability * severity, 0)
        return claim_probability, expected_amount

    def predict(self, X):
        claim_probability, expected_amount = self.predict_components(X)
        return np.where(
            claim_probability >= self.claim_probability_threshold,
            expected_amount,
            0.0,
        )

    def __setstate__(self, state):
        # Artifacts created before these options keep their original behavior.
        defaults = dict(
            oversampling_percent=0, use_profile_history=False,
            profile_history_smoothing=DEFAULT_PROFILE_HISTORY_SMOOTHING,
            severity_method="ridge", severity_scale=1.0,
        )
        self.__dict__.update({**defaults, **state})
