"""Serializable claim estimator covering every classifier and amount rule in the dashboard."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from src.dashboard_classifiers import build_dashboard_classifier, prepare_dashboard_features
from src.feature_schema import (
    ALL_FEATURE_COLUMNS, DASHBOARD_CLASSIFIER_LABELS, DASHBOARD_SEVERITY_LABELS,
    PROFILE_HISTORY_COLUMNS, PROFILE_LOG_AMOUNT_COLUMN, RANDOM_SEED,
)
from src.oversampling import correct_claim_probabilities, oversample_claims
from src.profile_history import ProfileHistory
from src.two_stage_model import build_claim_severity_pipeline


def profile_keys(features):
    return pd.util.hash_pandas_object(features[ALL_FEATURE_COLUMNS], index=False).to_numpy()


class DashboardClaimModel(RegressorMixin, BaseEstimator):
    def __init__(self, classifier_settings, claim_probability_threshold=.1, severity_method="weighted_profile",
                 ridge_alpha=.01, severity_scale=1., amount_cap=None, random_seed=RANDOM_SEED):
        self.classifier_settings = classifier_settings
        self.claim_probability_threshold = claim_probability_threshold
        self.severity_method = severity_method
        self.ridge_alpha = ridge_alpha
        self.severity_scale = severity_scale
        self.amount_cap = amount_cap
        self.random_seed = random_seed

    def fit(self, features, amounts):
        settings = self.classifier_settings
        if settings["family"] not in DASHBOARD_CLASSIFIER_LABELS or self.severity_method not in DASHBOARD_SEVERITY_LABELS:
            raise ValueError("Unsupported dashboard model or severity rule.")
        amounts = np.asarray(amounts)
        claims = (amounts > 0).astype(int)
        self.history_ = ProfileHistory(settings["smoothing"]).fit(features, amounts)
        self._fit_amount_history(features, amounts)
        classifier_features = self.history_.transform(features, own_claim=claims)
        if settings.get("amount_history"):
            classifier_features = self._add_amount_history(features, classifier_features, amounts)
        self.classifier_ = None
        if settings["family"] != "posterior":
            prepared = prepare_dashboard_features(classifier_features, settings["family"])
            sampled, labels, self.prior_ratio_ = oversample_claims(prepared, claims, settings["oversampling"])
            self.classifier_ = build_dashboard_classifier(settings, self.random_seed)
            self.classifier_.fit(sampled, labels)
        self.regressor_ = build_claim_severity_pipeline(ridge_alpha=self.ridge_alpha)
        self.regressor_.fit(features.loc[amounts > 0], amounts[amounts > 0])
        self.n_features_in_ = features.shape[1]
        return self

    def _fit_amount_history(self, features, amounts):
        keys = profile_keys(features)
        log_amount = np.log1p(amounts)
        self.log_amount_sums_ = pd.Series(log_amount).groupby(keys).sum()
        self.log_amount_prior_ = log_amount[amounts > 0].mean()
        positive = pd.DataFrame({"key": keys[amounts > 0], "log": np.log(amounts[amounts > 0])})
        grouped = positive.groupby("key")["log"]
        self.log_medians_ = grouped.agg(lambda values: np.median(values.to_numpy()))
        residuals = []
        # Exclude each claim when estimating the within-profile amount noise.
        for _, values in grouped:
            values = values.to_numpy()
            if len(values) >= 2:
                residuals.extend(values[i] - np.median(np.delete(values, i)) for i in range(len(values)))
        self.residuals_ = np.sort(np.asarray(residuals))
        self.positive_amounts_ = np.sort(amounts[amounts > 0])

    def _add_amount_history(self, features, classifier_features, own_amounts=None):
        result = classifier_features.copy()
        total = self.log_amount_sums_.reindex(profile_keys(features)).fillna(0).to_numpy()
        if own_amounts is not None:
            total = total - np.log1p(own_amounts)
        count = result[PROFILE_HISTORY_COLUMNS[1]].to_numpy()
        result[PROFILE_LOG_AMOUNT_COLUMN] = np.divide(
            total, count, out=np.full(len(features), self.log_amount_prior_), where=count > 0,
        )
        return result

    def predict_claim_probability(self, features):
        settings = self.classifier_settings
        prepared = self.history_.transform(features)
        if settings.get("amount_history"):
            prepared = self._add_amount_history(features, prepared)
        if settings["family"] == "posterior":
            return prepared[PROFILE_HISTORY_COLUMNS[2]].to_numpy()
        prepared = prepare_dashboard_features(prepared, settings["family"])
        return correct_claim_probabilities(self.classifier_.predict_proba(prepared)[:, 1], self.prior_ratio_)

    def _predict_amount(self, features, probability):
        fallback = np.maximum(0, self.regressor_.predict(features))
        median = self.history_.predict_amount(features, fallback)
        if self.severity_method == "weighted_ridge":
            return probability * fallback
        if self.severity_method == "weighted_profile":
            return probability * median
        if self.severity_method == "profile_median":
            return median
        quantile = np.maximum(.05, (probability - .5) / np.maximum(probability, 1e-9))
        global_amount = np.quantile(self.positive_amounts_, quantile)
        if not len(self.residuals_):
            return global_amount
        log_median = self.log_medians_.reindex(profile_keys(features)).to_numpy()
        local = np.exp(log_median + np.quantile(self.residuals_, quantile))
        return np.where(np.isfinite(local), local, global_amount)

    def predict(self, features):
        probability = self.predict_claim_probability(features)
        amount = self._predict_amount(features, probability)
        # Caps apply to the base amount before its multiplier, matching saved runs.
        if self.amount_cap is not None:
            amount = np.minimum(self.amount_cap, amount)
        return np.where(probability >= self.claim_probability_threshold, amount * self.severity_scale, 0.)
