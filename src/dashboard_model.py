"""Serializable claim estimator covering every classifier and amount rule in the dashboard."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from src.dashboard_classifiers import build_dashboard_classifier, prepare_dashboard_features
from src.feature_schema import (
    ALL_FEATURE_COLUMNS, DASHBOARD_CLASSIFIER_LABELS, DASHBOARD_SEVERITY_LABELS,
    PROFILE_HISTORY_COLUMNS, PROFILE_LOG_AMOUNT_COLUMN, PROFILE_LOG_AMOUNT_VARIANCE_COLUMN,
    PROFILE_WHOLE_CENT_SHARE_COLUMN, RANDOM_SEED,
)
from src.oversampling import correct_claim_probabilities, oversample_claims
from src.profile_history import ProfileHistory
from src.two_stage_model import build_claim_severity_pipeline


def profile_keys(features):
    return pd.util.hash_pandas_object(features[ALL_FEATURE_COLUMNS], index=False).to_numpy()


def _extended_amount_statistics(amounts):
    # Some amounts are stored to more than two decimals. Use an absolute
    # tolerance to recognize whole cents despite floating-point representation.
    whole_cents = (amounts > 0) & (np.abs(amounts - np.round(amounts, 2)) < 1e-7)
    return pd.DataFrame({
        PROFILE_WHOLE_CENT_SHARE_COLUMN: whole_cents.astype(float),
        PROFILE_LOG_AMOUNT_VARIANCE_COLUMN: np.log1p(amounts) ** 2,
    })


class DashboardClaimModel(RegressorMixin, BaseEstimator):
    def __init__(self, classifier_settings, claim_probability_threshold=.1, severity_method="weighted_profile",
                 ridge_alpha=.01, severity_scale=1., amount_cap=None, random_seed=RANDOM_SEED,
                 median_payout_pricing=False, aggregate_payout_pricing=False):
        self.classifier_settings = classifier_settings
        self.claim_probability_threshold = claim_probability_threshold
        self.severity_method = severity_method
        self.ridge_alpha = ridge_alpha
        self.severity_scale = severity_scale
        self.amount_cap = amount_cap
        self.random_seed = random_seed
        self.median_payout_pricing = median_payout_pricing
        self.aggregate_payout_pricing = aggregate_payout_pricing

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
        self.payout_pricer_ = None
        self.aggregate_payout_model_ = None
        if getattr(self, "median_payout_pricing", False):
            from src.payout_pricing import MedianPayoutPricer
            self.payout_pricer_ = MedianPayoutPricer().fit(features, amounts)
        if getattr(self, "aggregate_payout_pricing", False):
            from src.aggregate_payout_choices import AggregatePayoutChoices
            self.aggregate_payout_model_ = AggregatePayoutChoices().fit(features, amounts)
        self.n_features_in_ = features.shape[1]
        return self

    def _fit_amount_history(self, features, amounts):
        keys = profile_keys(features)
        log_amount = np.log1p(amounts)
        self.log_amount_sums_ = pd.Series(log_amount).groupby(keys).sum()
        self.log_amount_prior_ = log_amount[amounts > 0].mean()
        if self.classifier_settings.get("extended_amount_history"):
            self.extended_amount_sums_ = _extended_amount_statistics(amounts).groupby(keys).sum()
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
        keys = profile_keys(features)
        total = self.log_amount_sums_.reindex(keys).fillna(0).to_numpy()
        if own_amounts is not None:
            total = total - np.log1p(own_amounts)
        count = result[PROFILE_HISTORY_COLUMNS[1]].to_numpy()
        result[PROFILE_LOG_AMOUNT_COLUMN] = np.divide(
            total, count, out=np.full(len(features), self.log_amount_prior_), where=count > 0,
        )
        if self.classifier_settings.get("extended_amount_history"):
            self._add_extended_amount_history(result, keys, count, own_amounts)
        return result

    def _add_extended_amount_history(self, result, keys, count, own_amounts):
        totals = self.extended_amount_sums_.reindex(keys).fillna(0).to_numpy(copy=True)
        if own_amounts is not None:
            totals -= _extended_amount_statistics(own_amounts).to_numpy()
        denominator = np.maximum(1, count)
        result[PROFILE_WHOLE_CENT_SHARE_COLUMN] = totals[:, 0] / denominator
        mean_log = result[PROFILE_LOG_AMOUNT_COLUMN].to_numpy()
        variance = np.maximum(0, totals[:, 1] / denominator - mean_log ** 2)
        # Cancellation after removing a row can leave tiny differences for
        # identical histories. Round so trees cannot split on that noise.
        result[PROFILE_LOG_AMOUNT_VARIANCE_COLUMN] = np.where(count >= 2, np.round(variance, 8), 0)
        result[PROFILE_LOG_AMOUNT_COLUMN] = result[PROFILE_LOG_AMOUNT_COLUMN].round(9)

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

    def predict_components(self, features):
        """Return claim probabilities and scaled amounts before the cutoff."""
        probability = self.predict_claim_probability(features)
        amount = self._predict_amount(features, probability)
        # Caps apply to the base amount before its multiplier, matching saved runs.
        if self.amount_cap is not None:
            amount = np.minimum(self.amount_cap, amount)
        amount = amount * self.severity_scale
        pricer = getattr(self, "payout_pricer_", None)
        if pricer is not None:
            amount = pricer.predict(features, amount)
        aggregate = getattr(self, "aggregate_payout_model_", None)
        if aggregate is not None:
            _, amount = aggregate.predict(features, probability, amount)
        return probability, amount

    def predict(self, features):
        probability, amount = self.predict_components(features)
        return np.where(probability >= self.claim_probability_threshold, amount, 0.)
