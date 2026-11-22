"""Fit profile-wise model weights and a boundary-focused ranking correction."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.profile_claim_presence import group_frame, presence_forecast
from src.feature_schema import RANDOM_SEED
from src.payout_source_calibration import PayoutSourceCalibration


def expert_forecasts(train, query):
    score = query.probability.to_numpy()
    payout = PayoutSourceCalibration().fit(train, train.probability.to_numpy()).predict(query, score)[0]
    groups = group_frame(query.assign(probability_template_expected_f1=score))
    presence = presence_forecast(group_frame(train), groups)
    keys = pd.MultiIndex.from_frame(query[["profile", "fold"]])
    peers = groups.peers.reindex(keys).to_numpy()
    return (score + payout) / 2, presence.reindex(keys).to_numpy(), peers


def head_forecasts(data):
    predictions = np.empty((len(data), 3))
    # Every forecast scored by the gate comes from experts whose fitting data
    # excludes the receiving profile, including any of its sibling rows.
    splitter = GroupKFold(4, shuffle=True, random_state=RANDOM_SEED)
    for fitting, held in splitter.split(data, groups=data.profile):
        first, second, peers = expert_forecasts(data.iloc[fitting], data.iloc[held].drop(columns=["actual", "claim"]))
        predictions[held] = np.column_stack([first, second, peers])
    return predictions.T


def gate_inputs(rows, first, second, peers):
    return np.column_stack([
        np.log1p(rows.prior_positives), np.log1p(rows.prior_zeros), np.log1p(peers),
        rows.scaled_only, rows.whole_only, rows.copies_complete, rows.sum_attested,
        rows.log_source_ratio, rows.gap_fraction, rows.price_log, rows.exposure, rows.vehicle_age_code,
        np.clip((first + second) / 2, .03, .88), second - first,
    ]).astype(float)


class ProfileRankGate:
    def fit(self, rows):
        known = rows.prior_positives.gt(0)
        first, second, peers = head_forecasts(rows)
        first, second, peers = first[known], second[known], peers[known]
        known_rows = rows.loc[known]
        labels = known_rows.actual.gt(0).to_numpy(dtype=float)
        features = gate_inputs(known_rows, first, second, peers)
        self.scale_ = StandardScaler().fit(features)
        inputs = self.design(features)
        focal = .16 + np.exp(-(((first + second) / 2 - .25) / .12) ** 2)
        self.gate_ = self.fit_blend(inputs, first, second, labels, focal)
        self.rank_ = self.fit_rank(inputs, (first + second) / 2, labels, focal)
        return self

    def design(self, features):
        return np.column_stack([np.ones(len(features)), np.clip(self.scale_.transform(features), -5, 5)])

    @staticmethod
    def fit_blend(inputs, first, second, labels, weight):
        weight = weight / weight.sum()
        difference = second - first

        def estimate(coefficient):
            gate = expit(inputs @ coefficient)
            raw_score = first + gate * difference
            score = np.clip(raw_score, .001, .999)
            loss = -np.dot(weight, labels * np.log(score) + (1 - labels) * np.log1p(-score))
            derivative = weight * (score - labels) / (score * (1 - score)) * difference * gate * (1 - gate)
            derivative[raw_score != score] = 0.
            return loss + .002 * np.dot(coefficient, coefficient), inputs.T @ derivative + .004 * coefficient

        return fit_objective(estimate, inputs.shape[1])

    @staticmethod
    def fit_rank(inputs, probability, labels, weight):
        base = logit(np.clip(probability, .003, .997))
        adjusted_weight = weight / weight.sum()
        rng = np.random.default_rng(RANDOM_SEED)
        selected = [np.flatnonzero((labels == value) & (probability >= .12) & (probability <= .40)) for value in (0, 1)]
        if any(len(index) == 0 for index in selected):
            return np.zeros(inputs.shape[1])
        negative, positive = [rng.choice(index, size=6 * len(inputs), replace=True) for index in selected]
        pair_inputs, pair_base = inputs[positive] - inputs[negative], base[positive] - base[negative]

        # Compare retrieval pairs only where both profiles could cross the
        # eventual decision boundary. The pointwise term protects calibration.
        def estimate(coefficient):
            pointwise = base + inputs @ coefficient
            margin = pair_base + pair_inputs @ coefficient
            calibration = np.dot(adjusted_weight, np.logaddexp(0., pointwise) - labels * pointwise)
            ranking = np.logaddexp(0., -margin).mean()
            loss = calibration + .55 * ranking + .035 * np.dot(coefficient, coefficient)
            score_gradient = inputs.T @ (adjusted_weight * (expit(pointwise) - labels))
            rank_gradient = pair_inputs.T @ (-expit(-margin)) / len(margin)
            return loss, score_gradient + .55 * rank_gradient + .07 * coefficient

        return fit_objective(estimate, inputs.shape[1])

    def predict(self, rows, first, second, peers):
        known = rows.prior_positives.gt(0).to_numpy()
        result = (first + second) / 2
        if not known.any():
            return result.copy(), result
        features = gate_inputs(rows.loc[known], first[known], second[known], peers[known])
        design = self.design(features)
        pooled, ranked = result.copy(), result.copy()
        pooled[known] = first[known] + expit(design @ self.gate_) * (second[known] - first[known])
        ranked[known] = expit(logit(np.clip(result[known], .003, .997)) + design @ self.rank_)
        return pooled, ranked


def fit_objective(objective, size):
    fitted = minimize(objective, np.zeros(size), method="L-BFGS-B", jac=True)
    if not fitted.success:
        raise RuntimeError(f"Profile rank fit failed. {fitted.message}")
    return fitted.x
