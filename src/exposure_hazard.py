"""Fit claim occurrence and payout-source probabilities from policy exposure."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

from src.claim_rating_features import CATEGORIES, actuarial_features
from src.dashboard_model import profile_keys
from src.payout_pricing import whole_cent


def profile_marks(features, amounts):
    amounts = np.asarray(amounts)
    positive = amounts > 0
    cent = positive & whole_cent(amounts)
    outcomes = np.column_stack([~positive, cent, positive & ~cent])
    keys, first, inverse = np.unique(profile_keys(features), return_index=True, return_inverse=True)
    counts = np.column_stack([np.bincount(inverse, weights=column) for column in outcomes.T])
    return first, counts, pd.DataFrame(counts, index=keys)


def hazard_inputs(features, zeros, peers):
    return actuarial_features(features, zeros).assign(scoring_peers=np.log1p(peers))


class ExposureDesign:
    def __init__(self, flexible):
        self.flexible = flexible

    @staticmethod
    def exposure(frame):
        return np.clip(frame.feature1.to_numpy(), .005, 1.)

    def fit(self, frame):
        numeric = [name for name in frame if name not in CATEGORIES and name != "feature1" and frame[name].nunique() > 1]
        self.rating_ = ColumnTransformer([
            ("ratings", SplineTransformer(n_knots=4, degree=2, include_bias=False), numeric),
            ("categories", OneHotEncoder(handle_unknown="ignore", min_frequency=80, sparse_output=False), list(CATEGORIES)),
        ]).fit(frame)
        if self.flexible:
            self.duration_ = SplineTransformer(n_knots=5, degree=2, include_bias=False).fit(
                np.log(self.exposure(frame))[:, None],
            )
        return self

    def transform(self, frame):
        exposure = self.exposure(frame)
        columns = [np.ones(len(frame)), self.rating_.transform(frame)]
        if self.flexible:
            shape = self.duration_.transform(np.log(exposure)[:, None])
            # Duration patterns can vary by observed coding and batch size.
            # Regularization keeps these deviations smooth in sparse regions.
            strata = np.column_stack([frame.duration_form.eq(1), frame.duration_form.eq(2), frame.scoring_peers / 2.])
            columns.extend([shape, (shape[:, :, None] * strata[:, None, :]).reshape(len(frame), -1)])
        return np.column_stack(columns), np.log(exposure)[:, None]


class MarkedExposureHazard:
    """Model observed positive marks conditional on an exposure-scaled event rate."""

    def __init__(self, flexible):
        self.flexible = flexible

    def fit_profiles(self, features, amounts):
        first, counts, _ = profile_marks(features, amounts)
        inputs = hazard_inputs(features.iloc[first], np.zeros(len(first)), counts.sum(axis=1))
        return self.fit(inputs, counts)

    def predict_profiles(self, features):
        keys = pd.Index(profile_keys(features))
        peers = keys.map(keys.value_counts()).to_numpy()
        return self.predict(hazard_inputs(features, np.zeros(len(features)), peers))

    def fit(self, frame, count):
        self.design_ = ExposureDesign(self.flexible).fit(frame)
        matrix, offset = self.design_.transform(frame)
        self.coefficients_ = self.fit_risk(matrix, offset, count)
        return self

    @staticmethod
    def fit_risk(matrix, offset, count):
        negative, marks = count[:, 0], count[:, 1:]
        positive = marks.sum(axis=1)
        volume = count.sum()
        width = matrix.shape[1]
        penalty = np.full((width, 1), .002)
        penalty[0] = 0.
        initial = np.zeros((width, 2))
        initial[0] = np.log(np.maximum(.00001, marks.sum(axis=0) / np.dot(np.exp(offset[:, 0]), count.sum(axis=1))))

        def likelihood(weights):
            coefficients = weights.reshape(width, 2)
            raw = matrix @ coefficients + offset
            log_rate = np.clip(raw, -24., 5.)
            rate = np.exp(log_rate)
            total = rate.sum(axis=1)
            event = -np.expm1(-total)
            loss = (np.dot(negative, total) - np.dot(positive, np.log(event / total)) - np.sum(marks * log_rate)) / volume
            # Probability of observing a particular mark is the event
            # probability times that mark's share of the total hazard.
            slope = 1. - total * np.exp(-total) / event
            error = negative[:, None] * rate + positive[:, None] * (rate / total[:, None]) * slope[:, None] - marks
            error[raw != log_rate] = 0.
            difference = coefficients[:, 0] - coefficients[:, 1]
            loss += .5 * np.sum(penalty * coefficients ** 2) + .003 * np.dot(difference[1:], difference[1:])
            gradient = matrix.T @ error / volume + penalty * coefficients
            gradient[1:, 0] += .006 * difference[1:]
            gradient[1:, 1] -= .006 * difference[1:]
            return float(loss), gradient.ravel()

        fitted = minimize(likelihood, initial.ravel(), jac=True, method="L-BFGS-B", options={"maxiter": 125})
        if not np.isfinite(fitted.fun) or (not fitted.success and fitted.status != 1):
            raise RuntimeError(f"Exposure hazard failed to fit. {fitted.message}")
        return fitted.x.reshape(width, 2)

    def predict(self, frame):
        matrix, offset = self.design_.transform(frame)
        rate = np.exp(np.clip(matrix @ self.coefficients_ + offset, -24., 5.))
        total = rate.sum(axis=1)
        event = -np.expm1(-total)
        return np.column_stack([np.exp(-total), rate * (event / total)[:, None]])
