"""Fit odds corrections based on the observed cents of partial payouts."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.preprocessing import StandardScaler

from src.payout_residue_evidence import COLUMNS


def residue_inputs(rows, interactions):
    values = rows.loc[:, list(COLUMNS[1:])].to_numpy()
    if interactions:
        return np.c_[values, values * rows.gap_fraction.to_numpy()[:, None]]
    return values


def fit_residue_shift(fitting, probabilities, *, interactions=False):
    inputs = residue_inputs(fitting, interactions)
    scaler = StandardScaler().fit(inputs)
    design = np.clip(scaler.transform(inputs), -5, 5)
    y = fitting.actual.gt(0).to_numpy()
    offset = logit(np.clip(probabilities, .003, .997))
    weight = .25 + np.exp(-((probabilities - .25) / .17) ** 2)
    weight /= weight.sum()
    penalty = .045

    def objective(parameters):
        score = offset + design @ parameters
        loss = np.dot(weight, np.logaddexp(0., score) - y * score)
        gradient = design.T @ (weight * (expit(score) - y))
        return loss + penalty / 2 * np.dot(parameters, parameters), gradient + penalty * parameters

    optimum = minimize(objective, np.zeros(design.shape[1]), jac=True, method="L-BFGS-B")
    if not optimum.success:
        raise RuntimeError(f"Payout residue fit failed. {optimum.message}")
    return scaler, optimum.x


def predict_residue_shift(rows, fitted, *, interactions=False):
    scaler, weights = fitted
    query = np.clip(scaler.transform(residue_inputs(rows, interactions)), -5, 5)
    return np.clip(query @ weights, -.8, .8)


