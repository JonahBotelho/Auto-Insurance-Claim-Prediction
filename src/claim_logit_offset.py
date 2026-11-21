"""Estimate regularized offsets to a fitted claim probability."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit_logit_offset(design, base, outcome, weight, *, penalty):
    weights = weight / weight.sum()

    def objective(coefficients):
        linear = base + design @ coefficients
        error = weights * (expit(linear) - outcome)
        loss = np.dot(weights, np.logaddexp(0., linear) - outcome * linear)
        return loss + penalty / 2 * np.dot(coefficients, coefficients), design.T @ error + penalty * coefficients

    solution = minimize(objective, np.zeros(design.shape[1]), method="L-BFGS-B", jac=True)
    if not solution.success:
        raise RuntimeError(f"Profile stability fit failed. {solution.message}")
    return solution.x


