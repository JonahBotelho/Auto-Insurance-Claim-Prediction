"""
Baseline model that predicts a claim amount of $0 for every customer.

About 95% of training customers made no claim, so the median claim is $0.
The median minimises mean absolute error (MAE), the course metric, which
makes "always $0" a strong baseline (training MAE of about 103.05).
Every later model must beat this score to be worth submitting.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin


class ZeroClaimModel(RegressorMixin, BaseEstimator):
    """Predicts $0 for every row. Inherits scikit-learn's base classes so it
    works inside cross_validate like any other model."""

    def fit(self, features, claim_amounts) -> "ZeroClaimModel":
        # Nothing to learn: the prediction is always zero.
        return self

    def predict(self, features) -> np.ndarray:
        return np.zeros(len(features))
