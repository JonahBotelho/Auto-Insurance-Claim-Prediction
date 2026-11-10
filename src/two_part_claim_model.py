"""
Two-part claim model: first classify claim vs. no claim, then predict an amount.

Why this shape: 95% of customers claim $0. Under MAE, the best single
prediction for a customer is the median of their possible claim amounts.
If the classifier says a customer claims with probability p:
- p < 0.5: the median is $0, so predict $0
- p >= 0.5: the median is a claim, so predict a typical claim amount
The threshold is a parameter so cross-validation can check other values.

The predicted amount is the median claim among training customers who
claimed. A separate amount model could replace it later.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone


class TwoPartClaimModel(RegressorMixin, BaseEstimator):
    """Wraps any scikit-learn classifier that has predict_proba."""

    def __init__(self, classifier, threshold: float = 0.5):
        # scikit-learn's clone() rebuilds the model from these attributes
        # during cross-validation, so store them unchanged.
        self.classifier = classifier
        self.threshold = threshold

    def fit(self, features, claim_amounts):
        claim_amounts = np.asarray(claim_amounts)
        made_claim = claim_amounts > 0
        self.classifier_ = clone(self.classifier).fit(features, made_claim)
        self.claim_amount_ = float(np.median(claim_amounts[made_claim]))
        return self

    def predict_claim_probability(self, features) -> np.ndarray:
        # Column 1 is the probability of the True class (made a claim).
        return self.classifier_.predict_proba(features)[:, 1]

    def predict(self, features) -> np.ndarray:
        is_likely_claim = self.predict_claim_probability(features) >= self.threshold
        return np.where(is_likely_claim, self.claim_amount_, 0.0)
