"""
Unit tests for the two-part (classify, then predict amount) claim model.

Run from the repository folder:
    python -m unittest discover -s src/tests
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Tests live in src/tests, so add src/ to the import path to reach the modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from two_part_claim_model import TwoPartClaimModel  # noqa: E402


def make_separable_data():
    """feature1 = 1 always claims $1,000 or $3,000; feature1 = 0 never claims."""
    features = pd.DataFrame({"feature1": [0, 0, 0, 0, 1, 1, 1, 1]})
    claims = pd.Series([0, 0, 0, 0, 1000, 1000, 3000, 3000])
    return features, claims


class TwoPartClaimModelTests(unittest.TestCase):
    def test_predicts_median_claim_for_likely_claimers_and_zero_otherwise(self):
        features, claims = make_separable_data()
        model = TwoPartClaimModel(DecisionTreeClassifier(random_state=0)).fit(features, claims)
        predictions = model.predict(features)
        # Median of the claims 1000, 1000, 3000, 3000 is 2000.
        self.assertEqual(list(predictions), [0, 0, 0, 0, 2000, 2000, 2000, 2000])

    def test_threshold_above_every_probability_predicts_zero(self):
        features, claims = make_separable_data()
        model = TwoPartClaimModel(DecisionTreeClassifier(), threshold=1.01).fit(features, claims)
        self.assertTrue(np.all(model.predict(features) == 0))

    def test_works_inside_scikit_learn_cross_validation(self):
        # cross_validate clones the model, which needs the constructor
        # parameters stored unchanged.
        features, claims = make_separable_data()
        scores = cross_val_score(
            TwoPartClaimModel(DecisionTreeClassifier()), features, claims, cv=2,
            scoring="neg_mean_absolute_error",
        )
        self.assertEqual(len(scores), 2)


if __name__ == "__main__":
    unittest.main()
