"""
Unit tests for the ridge, polynomial, regression tree and k-NN builders.

Run from the repository folder:
    python -m unittest discover -s src/tests
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Tests live in src/tests, so add src/ to the import path to reach the modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from claim_models import (  # noqa: E402
    build_knn_claim_model, build_polynomial_claim_model, build_ridge_claim_model,
    build_tree_claim_model,
)
from preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES  # noqa: E402


def make_sample_data(row_count: int = 200, seed: int = 0):
    """Random features shaped like the training set, with mostly-zero claims."""
    generator = np.random.default_rng(seed)
    features = pd.DataFrame({
        **{name: generator.integers(0, 5, row_count) for name in CATEGORICAL_FEATURES},
        **{name: generator.random(row_count) for name in NUMERIC_FEATURES},
    })
    has_claim = generator.random(row_count) < 0.1
    claims = np.where(has_claim, generator.uniform(100, 5000, row_count), 0.0)
    return features, pd.Series(claims)


class ClaimModelBuilderTests(unittest.TestCase):
    def assert_valid_predictions(self, model):
        features, claims = make_sample_data()
        predictions = model.fit(features, claims).predict(features)
        self.assertEqual(len(predictions), len(features))
        self.assertGreaterEqual(predictions.min(), 0.0)

    def test_ridge(self):
        self.assert_valid_predictions(build_ridge_claim_model(alpha=1.0))

    def test_polynomial(self):
        self.assert_valid_predictions(build_polynomial_claim_model(degree=2, alpha=0.01))

    def test_regression_tree_with_median_leaves(self):
        self.assert_valid_predictions(build_tree_claim_model(3, criterion="absolute_error"))

    def test_regression_tree_on_log_target(self):
        self.assert_valid_predictions(build_tree_claim_model(3, log_target=True))

    def test_knn(self):
        self.assert_valid_predictions(build_knn_claim_model(neighbor_count=5))


if __name__ == "__main__":
    unittest.main()
