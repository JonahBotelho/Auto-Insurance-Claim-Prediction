"""
Unit tests for the lasso claim model pipeline.

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

from lasso_claim_model import (  # noqa: E402
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_lasso_claim_model,
)


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


class LassoClaimModelTests(unittest.TestCase):
    def test_predicts_one_value_per_row(self):
        features, claims = make_sample_data()
        model = build_lasso_claim_model(alpha=0.01).fit(features, claims)
        self.assertEqual(len(model.predict(features)), len(features))

    def test_log_target_predictions_are_not_negative(self):
        # The linear fit can go below zero on the log scale; the model must
        # floor the converted dollar amount at $0.
        features, claims = make_sample_data()
        model = build_lasso_claim_model(alpha=0.01).fit(features, claims)
        self.assertGreaterEqual(model.predict(features).min(), 0.0)

    def test_unseen_category_level_does_not_crash(self):
        features, claims = make_sample_data()
        model = build_lasso_claim_model(alpha=0.01).fit(features, claims)
        unseen = features.head(3).copy()
        unseen["feature3"] = 99
        self.assertEqual(len(model.predict(unseen)), 3)


if __name__ == "__main__":
    unittest.main()
