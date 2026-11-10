"""
Unit tests for the zero-claim baseline and the prediction file format.

Run from the repository folder:
    python -m unittest discover -s src/tests
"""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

# Tests live in src/tests, so add src/ to the import path to reach the modules.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_io import write_predictions  # noqa: E402
from zero_claim_model import ZeroClaimModel  # noqa: E402


class ZeroClaimModelTests(unittest.TestCase):
    def test_predicts_zero_for_every_row(self):
        features = pd.DataFrame({"feature1": [0.1, 0.5, 0.9]})
        predictions = ZeroClaimModel().fit(features, [0, 100, 0]).predict(features)
        self.assertEqual(list(predictions), [0.0, 0.0, 0.0])


class WritePredictionsTests(unittest.TestCase):
    def test_output_matches_submission_format(self):
        with tempfile.TemporaryDirectory() as temporary_folder:
            output_path = Path(temporary_folder) / "nested" / "predictions.csv"
            write_predictions(pd.Series([0, 1, 2]), [0.0, 0.0, 12.5], output_path)

            written = pd.read_csv(output_path)
            self.assertEqual(list(written.columns), ["rowIndex", "ClaimAmount"])
            self.assertEqual(list(written["rowIndex"]), [0, 1, 2])
            self.assertEqual(list(written["ClaimAmount"]), [0.0, 0.0, 12.5])


if __name__ == "__main__":
    unittest.main()
