"""
CLI entry point: loads the trained model saved by train.py, scores
data/testset.csv, and writes predictions in the rowIndex,ClaimAmount
format matching data/submission.csv.
"""

import argparse
from pathlib import Path

from src.data_loading import DEFAULT_TEST_PATH, load_test_data
from src.feature_schema import ALL_FEATURE_COLUMNS, ROW_INDEX_COLUMN, TARGET_COLUMN
from src.model_io import DEFAULT_MODEL_PATH, load_model

DEFAULT_SUBMISSION_PATH = Path(__file__).resolve().parents[1] / "data" / "submission.csv"


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate claim predictions for a test set.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input", type=Path, default=DEFAULT_TEST_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_SUBMISSION_PATH)
    return parser.parse_args(argv)


def generate_submission(model, test_data):
    predictions = model.predict(test_data[ALL_FEATURE_COLUMNS])
    return test_data[[ROW_INDEX_COLUMN]].assign(**{TARGET_COLUMN: predictions})


def main(argv=None):
    args = parse_args(argv)
    model = load_model(args.model_path)
    test_data = load_test_data(args.input)
    submission = generate_submission(model, test_data)
    submission.to_csv(args.output, index=False)
    print(f"Wrote {len(submission)} predictions to {args.output}")
    total_claims = len(submission)
    nonzero_claims = int((submission[TARGET_COLUMN] > 0).sum())
    nonzero_percentage = 100 * nonzero_claims / total_claims if total_claims else 0.0
    print(f"Non-zero claims: {nonzero_claims}/{total_claims} ({nonzero_percentage:.4f}%)")


if __name__ == "__main__":
    main()
