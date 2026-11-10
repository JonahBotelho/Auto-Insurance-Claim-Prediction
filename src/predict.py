"""
Train a model on the training set, report training MAE, and write predictions.

Run from the repository folder:
    python src/predict.py                   # zero baseline -> submissions/1_2_1.csv
    python src/predict.py --model lasso     # lasso -> submissions/1_2_2.csv
    python src/predict.py --input data/testset.csv --output submissions/1_2_3.csv

The training MAE printed here goes in the upload comments for each submission.
"""

import argparse
from pathlib import Path

from sklearn.metrics import mean_absolute_error

from data_io import load_feature_set, load_training_set, write_predictions
from lasso_claim_model import DEFAULT_ALPHA, build_lasso_claim_model
from zero_claim_model import ZeroClaimModel

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAINING_PATH = REPOSITORY_ROOT / "data" / "trainingset.csv"
DEFAULT_INPUT_PATH = REPOSITORY_ROOT / "data" / "testset.csv"
SUBMISSIONS_FOLDER = REPOSITORY_ROOT / "submissions"

# Default checkpoint file for each model.
# Name format: checkpointnumber_groupnumber_submissionnumber.csv
DEFAULT_OUTPUT_NAMES = {"zero": "1_2_1.csv", "lasso": "1_2_2.csv"}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict insurance claim amounts.")
    parser.add_argument("--model", choices=DEFAULT_OUTPUT_NAMES, default="zero")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Lasso regularization strength (lasso model only).")
    parser.add_argument("--training", type=Path, default=DEFAULT_TRAINING_PATH)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=None,
                        help="Defaults to submissions/<checkpoint file for the model>.")
    return parser.parse_args()


def build_model(arguments: argparse.Namespace):
    if arguments.model == "lasso":
        return build_lasso_claim_model(alpha=arguments.alpha)
    return ZeroClaimModel()


def main() -> None:
    arguments = parse_arguments()
    output_path = arguments.output or SUBMISSIONS_FOLDER / DEFAULT_OUTPUT_NAMES[arguments.model]

    training_features, training_claims = load_training_set(arguments.training)
    model = build_model(arguments).fit(training_features, training_claims)

    training_mae = mean_absolute_error(training_claims, model.predict(training_features))
    print(f"Model: {arguments.model}")
    print(f"Training MAE: {training_mae:.4f}")

    row_indexes, input_features = load_feature_set(arguments.input)
    write_predictions(row_indexes, model.predict(input_features), output_path)
    print(f"Wrote {len(row_indexes)} predictions to {output_path}")


if __name__ == "__main__":
    main()
