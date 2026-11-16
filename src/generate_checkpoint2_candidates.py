"""Generate ten checkpoint-2 submissions from the validated HGB blend."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.dashboard_model import DashboardClaimModel
from src.data_loading import load_test_data, load_training_data
from src.feature_schema import ALL_FEATURE_COLUMNS, RANDOM_SEED, ROW_INDEX_COLUMN, TARGET_COLUMN


PRIMARY_SETTINGS = {
    "family": "hist", "iterations": 125, "rate": 0.15, "leaves": 9,
    "leaf": 50, "l2": 5.0, "oversampling": 30, "smoothing": 0.25,
    "early_stopping": "auto",
}
SECONDARY_SETTINGS = {
    "family": "hist", "iterations": 250, "rate": 0.1, "leaves": 15,
    "leaf": 75, "l2": 10.0, "oversampling": 55, "smoothing": 0.5,
    "early_stopping": "auto", "amount_history": True,
}
# The first nine candidates cover the MAE, balanced, and F1 operating points
# confirmed across seeds 42–46. Candidate 10 is a standalone F1 hedge.
CANDIDATES = (
    ("2_2_1.csv", 0.4, 0.6, 0.32),
    ("2_2_2.csv", 0.4, 0.6, 0.34),
    ("2_2_3.csv", 0.4, 0.6, 0.36),
    ("2_2_4.csv", 0.6, 0.4, 0.26),
    ("2_2_5.csv", 0.6, 0.4, 0.28),
    ("2_2_6.csv", 0.6, 0.4, 0.30),
    ("2_2_7.csv", 0.8, 0.2, 0.20),
    ("2_2_8.csv", 0.8, 0.2, 0.22),
    ("2_2_9.csv", 0.8, 0.2, 0.24),
    ("2_2_10.csv", 1.0, 0.0, 0.16),
)
AMOUNT_MULTIPLIER = 0.8
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "data"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args(argv)


def fit_models(train_features, claim_amounts, seed):
    primary = DashboardClaimModel(PRIMARY_SETTINGS, random_seed=seed).fit(
        train_features, claim_amounts.to_numpy(),
    )
    secondary = DashboardClaimModel(SECONDARY_SETTINGS, random_seed=seed).fit(
        train_features, claim_amounts.to_numpy(),
    )
    return primary, secondary


def profile_amount(primary_model, test_features):
    ridge_amount = np.maximum(primary_model.regressor_.predict(test_features), 0.0)
    return primary_model.history_.predict_amount(test_features, ridge_amount)


def write_candidates(output_directory, seed):
    training_data = load_training_data()
    test_data = load_test_data()
    primary, secondary = fit_models(
        training_data[ALL_FEATURE_COLUMNS], training_data[TARGET_COLUMN], seed,
    )
    test_features = test_data[ALL_FEATURE_COLUMNS]
    first_probability = primary.predict_claim_probability(test_features)
    second_probability = secondary.predict_claim_probability(test_features)
    amount = profile_amount(primary, test_features) * AMOUNT_MULTIPLIER

    for filename, first_weight, second_weight, threshold in CANDIDATES:
        probability = first_weight * first_probability + second_weight * second_probability
        prediction = np.where(probability >= threshold, probability * amount, 0.0)
        output_path = output_directory / filename
        pd.DataFrame({
            ROW_INDEX_COLUMN: test_data[ROW_INDEX_COLUMN],
            TARGET_COLUMN: prediction,
        }).to_csv(output_path, index=False)
        print(f"{filename}: nonzero predictions = {(prediction > 0).sum()}")


def main(argv=None):
    args = parse_args(argv)
    write_candidates(args.output_dir, args.seed)


if __name__ == "__main__":
    main()
