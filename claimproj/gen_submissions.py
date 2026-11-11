"""Regenerate checkpoint submissions 7-10 and report their full-training MAE."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from claimproj.common import (
    add_group_feats, decide_group, fit_group_amounts, fit_group_stats,
    group_keys, load_test, load_train, make_model, residual_dist,
)
from src.feature_schema import (
    GROUP_SUBMISSION_SETTINGS, ROW_INDEX_COLUMN, TARGET_COLUMN,
)

DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "regenerated"


def fit_probabilities(training_features, labels, test_features):
    """Reuse the same fitted classifiers for every threshold and amount scale."""
    training_probabilities, test_probabilities = {}, {}
    for name in ("lgbm", "cat"):
        model = make_model(name).fit(training_features, labels)
        training_probabilities[name] = model.predict_proba(training_features)[:, 1]
        test_probabilities[name] = model.predict_proba(test_features)[:, 1]
    for probabilities in (training_probabilities, test_probabilities):
        probabilities["blend"] = (probabilities["lgbm"] + probabilities["cat"]) / 2
    return training_probabilities, test_probabilities


def generate_submissions(output_directory):
    features, amounts = load_train()
    identifiers, test_features = load_test()
    labels = (amounts > 0).astype(int)
    keys, test_keys = group_keys(features), group_keys(test_features)
    stats = fit_group_stats(keys, labels)
    training_features = add_group_feats(features, keys, stats, own_b=labels)
    test_features = add_group_feats(test_features, test_keys, stats)
    training_probabilities, test_probabilities = fit_probabilities(
        training_features, labels, test_features,
    )
    logs = fit_group_amounts(keys, amounts)
    residuals = residual_dist(logs)
    positive_amounts = np.sort(amounts[amounts > 0])
    output_directory.mkdir(parents=True, exist_ok=True)
    scores = []
    for number, (name, threshold, scale) in GROUP_SUBMISSION_SETTINGS.items():
        predictions = decide_group(test_probabilities[name], test_keys, logs, residuals,
                                   positive_amounts, threshold, scale=scale)
        training_predictions = decide_group(
            training_probabilities[name], keys, logs, residuals, positive_amounts,
            threshold, scale=scale,
        )
        filename = f"1_2_{number}.csv"
        pd.DataFrame({ROW_INDEX_COLUMN: identifiers, TARGET_COLUMN: predictions}).to_csv(
            output_directory / filename, index=False,
        )
        training_mae = float(np.abs(amounts - training_predictions).mean())
        scores.append({"file": filename, "training_mae": training_mae})
        print(f"{filename}: training MAE = {training_mae:.4f}; "
              f"nonzero = {(predictions > 0).sum()}")
    pd.DataFrame(scores).to_csv(output_directory / "training_mae.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY,
                        help="Destination; existing files in this directory are replaced.")
    args = parser.parse_args()
    generate_submissions(args.output_dir)


if __name__ == "__main__":
    main()
