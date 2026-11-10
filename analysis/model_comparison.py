"""
Compare every candidate model with the same grouped 5-fold cross-validation.

Prints one table with training MAE, cross-validated MAE, the gap between them
(a large gap means overfitting, Module 4) and run time, sorted best first.
The $0 baseline is included, so any model that does not beat it is not worth
submitting.

Run from the repository folder (about 3 minutes for all models; the k-NN
models take most of it):
    python analysis/model_comparison.py
    python analysis/model_comparison.py --only tree two-part   # names containing these

Results are saved to analysis/output/model_comparison.csv.
"""

import argparse
import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

# evaluation must be imported first: it adds src/ to the import path.
from evaluation import (
    OUTPUT_FOLDER, TRAINING_PATH, cross_validate_mae, duplicate_row_groups,
)
from claim_models import (
    build_knn_claim_model, build_polynomial_claim_model, build_ridge_claim_model,
    build_tree_claim_model,
)
from data_io import load_training_set
from lasso_claim_model import build_lasso_claim_model
from preprocessing import build_preprocessor
from two_part_claim_model import TwoPartClaimModel
from zero_claim_model import ZeroClaimModel

OUTPUT_PATH = OUTPUT_FOLDER / "model_comparison.csv"


def knn_classifier(neighbor_count: int) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("knn", KNeighborsClassifier(n_neighbors=neighbor_count)),
    ])


def candidate_models() -> dict:
    """Name -> unfitted model. Settings come from earlier cross-validation
    runs; each is the best (or a representative) setting."""
    return {
        "zero baseline": ZeroClaimModel(),
        "lasso, raw target, alpha=10": build_lasso_claim_model(10.0, log_target=False),
        "lasso, log target, alpha=0.1": build_lasso_claim_model(0.1),
        "ridge, raw target, alpha=100": build_ridge_claim_model(100.0, log_target=False),
        "ridge, log target, alpha=100": build_ridge_claim_model(100.0),
        "polynomial degree 2 + lasso, log target": build_polynomial_claim_model(2, 0.01),
        "regression tree, raw target, depth 6": build_tree_claim_model(6),
        "regression tree, log target, depth 3": build_tree_claim_model(3, log_target=True),
        "regression tree, median leaves, depth 6":
            build_tree_claim_model(6, criterion="absolute_error"),
        "k-NN regressor, log target, k=5": build_knn_claim_model(5),
        "k-NN regressor, log target, k=25": build_knn_claim_model(25),
        "two-part: tree classifier depth 12": TwoPartClaimModel(
            DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, random_state=0)),
        "two-part: k-NN classifier k=25": TwoPartClaimModel(knn_classifier(25)),
        "two-part: gradient boosting classifier": TwoPartClaimModel(
            HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=0)),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validate all candidate models.")
    parser.add_argument("--only", nargs="*", default=[],
                        help="Run only models whose names contain any of these words.")
    return parser.parse_args()


def select_models(models: dict, name_filters: list[str]) -> dict:
    if not name_filters:
        return models
    return {name: model for name, model in models.items()
            if any(word.lower() in name.lower() for word in name_filters)}


def main() -> None:
    arguments = parse_arguments()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    # Expected: a rare level can be missing from one training fold; the
    # encoder then maps it to the "infrequent" column, which is what we want.
    warnings.filterwarnings("ignore", message="Found unknown categories")
    features, claims = load_training_set(TRAINING_PATH)
    groups = duplicate_row_groups(features)
    baseline_mae = claims.abs().mean()

    rows = []
    for name, model in select_models(candidate_models(), arguments.only).items():
        print(f"Cross-validating: {name} ...", flush=True)
        scores = cross_validate_mae(model, features, claims, groups)
        rows.append({"model": name, **scores})

    results = pd.DataFrame(rows).sort_values("cv_mae")
    results["gap"] = results["cv_mae"] - results["training_mae"]
    results["beats_baseline"] = results["cv_mae"] < baseline_mae - 1e-6
    columns = ["model", "training_mae", "cv_mae", "gap", "seconds", "beats_baseline"]

    print(f"\n$0 baseline MAE: {baseline_mae:.2f}\n")
    print(results[columns].round(2).to_string(index=False))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results[columns].to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
