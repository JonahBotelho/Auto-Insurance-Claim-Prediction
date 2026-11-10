"""
Choose the lasso regularization strength (alpha) with 5-fold cross-validation.

Compares lasso on raw claim amounts against lasso on log(1 + ClaimAmount),
across a grid of alphas, and reports training MAE and cross-validated MAE next
to the $0 baseline. Also lists which features the chosen model keeps.

Run from the repository folder (about 30 seconds):
    python analysis/lasso_cross_validation.py

Results are saved to analysis/output/lasso_cv_results.csv.
"""

import warnings

import pandas as pd
from sklearn.exceptions import ConvergenceWarning

# evaluation must be imported first: it adds src/ to the import path.
from evaluation import (
    OUTPUT_FOLDER, TRAINING_PATH, cross_validate_mae, duplicate_row_groups,
)
from data_io import load_training_set
from lasso_claim_model import build_lasso_claim_model

LOG_TARGET_ALPHAS = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
# Raw-target alphas below 0.01 take 30+ seconds per fit to converge, because
# the $100k+ claims make the squared error huge, and they score no better.
RAW_TARGET_ALPHAS = [0.01, 0.1, 1.0, 10.0]
OUTPUT_PATH = OUTPUT_FOLDER / "lasso_cv_results.csv"


def alpha_grid() -> list[tuple[bool, float]]:
    raw = [(False, alpha) for alpha in RAW_TARGET_ALPHAS]
    log = [(True, alpha) for alpha in LOG_TARGET_ALPHAS]
    return raw + log


def print_kept_features(features, claims, alpha) -> None:
    """Fit on all training data and list the features with nonzero weights."""
    model = build_lasso_claim_model(alpha=alpha).fit(features, claims)
    pipeline = model.regressor_
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    weights = pd.Series(pipeline.named_steps["lasso"].coef_, index=names)
    kept = weights[weights != 0].sort_values(key=abs, ascending=False)
    print(f"\nFeatures kept at alpha={alpha}: {len(kept)} of {len(weights)}")
    print(kept.round(4).to_string() if len(kept) else "(none: predicts a constant)")


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    features, claims = load_training_set(TRAINING_PATH)
    groups = duplicate_row_groups(features)

    rows = []
    for log_target, alpha in alpha_grid():
        model = build_lasso_claim_model(alpha=alpha, log_target=log_target)
        scores = cross_validate_mae(model, features, claims, groups)
        rows.append({"target": "log(1+y)" if log_target else "raw", "alpha": alpha, **scores})

    results = pd.DataFrame(rows)
    print(f"$0 baseline MAE: {claims.abs().mean():.2f}\n")
    print(results.round({"training_mae": 2, "cv_mae": 2, "seconds": 1}).to_string(index=False))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    best = results.loc[results["cv_mae"].idxmin()]
    print(f"\nBest: target={best['target']}, alpha={best['alpha']}, CV MAE={best['cv_mae']:.2f}")
    # build_lasso_claim_model's default uses the log target, which wins here.
    print_kept_features(features, claims, alpha=best["alpha"])


if __name__ == "__main__":
    main()
