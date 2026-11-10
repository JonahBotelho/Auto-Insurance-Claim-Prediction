"""
CLI entry point: trains the two-stage claim model, reports
cross-validated and holdout performance, then refits on all available
training data and saves the result for predict.py to use.
"""

import argparse

from sklearn.metrics import classification_report, mean_absolute_error, roc_auc_score

from src.data_loading import add_has_claim_label, load_training_data
from src.evaluation import cross_validate_mae, stratified_train_validation_split
from src.feature_schema import ALL_FEATURE_COLUMNS, TARGET_COLUMN
from src.model_io import save_model
from src.two_stage_model import TwoStageClaimModel


THRESHOLD_CANDIDATES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the two-stage claim model.")
    parser.add_argument("--n-splits", type=int, default=5)
    return parser.parse_args(argv)


def _select_best_threshold(X_train, y_train, n_splits):
    # Compared only on the 80% training split via cross-validation, never
    # against the holdout, so the holdout stays a single unbiased check
    # rather than something used to pick a value.
    best_threshold = THRESHOLD_CANDIDATES[0]
    best_mean_mae = None
    for threshold in THRESHOLD_CANDIDATES:
        candidate_model = TwoStageClaimModel(claim_probability_threshold=threshold)
        fold_scores = cross_validate_mae(candidate_model, X_train, y_train, n_splits=n_splits)
        mean_mae = fold_scores.mean()
        print(f"Threshold {threshold:.1f}: mean CV MAE = {mean_mae:.2f}")
        if best_mean_mae is None or mean_mae < best_mean_mae:
            best_mean_mae = mean_mae
            best_threshold = threshold
    print(f"Selected threshold {best_threshold:.1f} (mean CV MAE {best_mean_mae:.2f})")
    return best_threshold


def _print_holdout_results(model, X_val, y_val):
    holdout_predictions = model.predict(X_val)
    print(f"Holdout MAE: {mean_absolute_error(y_val, holdout_predictions):.2f}")

    has_claim_val = (y_val > 0).astype(int)
    claim_probabilities = model.classifier_.predict_proba(X_val)[:, 1]
    print(f"Max predicted claim probability (holdout): {claim_probabilities.max():.4f}")
    print(f"Holdout classifier AUC: {roc_auc_score(has_claim_val, claim_probabilities):.4f}")
    print(classification_report(has_claim_val, model.classifier_.predict(X_val)))


def main(argv=None):
    args = parse_args(argv)

    training_data = add_has_claim_label(load_training_data())
    X = training_data[ALL_FEATURE_COLUMNS]
    y = training_data[TARGET_COLUMN]
    X_train, X_val, y_train, y_val = stratified_train_validation_split(X, y)

    best_threshold = _select_best_threshold(X_train, y_train, args.n_splits)
    model = TwoStageClaimModel(claim_probability_threshold=best_threshold)

    model.fit(X_train, y_train)
    _print_holdout_results(model, X_val, y_val)

    # Refit on everything (train + holdout combined) so the saved model
    # isn't missing out on the 20% of labeled data held back for evaluation.
    model.fit(X, y)
    saved_path = save_model(model)
    print(f"Saved trained model to {saved_path}")


if __name__ == "__main__":
    main()
