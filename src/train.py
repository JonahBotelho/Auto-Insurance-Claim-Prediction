"""
CLI entry point: trains the two-stage claim model, reports
cross-validated and holdout performance, then refits on all available
training data and saves the result for predict.py to use.
"""

import argparse
import sys

import numpy as np
from sklearn.metrics import classification_report, mean_absolute_error, roc_auc_score

from src.data_loading import add_has_claim_label, load_training_data
from src.evaluation import (
    cross_validate_mae, stratified_train_validation_split, validate_evaluation_partitions,
)
from src.feature_schema import (
    ALL_FEATURE_COLUMNS, DEFAULT_CV_FOLDS, DEFAULT_MODEL_ID, TARGET_COLUMN, THRESHOLD_SCALE,
)
from src.model_io import DEFAULT_MODEL_PATH, save_model
from src.training_config import TrainingSettings, confirm_training, parse_fold_count, prompt_settings
from src.two_stage_model import TwoStageClaimModel


MODEL_FACTORIES = {DEFAULT_MODEL_ID: TwoStageClaimModel}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the two-stage claim model.")
    parser.add_argument("--config", action="store_true", help="Configure this run interactively.")
    parser.add_argument("--n-splits", type=parse_fold_count, default=DEFAULT_CV_FOLDS,
                        help="CV fold count (also the initial value in config mode).")
    return parser.parse_args(argv)


def _build_model(settings, threshold):
    return MODEL_FACTORIES[settings.model_id](claim_probability_threshold=threshold / THRESHOLD_SCALE)


def _select_best_threshold(X_train, y_train, settings):
    # Holdout rows never influence selection; ascending candidates and strict
    # improvement make ties reproducibly prefer the lowest threshold.
    best_threshold = settings.min_threshold
    best_mean_mae = float("inf")
    for threshold in settings.threshold_candidates():
        candidate_model = _build_model(settings, threshold)
        try:
            fold_scores = np.asarray(cross_validate_mae(
                candidate_model, X_train, y_train, n_splits=settings.n_splits
            ))
        except Exception as error:
            raise ValueError(f"Cross-validation failed at threshold "
                             f"{threshold / THRESHOLD_SCALE:.2f}: {error}") from error
        if fold_scores.size != settings.n_splits or not np.isfinite(fold_scores).all():
            raise ValueError("Cross-validation returned missing or nonfinite scores; training stopped.")
        mean_mae = fold_scores.mean()
        print(f"Threshold {threshold / THRESHOLD_SCALE:.2f}: mean CV MAE = {mean_mae:.2f}")
        if mean_mae < best_mean_mae:
            best_mean_mae = mean_mae
            best_threshold = threshold
    print(f"Selected threshold {best_threshold / THRESHOLD_SCALE:.2f} (mean CV MAE {best_mean_mae:.2f})")
    return best_threshold


def _print_holdout_results(model, X_val, y_val):
    holdout_predictions = model.predict(X_val)
    print(f"Holdout MAE: {mean_absolute_error(y_val, holdout_predictions):.2f}")

    has_claim_val = (y_val > 0).astype(int)
    claim_probabilities = model.classifier_.predict_proba(X_val)[:, 1]
    print(f"Max predicted claim probability (holdout): {claim_probabilities.max():.4f}")
    print(f"Holdout classifier AUC: {roc_auc_score(has_claim_val, claim_probabilities):.4f}")
    print(classification_report(has_claim_val, model.classifier_.predict(X_val)))


def _prepare_training_data(settings):
    training_data = add_has_claim_label(load_training_data())
    X = training_data[ALL_FEATURE_COLUMNS]
    y = training_data[TARGET_COLUMN]
    try:
        partitions = stratified_train_validation_split(X, y, test_size=settings.evaluation_fraction)
    except ValueError as error:
        raise ValueError("Cannot create the stratified split; adjust the evaluation percentage "
                         f"or provide more examples of both claim classes. {error}") from error
    validate_evaluation_partitions(partitions[2], partitions[3], settings.n_splits)
    return X, y, partitions


def _train_and_save(settings, X, y, partitions):
    X_train, X_val, y_train, y_val = partitions
    best_threshold = _select_best_threshold(X_train, y_train, settings)
    model = _build_model(settings, best_threshold)

    model.fit(X_train, y_train)
    _print_holdout_results(model, X_val, y_val)

    # Evaluation is complete; use all labeled rows for the saved model.
    model.fit(X, y)
    saved_path = save_model(model)
    print(f"Saved trained model to {saved_path}")


def main(argv=None):
    args = parse_args(argv)
    try:
        settings = TrainingSettings(n_splits=args.n_splits)
        if args.config:
            settings = prompt_settings(settings)
        X, y, partitions = _prepare_training_data(settings)
        if args.config and not confirm_training(settings, DEFAULT_MODEL_PATH):
            print("Training cancelled.")
            return 0
    except (EOFError, KeyboardInterrupt):
        print("\nTraining cancelled before fitting.")
        return 130
    except (ValueError, OSError) as error:
        print(f"Training configuration error: {error}", file=sys.stderr)
        return 2

    try:
        _train_and_save(settings, X, y, partitions)
    except ValueError as error:
        print(f"Training failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
