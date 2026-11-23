"""
CLI entry point: trains the two-stage claim model, reports
cross-validated and holdout performance, then refits on all available
training data and saves the result for predict.py to use.
"""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

from sklearn.metrics import classification_report, confusion_matrix, f1_score, mean_absolute_error, roc_auc_score

from src.data_loading import add_has_claim_label, load_training_data
from src.evaluation import (
    cross_validate_thresholds, stratified_train_validation_split, validate_evaluation_partitions,
)
from src.feature_schema import (
    ALL_FEATURE_COLUMNS, DEFAULT_CV_FOLDS, DEFAULT_MIN_CLAIM_F1, RANDOM_SEED, TARGET_COLUMN, THRESHOLD_SCALE,
)
from src.model_io import DEFAULT_MODEL_PATH, save_model
from src.training_config import (
    TrainingSettings, confirm_training, parse_claim_f1, parse_fold_count, parse_random_seed, prompt_settings,
)
from src.two_stage_model import build_two_stage_model


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the two-stage claim model.")
    parser.add_argument("--config", nargs="?", const=True, default=False, type=Path, metavar="JSON_FILE",
                        help="Configure interactively, or load a dashboard JSON configuration file.")
    parser.add_argument("--n-splits", type=parse_fold_count, default=DEFAULT_CV_FOLDS,
                        help="CV fold count (also the initial value in config mode).")
    parser.add_argument("--seed", type=parse_random_seed, default=RANDOM_SEED,
                        help="Evaluation/model seed (also the initial value in config mode).")
    parser.add_argument("--min-claim-f1", type=parse_claim_f1, default=DEFAULT_MIN_CLAIM_F1,
                        help="Minimum mean CV claim F1 when selecting a threshold (0 uses MAE alone).")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Output model path.")
    parser.add_argument("--full-fit-only", action="store_true",
                        help="Fit all training rows using a JSON config with one operating policy.")
    return parser.parse_args(argv)


def _build_model(settings, threshold):
    if settings.reproduction is not None:
        from src.dashboard_config import build_dashboard_model
        return build_dashboard_model(settings.reproduction, float(threshold) / THRESHOLD_SCALE)
    return build_two_stage_model(settings.model_id, float(threshold) / THRESHOLD_SCALE,
                                 min_samples_leaf=settings.min_samples_leaf,
                                 oversampling_percent=settings.oversampling_percent,
                                 classifier_params=settings.classifier_params,
                                 use_profile_history=settings.use_profile_history,
                                 profile_history_smoothing=settings.profile_history_smoothing,
                                 severity_method=settings.severity_method, ridge_alpha=settings.ridge_alpha,
                                 severity_scale=settings.severity_scale, random_seed=settings.random_seed)


def _select_best_threshold(X_train, y_train, settings):
    candidates = settings.threshold_candidates()
    probabilities = [float(threshold) / THRESHOLD_SCALE for threshold in candidates]
    try:
        mae, f1 = cross_validate_thresholds(
            _build_model(settings, candidates[0]), X_train, y_train, probabilities,
            n_splits=settings.n_splits, random_seed=settings.random_seed,
        )
    except Exception as error:
        raise ValueError(f"Threshold cross-validation failed: {error}") from error
    if settings.reproduction and settings.reproduction.get("template_claim_ensemble"):
        print(f"Expected-F1 decision rule: mean CV MAE {mae[0]:.2f}, claim F1 {f1[0]:.4f}")
        return candidates[0]
    for threshold, mean_mae, mean_f1 in zip(probabilities, mae, f1):
        print(f"Threshold {threshold:.3f}: mean CV MAE {mean_mae:.2f}, claim F1 {mean_f1:.4f}")

    eligible = [index for index, score in enumerate(f1) if score >= settings.min_claim_f1]
    if eligible:
        best = min(eligible, key=lambda index: mae[index])
    elif len(candidates) == 1:
        best = 0
    else:
        print(f"No threshold met claim F1 {settings.min_claim_f1:.3f}; selecting by highest F1, then lowest MAE.")
        best = min(range(len(candidates)), key=lambda index: (-f1[index], mae[index]))
    print(f"Selected threshold {probabilities[best]:.3f} (mean CV MAE {mae[best]:.2f}, claim F1 {f1[best]:.4f})")
    return candidates[best]


def _print_holdout_results(model, X_val, y_val):
    holdout_predictions = model.predict(X_val)
    print(f"Holdout MAE: {mean_absolute_error(y_val, holdout_predictions):.2f}")

    has_claim_val = (y_val > 0).astype(int)
    claim_probabilities = model.predict_claim_probability(X_val)
    print(f"Max predicted claim probability (holdout): {claim_probabilities.max():.4f}")
    print(f"Holdout classifier AUC: {roc_auc_score(has_claim_val, claim_probabilities):.4f}")
    # Count the final nonzero outputs so the report reflects the selected
    # threshold and zero-clipping, rather than the classifier's default cutoff.
    predicted_has_claim = (holdout_predictions > 0).astype(int)
    print(f"Holdout claim F1: {f1_score(has_claim_val, predicted_has_claim, zero_division=0):.4f}")
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        has_claim_val, predicted_has_claim, labels=[0, 1]
    ).ravel()
    print("Holdout claim classification (actual/predicted ClaimAmount > 0):")
    print(f"True positives: {true_positives}")
    print(f"False positives: {false_positives}")
    print(f"True negatives: {true_negatives}")
    print(f"False negatives: {false_negatives}")
    print(classification_report(has_claim_val, predicted_has_claim, labels=[0, 1], zero_division=0))


def _prepare_training_data(settings):
    training_data = add_has_claim_label(load_training_data())
    X = training_data[ALL_FEATURE_COLUMNS]
    y = training_data[TARGET_COLUMN]
    try:
        partitions = stratified_train_validation_split(
            X, y, test_size=settings.evaluation_fraction, random_seed=settings.random_seed,
        )
    except ValueError as error:
        raise ValueError("Cannot create the stratified split; adjust the evaluation percentage "
                         f"or provide more examples of both claim classes. {error}") from error
    validate_evaluation_partitions(partitions[2], partitions[3], settings.n_splits)
    return X, y, partitions


def _train_and_save(settings, X, y, partitions, model_path=DEFAULT_MODEL_PATH):
    X_train, X_val, y_train, y_val = partitions
    best_threshold = _select_best_threshold(X_train, y_train, settings)
    model = _build_model(settings, best_threshold)

    model.fit(X_train, y_train)
    training_predictions = model.predict(X_train)
    print(f"Training MAE: {mean_absolute_error(y_train, training_predictions):.2f}")
    _print_holdout_results(model, X_val, y_val)

    _save_full_fit(model, X, y, model_path)


def _save_full_fit(model, X, y, model_path):
    model.fit(X, y)
    saved_path = save_model(model, model_path)
    print(f"Saved trained model to {saved_path}")


def main(argv=None):
    args = parse_args(argv)
    try:
        settings = TrainingSettings(n_splits=args.n_splits, random_seed=args.seed, min_claim_f1=args.min_claim_f1)
        if args.config is True:
            settings = prompt_settings(settings)
        elif args.config:
            from src.dashboard_config import load_dashboard_settings
            settings = replace(load_dashboard_settings(args.config), min_claim_f1=args.min_claim_f1)
        if args.full_fit_only:
            if settings.reproduction is None or len(settings.threshold_candidates()) != 1:
                raise ValueError("--full-fit-only requires a JSON config with one operating policy.")
            data = load_training_data()
            X, y, partitions = data[ALL_FEATURE_COLUMNS], data[TARGET_COLUMN], None
        else:
            X, y, partitions = _prepare_training_data(settings)
        if args.config:
            confirm_training(settings, args.model_path, full_fit_only=args.full_fit_only)
    except (EOFError, KeyboardInterrupt):
        print("\nTraining cancelled before fitting.")
        return 130
    except (ValueError, OSError) as error:
        print(f"Training configuration error: {error}", file=sys.stderr)
        return 2

    try:
        if args.full_fit_only:
            model = _build_model(settings, settings.threshold_candidates()[0])
            _save_full_fit(model, X, y, args.model_path)
        else:
            _train_and_save(settings, X, y, partitions, args.model_path)
    except (ValueError, ImportError, OSError) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        if isinstance(error, ImportError):
            print("Install optional model dependencies with: python -m pip install -r src/optional_requirements.txt", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
