"""Validated one-run training settings and dependency-free terminal prompts."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

from src.feature_schema import (
    DEFAULT_CV_FOLDS,
    DEFAULT_EVALUATION_PERCENT,
    DEFAULT_MAX_THRESHOLD_HUNDREDTHS,
    DEFAULT_MIN_THRESHOLD_HUNDREDTHS,
    DEFAULT_MODEL_ID,
    MAX_EVALUATION_PERCENT,
    MAX_THRESHOLD_HUNDREDTHS,
    MIN_CV_FOLDS,
    MIN_EVALUATION_PERCENT,
    MIN_THRESHOLD_HUNDREDTHS,
    MODEL_LABELS,
    THRESHOLD_SCALE,
)


def _validate_integer(value, minimum, maximum, label):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        bounds = f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
        raise ValueError(f"{label} must be a whole number {bounds}.")
    return value


@dataclass(frozen=True)
class TrainingSettings:
    """Threshold bounds are integer hundredths, not floating-point probabilities."""

    min_threshold: int = DEFAULT_MIN_THRESHOLD_HUNDREDTHS
    max_threshold: int = DEFAULT_MAX_THRESHOLD_HUNDREDTHS
    evaluation_percent: int = DEFAULT_EVALUATION_PERCENT
    model_id: str = DEFAULT_MODEL_ID
    n_splits: int = DEFAULT_CV_FOLDS

    def __post_init__(self):
        for value in (self.min_threshold, self.max_threshold):
            _validate_integer(value, MIN_THRESHOLD_HUNDREDTHS, MAX_THRESHOLD_HUNDREDTHS,
                              "Threshold in hundredths")
        if self.min_threshold > self.max_threshold:
            raise ValueError("Maximum threshold must be at least the minimum threshold.")
        _validate_integer(self.evaluation_percent, MIN_EVALUATION_PERCENT,
                          MAX_EVALUATION_PERCENT, "Evaluation percentage")
        _validate_integer(self.n_splits, MIN_CV_FOLDS, None, "CV folds")
        if self.model_id not in MODEL_LABELS:
            raise ValueError("Choose an available model.")

    @property
    def evaluation_fraction(self):
        return self.evaluation_percent / 100

    def threshold_candidates(self):
        """Keep exact integer candidates until the model factory needs floats."""
        return range(self.min_threshold, self.max_threshold + 1)


def parse_threshold(text):
    message = "Threshold must be between 0.00 and 1.00 in steps of 0.01."
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(message) from error
    if not value.is_finite() or not 0 <= value <= 1:
        raise ValueError(message)
    # Integer ratios avoid rounding long decimal input to a valid hundredth.
    numerator, denominator = value.as_integer_ratio()
    hundredths, remainder = divmod(numerator * THRESHOLD_SCALE, denominator)
    if remainder:
        raise ValueError(message)
    return hundredths


def parse_fold_count(text):
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError(f"CV folds must be a whole number at least {MIN_CV_FOLDS}.") from error
    return _validate_integer(value, MIN_CV_FOLDS, None, "CV folds")


def _parse_evaluation_percent(text):
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError("Evaluation percentage must be a whole number from 1 to 99.") from error
    return _validate_integer(value, MIN_EVALUATION_PERCENT, MAX_EVALUATION_PERCENT,
                             "Evaluation percentage")


def _prompt(label, default, parse):
    while True:
        response = input(f"{label} [{default}]: ").strip()
        try:
            return parse(response or str(default))
        except ValueError as error:
            print(error)


def _prompt_model(default_model_id):
    model_ids = list(MODEL_LABELS)
    for number, model_id in enumerate(model_ids, start=1):
        print(f"  {number}. {MODEL_LABELS[model_id]}")

    def parse_choice(text):
        if text not in [str(number) for number in range(1, len(model_ids) + 1)]:
            raise ValueError("Choose one of the listed model numbers.")
        return model_ids[int(text) - 1]

    return _prompt("Model", model_ids.index(default_model_id) + 1, parse_choice)


def prompt_settings(defaults):
    minimum = _prompt("Minimum threshold", f"{defaults.min_threshold / THRESHOLD_SCALE:.2f}",
                      parse_threshold)

    def parse_maximum(text):
        maximum = parse_threshold(text)
        if maximum < minimum:
            raise ValueError("Maximum threshold must be at least the minimum threshold.")
        return maximum

    maximum_default = max(defaults.max_threshold, minimum)
    maximum = _prompt("Maximum threshold", f"{maximum_default / THRESHOLD_SCALE:.2f}", parse_maximum)
    evaluation = _prompt("Evaluation percentage", defaults.evaluation_percent,
                         _parse_evaluation_percent)
    print(f"Split: {100 - evaluation}% training / {evaluation}% evaluation")
    model_id = _prompt_model(defaults.model_id)
    folds = _prompt("CV folds", defaults.n_splits, parse_fold_count)
    return TrainingSettings(minimum, maximum, evaluation, model_id, folds)


def confirm_training(settings, output_path):
    print("\nTraining configuration:")
    print(f"Model: {MODEL_LABELS[settings.model_id]}")
    print(f"Thresholds: {settings.min_threshold / THRESHOLD_SCALE:.2f}-"
          f"{settings.max_threshold / THRESHOLD_SCALE:.2f} in steps of 0.01 "
          f"({len(settings.threshold_candidates())} candidates)")
    print(f"Split: {100 - settings.evaluation_percent}% training / "
          f"{settings.evaluation_percent}% evaluation; CV folds: {settings.n_splits}")
    print("The split is for evaluation; the saved model is refitted on all rows.")
    print(f"Model output: {output_path} (replaces the file if it already exists)")
    return input("Start training? [y/N]: ").strip().lower() in {"y", "yes"}
