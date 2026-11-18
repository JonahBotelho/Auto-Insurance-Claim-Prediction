"""Validated training settings and terminal configuration for editable model options."""

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
import math

from src.feature_schema import (
    DEFAULT_CV_FOLDS, DEFAULT_EVALUATION_PERCENT, DEFAULT_MAX_THRESHOLD_HUNDREDTHS,
    DEFAULT_MIN_CLAIM_F1, DEFAULT_MIN_THRESHOLD_HUNDREDTHS, DEFAULT_MODEL_ID, DEFAULT_OVERSAMPLING_PERCENT,
    DEFAULT_PROFILE_HISTORY_SMOOTHING, HIST_CONFIG_OPTIONS, HIST_GRADIENT_BOOSTING_MODEL_ID,
    HIST_GRADIENT_BOOSTING_PARAMS, MAX_EVALUATION_PERCENT, MAX_RANDOM_SEED, MAX_THRESHOLD_HUNDREDTHS,
    MIN_CV_FOLDS, MIN_EVALUATION_PERCENT, MIN_THRESHOLD_HUNDREDTHS, MODEL_LABELS,
    RANDOM_SEED, RIDGE_PARAMS, SEVERITY_METHOD_LABELS, THRESHOLD_SCALE, THRESHOLD_SUBDIVISIONS,
    TREE_CLASSIFIER_PARAMS, MIN_OVERSAMPLING_PERCENT, MAX_OVERSAMPLING_PERCENT,
)
from src.oversampling import validate_oversampling_percent


def _validate_integer(value, minimum, maximum, label):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        bounds = f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
        raise ValueError(f"{label} must be a whole number {bounds}.")
    return value


def _validate_real(value, label, allow_zero=True):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite.")
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{label} must be {'nonnegative' if allow_zero else 'positive'}.")
    return value


def _validate_hist_value(name, value):
    label, kind, minimum = HIST_CONFIG_OPTIONS[name]
    if kind is int:
        return _validate_integer(value, minimum, None, label)
    return _validate_real(value, label, allow_zero=name != "learning_rate")


def _threshold_subunits(value):
    # Public settings retain hundredth units for existing programmatic callers.
    # Fractional hundredths allow CLI inputs such as 0.365 without float drift.
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError("Invalid threshold.")
    decimal = Decimal(str(value))
    if not decimal.is_finite() or not MIN_THRESHOLD_HUNDREDTHS <= decimal <= MAX_THRESHOLD_HUNDREDTHS:
        raise ValueError("Threshold must be between 0 and 1.")
    numerator, denominator = decimal.as_integer_ratio()
    units, remainder = divmod(numerator * THRESHOLD_SUBDIVISIONS, denominator)
    if remainder:
        raise ValueError("Threshold precision must be at most three decimal places.")
    return units



@dataclass(frozen=True)
class TrainingSettings:
    """Threshold fields use hundredth units, including fractional hundredths."""

    min_threshold: int | Decimal = DEFAULT_MIN_THRESHOLD_HUNDREDTHS
    max_threshold: int | Decimal = DEFAULT_MAX_THRESHOLD_HUNDREDTHS
    evaluation_percent: int = DEFAULT_EVALUATION_PERCENT
    model_id: str = DEFAULT_MODEL_ID
    n_splits: int = DEFAULT_CV_FOLDS
    min_samples_leaf: int | None = None
    oversampling_percent: int = DEFAULT_OVERSAMPLING_PERCENT
    classifier_params: dict = field(default_factory=dict)
    use_profile_history: bool = False
    profile_history_smoothing: float = DEFAULT_PROFILE_HISTORY_SMOOTHING
    severity_method: str = "ridge"
    ridge_alpha: float = RIDGE_PARAMS["alpha"]
    severity_scale: float = 1.0
    random_seed: int = RANDOM_SEED
    reproduction: dict | None = None
    min_claim_f1: float = DEFAULT_MIN_CLAIM_F1

    def __post_init__(self):
        if self.reproduction is not None:
            from src.dashboard_config import validate_dashboard_config
            validate_dashboard_config(self.reproduction)
        validate_oversampling_percent(self.oversampling_percent)
        for value in (self.min_threshold, self.max_threshold):
            _threshold_subunits(value)
        if self.min_threshold > self.max_threshold:
            raise ValueError("Maximum threshold must be at least the minimum threshold.")
        _validate_integer(self.evaluation_percent, MIN_EVALUATION_PERCENT, MAX_EVALUATION_PERCENT, "Evaluation percentage")
        _validate_integer(self.n_splits, MIN_CV_FOLDS, None, "CV folds")
        _validate_integer(self.random_seed, 0, MAX_RANDOM_SEED, "Evaluation/model random seed")
        if self.model_id not in MODEL_LABELS:
            raise ValueError("Choose an available model.")
        if self.min_samples_leaf is not None:
            _validate_integer(self.min_samples_leaf, 1, None, "Minimum samples per leaf")
            if self.model_id not in TREE_CLASSIFIER_PARAMS:
                raise ValueError("Minimum samples per leaf requires a tree classifier.")
        if not isinstance(self.classifier_params, dict):
            raise ValueError("Classifier parameters must be a dictionary.")
        if self.classifier_params and self.model_id != HIST_GRADIENT_BOOSTING_MODEL_ID:
            raise ValueError("Advanced classifier options require Histogram Gradient Boosting.")
        for name, value in self.classifier_params.items():
            if name not in HIST_CONFIG_OPTIONS:
                raise ValueError(f"Unsupported classifier option: {name}")
            _validate_hist_value(name, value)
        if type(self.use_profile_history) is not bool:
            raise ValueError("Profile history must be enabled or disabled.")
        if self.severity_method not in SEVERITY_METHOD_LABELS:
            raise ValueError("Choose an available severity method.")
        _validate_real(self.profile_history_smoothing, "Profile smoothing")
        _validate_real(self.ridge_alpha, "Ridge alpha")
        _validate_real(self.severity_scale, "Severity scale", allow_zero=False)
        _validate_claim_f1(self.min_claim_f1)

    @property
    def evaluation_fraction(self):
        return self.evaluation_percent / 100

    def threshold_candidates(self):
        """Preserve 0.01 steps and include both bounds, even with thousandth inputs."""
        start, stop = (_threshold_subunits(v) for v in (self.min_threshold, self.max_threshold))
        candidates = list(range(start, stop + 1, THRESHOLD_SUBDIVISIONS))
        if candidates[-1] != stop:
            candidates.append(stop)
        return [Decimal(value) / THRESHOLD_SUBDIVISIONS for value in candidates]


def parse_threshold(text):
    message = "Threshold must be between 0 and 1 in steps of 0.001."
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise ValueError(message) from error
    if not value.is_finite() or not 0 <= value <= 1:
        raise ValueError(message)
    numerator, denominator = value.as_integer_ratio()
    units, remainder = divmod(numerator * THRESHOLD_SCALE * THRESHOLD_SUBDIVISIONS, denominator)
    if remainder:
        raise ValueError(message)
    return units // THRESHOLD_SUBDIVISIONS if units % THRESHOLD_SUBDIVISIONS == 0 else Decimal(units) / THRESHOLD_SUBDIVISIONS


def _parse_integer(text, minimum, maximum, label):
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError(f"{label} must be a whole number.") from error
    return _validate_integer(value, minimum, maximum, label)


def parse_fold_count(text):
    return _parse_integer(text, MIN_CV_FOLDS, None, "CV folds")


def parse_random_seed(text):
    return _parse_integer(text, 0, MAX_RANDOM_SEED, "Evaluation/model random seed")


def _parse_real(text, label, allow_zero=True):
    try:
        value = float(text)
    except ValueError as error:
        raise ValueError(f"{label} must be a number.") from error
    return _validate_real(value, label, allow_zero)


def _validate_claim_f1(value):
    if _validate_real(value, "Minimum CV claim F1") > 1:
        raise ValueError("Minimum CV claim F1 must be between 0 and 1.")
    return value


def parse_claim_f1(text):
    return _validate_claim_f1(_parse_real(text, "Minimum CV claim F1"))


def _parse_boolean(text):
    if text.lower() in ("y", "yes"):
        return True
    if text.lower() in ("n", "no"):
        return False
    raise ValueError("Enter yes or no.")


def _prompt(label, default, parse):
    while True:
        response = input(f"{label} [{default}]: ").strip()
        try:
            return parse(response or str(default))
        except ValueError as error:
            print(error)


def _prompt_choice(label, labels, default):
    options = list(labels)
    for number, key in enumerate(options, 1):
        print(f"  {number}. {labels[key]}")

    def parse(text):
        if text not in [str(n) for n in range(1, len(options) + 1)]:
            raise ValueError("Choose one of the listed options.")
        return options[int(text) - 1]

    return _prompt(label, options.index(default) + 1, parse)


def _prompt_hist_parameters(defaults):
    parameters = {}
    for name, (label, kind, _) in HIST_CONFIG_OPTIONS.items():
        def parse(text, name=name, kind=kind):
            try:
                value = kind(text)
            except ValueError as error:
                raise ValueError(f"{HIST_CONFIG_OPTIONS[name][0]} must be a valid number.") from error
            return _validate_hist_value(name, value)
        value = defaults.classifier_params.get(name, HIST_GRADIENT_BOOSTING_PARAMS[name])
        parameters[name] = _prompt(label, value, parse)
    return parameters


def prompt_settings(defaults):
    minimum = _prompt("Minimum threshold", f"{defaults.min_threshold / THRESHOLD_SCALE:.3f}", parse_threshold)

    def parse_maximum(text):
        maximum = parse_threshold(text)
        if maximum < minimum:
            raise ValueError("Maximum threshold must be at least the minimum threshold.")
        return maximum

    maximum = _prompt("Maximum threshold", f"{max(defaults.max_threshold, minimum) / THRESHOLD_SCALE:.3f}", parse_maximum)
    evaluation = _prompt("Evaluation percentage", defaults.evaluation_percent,
                         lambda text: _parse_integer(text, MIN_EVALUATION_PERCENT, MAX_EVALUATION_PERCENT, "Evaluation percentage"))
    print(f"Split: {100 - evaluation}% training / {evaluation}% evaluation")
    model_id = _prompt_choice("Model", MODEL_LABELS, defaults.model_id)
    leaf_size = None
    if model_id in TREE_CLASSIFIER_PARAMS:
        value = TREE_CLASSIFIER_PARAMS[model_id]["min_samples_leaf"]
        if model_id == defaults.model_id and defaults.min_samples_leaf is not None:
            value = defaults.min_samples_leaf
        leaf_size = _prompt("Minimum samples per leaf", value,
                            lambda text: _parse_integer(text, 1, None, "Minimum samples per leaf"))
    parameters = _prompt_hist_parameters(defaults) if model_id == HIST_GRADIENT_BOOSTING_MODEL_ID else {}
    oversampling = _prompt("Target positive claim percentage (0 disables oversampling)", defaults.oversampling_percent,
                          lambda text: validate_oversampling_percent(_parse_integer(text, MIN_OVERSAMPLING_PERCENT, MAX_OVERSAMPLING_PERCENT, "Oversampling percentage")))
    history = _prompt("Use profile-history classification features (yes/no)",
                      "yes" if defaults.use_profile_history else "no", _parse_boolean)
    smoothing = defaults.profile_history_smoothing
    if history:
        smoothing = _prompt("Profile-history smoothing strength", smoothing,
                            lambda text: _parse_real(text, "Profile smoothing"))
    severity = _prompt_choice("Severity method", SEVERITY_METHOD_LABELS, defaults.severity_method)
    alpha = _prompt("Ridge alpha (also used for fallback)", defaults.ridge_alpha,
                    lambda text: _parse_real(text, "Ridge alpha"))
    scale = _prompt("Severity amount multiplier", defaults.severity_scale,
                    lambda text: _parse_real(text, "Severity scale", allow_zero=False))
    folds = _prompt("CV folds", defaults.n_splits, parse_fold_count)
    seed = _prompt("Evaluation/model random seed", defaults.random_seed, parse_random_seed)
    return TrainingSettings(
        minimum, maximum, evaluation, model_id, folds, min_samples_leaf=leaf_size,
        oversampling_percent=oversampling, classifier_params=parameters,
        use_profile_history=history, profile_history_smoothing=smoothing,
        severity_method=severity, ridge_alpha=alpha, severity_scale=scale, random_seed=seed,
        min_claim_f1=defaults.min_claim_f1,
    )


def confirm_training(settings, output_path):
    print("\nTraining configuration:")
    if settings.reproduction is not None:
        from src.dashboard_config import print_dashboard_configuration
        print_dashboard_configuration(settings)
    else:
        print(f"Model: {MODEL_LABELS[settings.model_id]}")
    if settings.model_id in TREE_CLASSIFIER_PARAMS:
        leaf = settings.min_samples_leaf if settings.min_samples_leaf is not None else TREE_CLASSIFIER_PARAMS[settings.model_id]["min_samples_leaf"]
        print(f"Minimum samples per leaf: {leaf}")
    for name, value in settings.classifier_params.items():
        print(f"{HIST_CONFIG_OPTIONS[name][0]}: {value}")
    if settings.reproduction is None:
        print(f"Classifier oversampling target: {settings.oversampling_percent}% (0 disables)")
        print(f"Profile-history features: {settings.use_profile_history}; smoothing: {settings.profile_history_smoothing}")
        print(f"Severity: {SEVERITY_METHOD_LABELS[settings.severity_method]}; Ridge alpha: {settings.ridge_alpha}; multiplier: {settings.severity_scale}")
    print(f"Thresholds: {settings.min_threshold / THRESHOLD_SCALE:.3f}-{settings.max_threshold / THRESHOLD_SCALE:.3f} "
          f"in 0.01 steps, including both bounds ({len(settings.threshold_candidates())} candidates)")
    print(f"Minimum mean CV claim F1 for threshold selection: {settings.min_claim_f1:.3f}")
    print(f"Split: {100 - settings.evaluation_percent}% training / {settings.evaluation_percent}% evaluation; CV folds: {settings.n_splits}")
    print(f"Evaluation/model seed: {settings.random_seed}; oversampling sampling seed: {RANDOM_SEED}")
    print("The split is for evaluation; the saved model is refitted on all rows.")
    print(f"Model output: {output_path} (replaces the file if it already exists)")
    input("Press Enter to start training, or Ctrl+C to cancel: ")
