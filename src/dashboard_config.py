"""Validate JSON files downloaded from the dashboard and build training settings."""

import json
import math
from pathlib import Path

from src.feature_schema import (
    CONFIG_FILE_SCHEMA_VERSION, DASHBOARD_CLASSIFIER_LABELS, DASHBOARD_SEVERITY_LABELS,
)
from src.oversampling import validate_oversampling_percent


def _number(value, label, minimum=0, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number.")
    if integer and type(value) is not int:
        raise ValueError(f"{label} must be a whole number.")
    if value < minimum:
        raise ValueError(f"{label} must be at least {minimum}.")
    return value


def validate_dashboard_config(config):
    allowed = {"schema_version", "classifier", "threshold", "evaluation_percent", "random_seed",
               "severity_method", "ridge_alpha", "severity_scale", "amount_cap", "n_splits"}
    if not isinstance(config, dict) or set(config) - allowed:
        raise ValueError("Unknown configuration fields or invalid JSON configuration object.")
    if type(config.get("schema_version")) is not int or config["schema_version"] != CONFIG_FILE_SCHEMA_VERSION:
        raise ValueError("Unsupported configuration schema version.")
    settings = config.get("classifier")
    if not isinstance(settings, dict) or settings.get("family") not in DASHBOARD_CLASSIFIER_LABELS:
        raise ValueError("Choose a supported classifier family.")
    allowed_classifier = {"name", "family", "iterations", "rate", "leaves", "leaf", "depth", "l2",
                          "oversampling", "smoothing", "early_stopping", "amount_history",
                          "extended_amount_history"}
    if set(settings) - allowed_classifier:
        raise ValueError("Unknown classifier configuration field.")
    validate_oversampling_percent(settings.get("oversampling"))
    _number(settings.get("smoothing"), "Profile smoothing")
    if type(settings.get("amount_history", False)) is not bool:
        raise ValueError("Past-amount history must be true or false.")
    if type(settings.get("extended_amount_history", False)) is not bool:
        raise ValueError("Extended amount history must be true or false.")
    if settings.get("extended_amount_history") and not settings.get("amount_history"):
        raise ValueError("Extended amount history requires past-amount history.")
    family = settings["family"]
    if family != "posterior":
        keys = ["leaf"] if family in ["forest", "extra"] else ["iterations"]
        keys += ["leaves", "leaf"] if family == "hist" else []
        keys += ["depth"] if family in ["cat", "cat_native", "lgbm", "xgb"] else []
        keys += ["leaf"] if family == "lgbm" and "leaf" in settings else []
        for key in keys:
            _number(settings.get(key), key, minimum=2 if key == "leaves" else 1, integer=True)
        if family not in ["forest", "extra"]:
            if _number(settings.get("rate"), "Learning rate") == 0:
                raise ValueError("Learning rate must be positive.")
            _number(settings.get("l2"), "L2 regularization")
    early = settings.get("early_stopping", False)
    if type(early) is not bool and early != "auto":
        raise ValueError("Early stopping must be true, false, or auto.")
    if config.get("severity_method") not in DASHBOARD_SEVERITY_LABELS:
        raise ValueError("Unsupported severity method.")
    _number(config.get("ridge_alpha"), "Ridge alpha")
    if _number(config.get("severity_scale"), "Amount multiplier") == 0:
        raise ValueError("Amount multiplier must be positive.")
    if config.get("amount_cap") is not None and _number(config["amount_cap"], "Amount cap") == 0:
        raise ValueError("Amount cap must be positive.")


def load_dashboard_settings(path):
    from src.training_config import TrainingSettings, parse_threshold
    config = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    validate_dashboard_config(config)
    threshold = parse_threshold(str(config.get("threshold")))
    return TrainingSettings(min_threshold=threshold, max_threshold=threshold,
                            evaluation_percent=config.get("evaluation_percent"), n_splits=config.get("n_splits"),
                            random_seed=config.get("random_seed"), ridge_alpha=config["ridge_alpha"],
                            severity_scale=config["severity_scale"], reproduction=config)


def build_dashboard_model(config, threshold):
    from src.dashboard_model import DashboardClaimModel
    validate_dashboard_config(config)
    return DashboardClaimModel(config["classifier"], claim_probability_threshold=threshold,
                               severity_method=config["severity_method"], ridge_alpha=config["ridge_alpha"],
                               severity_scale=config["severity_scale"], amount_cap=config.get("amount_cap"),
                               random_seed=config["random_seed"])


def print_dashboard_configuration(settings):
    config = settings.reproduction
    family = config["classifier"]["family"]
    print(f"Model: {DASHBOARD_CLASSIFIER_LABELS[family]}")
    for key, value in config["classifier"].items():
        if key not in ["name", "family"]:
            print(f"  {key}: {value}")
    print(f"Severity: {DASHBOARD_SEVERITY_LABELS[config['severity_method']]}")
    print(f"Ridge alpha: {config['ridge_alpha']}; amount multiplier: {config['severity_scale']}; cap: {config.get('amount_cap')}")
