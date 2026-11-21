"""Validate JSON files downloaded from the dashboard and build training settings."""

import json
import math
from pathlib import Path

from src.feature_schema import (
    CONFIG_FILE_SCHEMA_VERSION, DASHBOARD_CLASSIFIER_LABELS, DASHBOARD_SEVERITY_LABELS, DEFAULT_CV_FOLDS,
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
               "severity_method", "ridge_alpha", "severity_scale", "amount_cap", "n_splits", "median_payout_pricing",
               "aggregate_payout_pricing", "profile_history_ensemble", "template_claim_ensemble",
               "payout_source_calibration", "source_tail_pricing", "cold_profile_fallback", "cold_payout_tail_pricing",
               "cold_exposure_hazard", "local_tail_pricing", "pairwise_payout_ranking",
               "bonus_malus_timeline", "count_retrieval_refinement"}
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
    if type(config.get("median_payout_pricing", False)) is not bool:
        raise ValueError("Median payout pricing must be true or false.")
    if type(config.get("aggregate_payout_pricing", False)) is not bool:
        raise ValueError("Aggregate payout pricing must be true or false.")
    if config.get("aggregate_payout_pricing") and config.get("median_payout_pricing"):
        raise ValueError("Select a single profile payout pricer.")
    if type(config.get("profile_history_ensemble", False)) is not bool:
        raise ValueError("Profile history ensemble must be true or false.")
    if type(config.get("template_claim_ensemble", False)) is not bool:
        raise ValueError("Template claim ensemble must be true or false.")
    if config.get("template_claim_ensemble") and config.get("profile_history_ensemble"):
        raise ValueError("Select a single profile detector.")
    if type(config.get("payout_source_calibration", False)) is not bool:
        raise ValueError("Payout source calibration must be true or false.")
    if config.get("payout_source_calibration") and not config.get("template_claim_ensemble"):
        raise ValueError("Payout source calibration requires the template claim ensemble.")
    if type(config.get("pairwise_payout_ranking", False)) is not bool:
        raise ValueError("Pairwise payout ranking must be true or false.")
    if config.get("pairwise_payout_ranking") and (not config.get("payout_source_calibration") or config.get("source_tail_pricing")):
        raise ValueError("Pairwise payout ranking requires payout-source calibration and supplies its own tail pricing.")
    if type(config.get("bonus_malus_timeline", False)) is not bool:
        raise ValueError("Bonus-Malus timeline detection must be true or false.")
    if config.get("bonus_malus_timeline") and not config.get("pairwise_payout_ranking"):
        raise ValueError("Bonus-Malus timeline detection requires pairwise payout ranking.")
    if type(config.get("count_retrieval_refinement", False)) is not bool:
        raise ValueError("Count-retrieval refinement must be true or false.")
    if config.get("count_retrieval_refinement") and not config.get("bonus_malus_timeline"):
        raise ValueError("Count-retrieval refinement requires Bonus-Malus timeline detection.")
    if type(config.get("source_tail_pricing", False)) is not bool:
        raise ValueError("Source-tail pricing must be true or false.")
    if config.get("source_tail_pricing") and not config.get("payout_source_calibration"):
        raise ValueError("Source-tail pricing requires payout source calibration.")
    if type(config.get("local_tail_pricing", False)) is not bool:
        raise ValueError("Local-tail pricing must be true or false.")
    if config.get("local_tail_pricing") and (not config.get("source_tail_pricing") or config.get("cold_profile_fallback")):
        raise ValueError("Local-tail pricing requires source-tail pricing without the cold-profile fallback.")
    if type(config.get("cold_profile_fallback", False)) is not bool:
        raise ValueError("Cold-profile fallback must be true or false.")
    if config.get("cold_profile_fallback") and not config.get("source_tail_pricing"):
        raise ValueError("Cold-profile fallback requires source-tail pricing.")
    if type(config.get("cold_payout_tail_pricing", False)) is not bool:
        raise ValueError("Cold-profile payout-tail pricing must be true or false.")
    if config.get("cold_payout_tail_pricing") and not config.get("cold_profile_fallback"):
        raise ValueError("Cold-profile payout-tail pricing requires the cold-profile fallback.")
    if type(config.get("cold_exposure_hazard", False)) is not bool:
        raise ValueError("Cold exposure hazard must be true or false.")
    if config.get("cold_exposure_hazard") and not config.get("cold_payout_tail_pricing"):
        raise ValueError("Cold exposure hazard requires cold-profile payout-tail pricing.")
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
    baseline = DashboardClaimModel(config["classifier"], claim_probability_threshold=threshold,
                                   severity_method=config["severity_method"], ridge_alpha=config["ridge_alpha"],
                                   severity_scale=config["severity_scale"], amount_cap=config.get("amount_cap"),
                                   random_seed=config["random_seed"], median_payout_pricing=config.get("median_payout_pricing", False),
                                   aggregate_payout_pricing=config.get("aggregate_payout_pricing", False))
    if config.get("profile_history_ensemble"):
        from src.profile_ensemble_model import ProfileHistoryEnsemble
        return ProfileHistoryEnsemble(baseline, claim_probability_threshold=threshold,
                                      n_splits=config.get("n_splits", DEFAULT_CV_FOLDS), random_seed=config["random_seed"])
    if config.get("template_claim_ensemble"):
        from src.template_claim_ensemble import TemplateClaimEnsemble
        ensemble = TemplateClaimEnsemble
        if config.get("payout_source_calibration"):
            from src.payout_source_claim_ensemble import (
                ColdExposureHazardClaimEnsemble, ColdPayoutTailClaimEnsemble, ColdProfileClaimEnsemble,
                LocalTailClaimEnsemble, PayoutSourceClaimEnsemble, SourceTailClaimEnsemble,
            )
            if config.get("count_retrieval_refinement"):
                from src.bonus_malus_claim_ensemble import CountRetrievalTimelineClaimEnsemble

                ensemble = CountRetrievalTimelineClaimEnsemble
            elif config.get("bonus_malus_timeline"):
                from src.bonus_malus_claim_ensemble import PairwiseTimelineClaimEnsemble

                ensemble = PairwiseTimelineClaimEnsemble
            elif config.get("pairwise_payout_ranking"):
                from src.pairwise_payout_ensemble import PairwisePayoutEnsemble

                ensemble = PairwisePayoutEnsemble
            elif config.get("cold_exposure_hazard"):
                ensemble = ColdExposureHazardClaimEnsemble
            elif config.get("cold_payout_tail_pricing"):
                ensemble = ColdPayoutTailClaimEnsemble
            elif config.get("cold_profile_fallback"):
                ensemble = ColdProfileClaimEnsemble
            elif config.get("local_tail_pricing"):
                ensemble = LocalTailClaimEnsemble
            elif config.get("source_tail_pricing"):
                ensemble = SourceTailClaimEnsemble
            else:
                ensemble = PayoutSourceClaimEnsemble
        return ensemble(baseline, claim_probability_threshold=threshold,
                        n_splits=config.get("n_splits", DEFAULT_CV_FOLDS), random_seed=config["random_seed"])
    return baseline


def print_dashboard_configuration(settings):
    config = settings.reproduction
    family = config["classifier"]["family"]
    print(f"Model: {DASHBOARD_CLASSIFIER_LABELS[family]}")
    for key, value in config["classifier"].items():
        if key not in ["name", "family"]:
            print(f"  {key}: {value}")
    print(f"Severity: {DASHBOARD_SEVERITY_LABELS[config['severity_method']]}")
    if config.get("median_payout_pricing"):
        print("Median payout pricing: enabled for profiles with positive claim history")
    if config.get("aggregate_payout_pricing"):
        print("Aggregate payout pricing: enabled for profiles with positive claim history")
    if config.get("profile_history_ensemble"):
        print("Claim detection: out-of-fold profile history ensemble")
    if config.get("template_claim_ensemble"):
        print("Claim detection: payout-template ensemble with expected-F1 cutoff")
        print("Settlement pricing: profile-relative median with fitting-derived cap")
    if config.get("payout_source_calibration"):
        print("Payout source calibration: separate whole-cent and scaled probabilities")
    if config.get("pairwise_payout_ranking"):
        print("Pairwise payout ranking: rerank claim flags using fitting payout histories and quote support")
    if config.get("count_retrieval_refinement"):
        print("Warm refinement: count-hurdle agreement and dual leaf-retrieval admissions")
    if config.get("bonus_malus_timeline"):
        print("Cold detection: calibrated French Bonus-Malus reference-window timeline detector")
    if config.get("source_tail_pricing"):
        print("Payout floor: source-specific fitted lower tails")
    if config.get("local_tail_pricing"):
        print("Local-tail pricing: reduce unsupported low quotes on low-risk profiles")
    if config.get("cold_profile_fallback"):
        print("Profile-disjoint batches: CatBoost on rating variables and batch profile counts")
    if config.get("cold_exposure_hazard"):
        print("Profile-disjoint hazard confirmation and payout-source quotes enabled")
    elif config.get("cold_payout_tail_pricing"):
        print("Profile-disjoint payout floor: pooled fitting 2nd percentile for low-confidence quotes")
    print(f"Ridge alpha: {config['ridge_alpha']}; amount multiplier: {config['severity_scale']}; cap: {config.get('amount_cap')}")
