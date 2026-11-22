"""Fit the deployable Bonus-Malus timeline model once per holdout and cache per-row outputs.

Fitting is the slow step, so this script stores the claim risk, price and flag of every
held-out row for three variants of one fitted model: pairwise only, pairwise with the
timeline, and count retrieval with the timeline. analysis/evaluate_final_candidates.py
then scores pricing and flag-volume variants from these files without refitting.
Splits come from src.reproduce_timeline_models, so seeds 41-45 match the dashboard runs.

Example: python -m analysis.cache_final_model_outputs --cohort matched_20 --seeds 41 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.bonus_malus_claim_ensemble import _align_risk_with_flags
from src.dashboard_config import build_dashboard_model, load_dashboard_settings
from src.dashboard_model import profile_keys
from src.data_loading import load_training_data
from src.feature_schema import ALL_FEATURE_COLUMNS, RANDOM_SEED, ROW_INDEX_COLUMN, TARGET_COLUMN
from src.pairwise_payout_ensemble import PairwisePayoutEnsemble
from src.reproduce_timeline_models import CONFIG, SEEDS, claim_scores, cohort_splits

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = ROOT / "analysis/output/final_search"
COHORTS = ("matched_20", "matched_25")


def cache_path(directory, cohort, seed, model_seed=RANDOM_SEED):
    """Name cached outputs by split seed, adding the model seed only when it differs from the default."""
    suffix = "" if model_seed == RANDOM_SEED else f"_model{model_seed}"
    return Path(directory) / f"{cohort}_seed{seed}{suffix}.csv.gz"


def score_holdout(model, query):
    """Return (risk, price, flag) for each model variant, mirroring the ensembles' _score_batch.

    Reusing one pairwise pass avoids scoring the expensive base layers three times.
    """
    risk, prices, flags, cutoff = PairwisePayoutEnsemble._score_batch(model, query)
    timeline = model.timeline_.augment_predictions(query, risk, prices, flags, cutoff)
    counted_prices, counted_flags = model._refine_warm_flags(query, risk, prices, flags, cutoff)
    counted = model.timeline_.augment_predictions(query, risk, counted_prices, counted_flags, cutoff)
    outputs = {"base": (risk, prices, flags)}
    for name, (variant_risk, variant_prices, variant_flags) in (("pt", timeline), ("cr", counted)):
        outputs[name] = (_align_risk_with_flags(variant_risk, variant_flags, cutoff), variant_prices, variant_flags)
    return outputs, cutoff


def profile_history(fitting, held):
    """Flag held-out rows whose exact profile is in fitting, and those whose profile had a paid claim."""
    fitting_keys, held_keys = profile_keys(fitting), profile_keys(held)
    paid_keys = fitting_keys[fitting[TARGET_COLUMN].to_numpy() > 0]
    return np.isin(held_keys, fitting_keys), np.isin(held_keys, paid_keys)


def held_out_frame(seed, held, outputs, cutoff, seen, warm):
    frame = pd.DataFrame({
        "seed": seed, ROW_INDEX_COLUMN: held[ROW_INDEX_COLUMN].to_numpy(),
        "actual": held[TARGET_COLUMN].to_numpy(), "cutoff": cutoff, "seen": seen, "warm": warm,
    })
    for name, (risk, prices, flags) in outputs.items():
        frame[f"{name}_risk"], frame[f"{name}_price"], frame[f"{name}_flag"] = risk, prices, flags
    return frame


def cache_seed(config, seed, fitting, held, verify):
    model = build_dashboard_model(config, config["threshold"])
    model.fit(fitting[ALL_FEATURE_COLUMNS], fitting[TARGET_COLUMN].to_numpy())
    query = held[ALL_FEATURE_COLUMNS]
    outputs, cutoff = score_holdout(model, query)
    if verify:
        # Guard against drift between this replica and the deployed prediction path.
        _, prices, flags = outputs["cr"]
        if not np.allclose(model.predict(query), np.where(flags, prices, 0.)):
            raise RuntimeError("Cached count-retrieval outputs differ from model.predict().")
    seen, warm = profile_history(fitting, held)
    return held_out_frame(seed, held, outputs, cutoff, seen, warm)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="matched_20", choices=COHORTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--model-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = load_training_data()
    config = {**load_dashboard_settings(CONFIG).reproduction, "random_seed": args.model_seed}
    with threadpool_limits(limits=2):
        for index, (seed, fitting, held) in enumerate(cohort_splits(data, args.cohort, tuple(args.seeds))):
            frame = cache_seed(config, seed, fitting, held, verify=index == 0)
            frame.to_csv(cache_path(args.output_dir, args.cohort, seed, args.model_seed), index=False)
            for name in ("base", "pt", "cr"):
                f1, mae = claim_scores(frame.actual.to_numpy(), np.where(frame[f"{name}_flag"], frame[f"{name}_price"], 0.))
                print(f"{args.cohort} seed {seed} model {args.model_seed} {name:<4} F1 {f1:.6f} MAE {mae:.4f}", flush=True)


if __name__ == "__main__":
    main()
