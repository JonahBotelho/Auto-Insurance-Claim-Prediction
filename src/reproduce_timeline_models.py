"""Natively reproduce the five-seed Bonus-Malus timeline runs from src/ without analysis/ files."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from threadpoolctl import threadpool_limits

from src.dashboard_config import build_dashboard_model, load_dashboard_settings
from src.dashboard_model import profile_keys
from src.data_loading import load_training_data
from src.evaluation import stratified_train_validation_split
from src.feature_schema import ALL_FEATURE_COLUMNS, ROW_INDEX_COLUMN, TARGET_COLUMN
from src.pairwise_payout_ensemble import PairwisePayoutEnsemble

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/claim_history/count_retrieval_bm_timeline_ensemble.json"
SEEDS = (41, 42, 43, 44, 45)
ALL_COHORTS = ("matched_20", "matched_25", "mixed_25", "disjoint_25")


def claim_scores(actual, predicted):
    claimed, flagged = actual > 0.0, predicted > 0.0
    denom = int(claimed.sum() + flagged.sum())
    f1 = 2.0 * np.count_nonzero(claimed & flagged) / denom if denom else 0.0
    return float(f1), float(np.abs(actual - predicted).mean())


def cohort_splits(data, cohort, seeds):
    """Build deterministic holdout partitions using only data/trainingset.csv."""
    if cohort == "matched_20":
        return [(s, *stratified_train_validation_split(data, data[TARGET_COLUMN], 0.20, s)[:2]) for s in seeds]
    if cohort == "matched_25":
        return [(s, *stratified_train_validation_split(data, data[TARGET_COLUMN], 0.25, s)[:2]) for s in seeds]
    if cohort == "mixed_25":
        keys = profile_keys(data)
        size = len(data) // 4
        has_claim = data[TARGET_COLUMN].gt(0).to_numpy()
        splits = []
        for s in seeds:
            remaining, unseen = next(GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=s).split(data, groups=keys))
            fitting, ordinary = train_test_split(remaining, test_size=size - len(unseen), random_state=s, stratify=has_claim[remaining])
            held = np.r_[unseen, ordinary]
            splits.append((s, data.iloc[np.sort(fitting)], data.iloc[held]))
        return splits
    if cohort == "disjoint_25":
        keys = profile_keys(data)
        unique_keys = np.unique(keys)
        splits = []
        for s in seeds:
            rng = np.random.default_rng(s)
            held_keys = set(rng.choice(unique_keys, size=int(round(0.25 * len(unique_keys))), replace=False))
            is_held = np.isin(keys, list(held_keys))
            splits.append((s, data.iloc[~is_held], data.iloc[is_held]))
        return splits
    raise ValueError(f"Unsupported cohort '{cohort}'. Choose from {ALL_COHORTS}.")


def run_cohort(data, config, cohort, seeds, *, show_components=True):
    print(f"\nCohort: {cohort} (seeds {', '.join(map(str, seeds))})", flush=True)
    metrics, frames = [], []
    with threadpool_limits(limits=2):
        for seed, fitting, held in cohort_splits(data, cohort, seeds):
            model = build_dashboard_model(config, config["threshold"])
            features, query = fitting[ALL_FEATURE_COLUMNS], held[ALL_FEATURE_COLUMNS]
            target, actual = fitting[TARGET_COLUMN].to_numpy(), held[TARGET_COLUMN].to_numpy()
            model.fit(features, target)

            if show_components:
                base_risk, base_prices, base_flags, base_cut = PairwisePayoutEnsemble._score_batch(model, query)
                _, pt_prices, pt_flags = model.timeline_.augment_predictions(query, base_risk, base_prices, base_flags, base_cut)
                predictions = {
                    "pairwise_base": np.where(base_flags, base_prices, 0.0),
                    "pairwise_bm_timeline": np.where(pt_flags, pt_prices, 0.0),
                    "count_retrieval_bm_timeline": model.predict(query),
                }
            else:
                predictions = {"configured": model.predict(query)}
            for name, amounts in predictions.items():
                f1, mae = claim_scores(actual, amounts)
                print(f"{seed}  {name:<28} F1 {f1:.8f}  MAE {mae:.8f}", flush=True)
                metrics.append((name, f1, mae))
            frames.append(pd.DataFrame({
                "cohort": cohort, "seed": seed, ROW_INDEX_COLUMN: held[ROW_INDEX_COLUMN].to_numpy(),
                "actual": actual, **predictions,
            }))
    summary = pd.DataFrame(metrics, columns=["model", "F1", "MAE"]).groupby("model", sort=False).mean()
    print(f"\nMean ({cohort})")
    print(summary.to_string(float_format=lambda v: f"{v:.8f}"), flush=True)
    return pd.concat(frames, ignore_index=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", nargs="+", default=["matched_20"], choices=ALL_COHORTS)
    parser.add_argument("--config", type=Path, help="Benchmark a saved native configuration.")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--predictions", type=Path, help="Optional CSV or CSV.gz path for per-row predictions.")
    args = parser.parse_args(argv)
    data = load_training_data()
    config = load_dashboard_settings(args.config or CONFIG).reproduction
    outputs = [run_cohort(data, config, cohort, tuple(args.seeds), show_components=args.config is None)
               for cohort in args.cohorts]
    if args.predictions:
        pd.concat(outputs, ignore_index=True).to_csv(args.predictions, index=False)


if __name__ == "__main__":
    main()
