"""Reproduce the five-seed 20% claim benchmark with its frozen upstream inputs."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.claim_benchmark_pipeline import benchmark_predictions
from src.dashboard_config import load_dashboard_settings
from src.data_loading import DEFAULT_TRAINING_PATH, load_training_data
from src.evaluation import stratified_train_validation_split
from src.feature_schema import ALL_FEATURE_COLUMNS, ROW_INDEX_COLUMN, TARGET_COLUMN

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = Path(__file__).with_name("reference") / "claim_benchmark_20"
CONFIG = ROOT / "configs/claim_history/payout_source_tail_ensemble.json"
EVALUATION_SHARE = .20
SEEDS = (41, 42, 43, 44, 45)


def benchmark_data():
    manifest = json.loads(REFERENCE.with_suffix(".json").read_text(encoding="utf-8"))
    normalized = DEFAULT_TRAINING_PATH.read_bytes().replace(b"\r\n", b"\n")
    if sha256(normalized).hexdigest() != manifest["training_sha256_lf"]:
        raise ValueError("The frozen benchmark requires the original data/trainingset.csv.")
    reference = pd.read_csv(REFERENCE.with_suffix(".csv.gz"), float_precision="round_trip").set_index(["seed", ROW_INDEX_COLUMN])
    if not reference.index.is_unique:
        raise ValueError("The frozen reference must contain one row per seed and row ID.")
    return load_training_data(), reference


def run_seed(seed, data, reference, settings):
    fitting, held, *_ = stratified_train_validation_split(data, data[TARGET_COLUMN], EVALUATION_SHARE, seed)
    cached = reference.loc[seed]
    ids = held[ROW_INDEX_COLUMN]
    if len(cached) != len(ids) or not cached.index.difference(ids).empty:
        raise ValueError(f"The frozen reference rows do not match holdout seed {seed}.")
    cases = cached.loc[ids]
    if cases["cutoff"].nunique() != 1:
        raise ValueError(f"Expected a single reference cutoff for seed {seed}.")
    target, actual = fitting[TARGET_COLUMN].to_numpy(), held[TARGET_COLUMN].to_numpy()
    prediction = benchmark_predictions(settings, fitting[ALL_FEATURE_COLUMNS], target, held[ALL_FEATURE_COLUMNS], cases)
    rows = pd.DataFrame({"seed": seed, ROW_INDEX_COLUMN: ids.to_numpy(), "actual": actual, **prediction})
    return rows, [(name, *scores(actual, amounts)) for name, amounts in prediction.items()]


def scores(actual, amounts):
    claimed, flagged = actual > 0., amounts > 0.
    denominator = claimed.sum() + flagged.sum()
    f1 = 2 * np.count_nonzero(claimed & flagged) / denominator if denominator else 0.
    return float(f1), float(np.abs(actual - amounts).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, help="Optional CSV or CSV.gz path for per-row predictions.")
    args = parser.parse_args()
    data, reference = benchmark_data()
    settings = load_dashboard_settings(CONFIG).reproduction
    metrics, predictions = [], []
    print("20% holdout, seeds 41–45. Refitting downstream models from the original training CSV.", flush=True)
    with threadpool_limits(limits=2):
        for seed in SEEDS:
            rows, result = run_seed(seed, data, reference, settings)
            for name, f1, mae in result:
                print(f"{seed}  {name:<28} F1 {f1:.8f}  MAE {mae:.8f}", flush=True)
                metrics.append((name, f1, mae))
            if args.predictions:
                predictions.append(rows)
    if args.predictions:
        pd.concat(predictions, ignore_index=True).to_csv(args.predictions, index=False)
    summary = pd.DataFrame(metrics, columns=["policy", "F1", "MAE"]).groupby("policy", sort=False).mean()
    print("\nFive-seed means")
    print(summary.to_string(float_format=lambda value: f"{value:.8f}"))


if __name__ == "__main__":
    main()
