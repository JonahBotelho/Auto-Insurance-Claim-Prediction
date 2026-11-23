"""Compare claim detection and amount rules around Group 2's run 1190003.

Use the project's Python environment with its optional requirements installed.
Run: python compare_competition.py PATH_TO_PROJECT [--seeds 41 42 43 44 45]
Writes held-out components and per-seed scores beside this script.
"""
import argparse
import sys
from pathlib import Path
from time import monotonic

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
DETECTORS = ("original", "cold_loose", "cold_strict")
PRICERS = ("original", "scale_090", "scale_110", "near_half", "near_three_quarters")


def predict_rule(frame, detector, pricer):
    """Return amounts from the saved native risk, quotes and cold-profile scores."""
    flags = frame.flag.to_numpy(bool).copy()
    prices = frame.price.to_numpy(float).copy()
    cold = frame.cold.to_numpy(bool)
    cold_p = frame.cold_risk.to_numpy()
    cutoff = frame.cutoff.to_numpy()
    if detector == "cold_loose":
        added = ~flags & cold & (cold_p >= np.maximum(.12, cutoff - .025))
        flags |= added
        prices[added] = frame.cold_quote.to_numpy()[added]
    elif detector == "cold_strict":
        flags &= ~(cold & (cold_p < cutoff + .025) & (frame.risk.to_numpy() < cutoff + .025))
    minimum = np.minimum(1., prices)
    if pricer == "scale_090":
        prices *= .90
    elif pricer == "scale_110":
        prices *= 1.10
    elif pricer in ("near_half", "near_three_quarters"):
        borderline = frame.risk.to_numpy() < cutoff + .05
        prices[borderline] *= .5 if pricer == "near_half" else .75
    elif pricer.startswith("snap"):
        support = frame.support.to_numpy()
        radius = .10 if pricer == "snap10" else .25
        eligible = (prices > 1) & np.isfinite(support) & (np.abs(support-prices) <= radius*prices)
        if pricer == "snap25_high":
            eligible &= frame.risk.to_numpy() >= .50
        prices[eligible] = support[eligible]
    return np.where(flags, np.maximum(minimum, prices), 0.0)


def fit_and_score(fit, held, config, label):
    from src.dashboard_config import build_dashboard_model
    from src.feature_schema import ALL_FEATURE_COLUMNS
    fit_features = fit[ALL_FEATURE_COLUMNS]
    query = held[ALL_FEATURE_COLUMNS]
    model = build_dashboard_model(config, config["threshold"])
    start = monotonic()
    print(f"Fitting {label}", flush=True)
    model.fit(fit_features, fit.ClaimAmount.to_numpy())
    print(f"Scoring {label} after {monotonic()-start:.0f} seconds of training", flush=True)
    risk, quote, flag, cutoff = model.score_batch(query)
    cold, cold_risk = model.timeline_.predict_cold_risk(query)
    cold_quote = model.timeline_.quote_cold_claims(query, cold_risk)
    result = pd.DataFrame(dict(rowIndex=held.rowIndex.to_numpy(), risk=risk, price=quote, flag=flag, cutoff=cutoff, cold=cold,
                               cold_risk=cold_risk, cold_quote=cold_quote))
    if not np.isfinite(result.select_dtypes("number")).all().all():
        raise ValueError("Nonfinite scoring component")
    return result


def score(frame, pricers=PRICERS):
    actual = frame.actual.to_numpy()
    records = []
    for detector in DETECTORS:
        for pricer in pricers:
            prediction = predict_rule(frame, detector, pricer)
            positive = prediction > 0
            tp = (positive & (actual > 0)).sum()
            f1 = 2 * tp / (positive.sum() + (actual > 0).sum())
            records.append(dict(seed=int(frame.seed.iloc[0]), detector=detector, pricer=pricer,
                                mae=np.abs(actual-prediction).mean(), f1=f1, flagged=positive.sum()))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(41,46)))
    parser.add_argument("--output", type=Path, default=HERE / "competition")
    parser.add_argument("--test-components", action="store_true", help="Fit on all labels and score data/testset.csv")
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    from src.dashboard_config import load_dashboard_settings
    from src.data_loading import load_training_data, load_test_data
    config = load_dashboard_settings(args.repo/"configs/claim_history/em_timeline_discount_quantile_ensemble.json").reproduction
    data = load_training_data(args.repo/"data/trainingset.csv")
    args.output.mkdir(exist_ok=True, parents=True)
    with threadpool_limits(limits=2):
        if args.test_components:
            test = load_test_data(args.repo/"data/testset.csv")
            frame = fit_and_score(data, test, config, "full training set")
            frame.to_csv(args.output/"test_components.csv.gz", index=False)
            return
        for seed in args.seeds:
            fit, held = train_test_split(data, test_size=.2, stratify=data.ClaimAmount.gt(0), random_state=seed)
            frame = fit_and_score(fit, held, config, f"seed {seed}").assign(seed=seed, actual=held.ClaimAmount.to_numpy())
            records = pd.DataFrame(score(frame))
            records.to_csv(args.output/f"scores-{seed}.csv", index=False)
            frame.to_csv(args.output/f"components-{seed}.csv.gz", index=False)
            print(records.iloc[0].to_dict(), flush=True)


if __name__ == "__main__":
    main()
