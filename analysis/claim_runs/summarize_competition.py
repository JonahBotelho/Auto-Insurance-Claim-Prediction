"""Select rules on 60% of exact profile IDs and check them on the other 40%."""
import argparse
from pathlib import Path
from collections import Counter
import sys

import numpy as np
import pandas as pd

from compare_competition import PRICERS, predict_rule, score


def attach_support(data, paid, keys):
    """Find the closest supported payout using only this split's fitting labels."""
    from src.masked_payout_history import observed_support
    from src.payout_pricing import payout_signature
    lookup = {}
    for key, amount in paid:
        lookup.setdefault(key, Counter())[payout_signature(amount)] += 1
    possible = {k: observed_support(v) for k, v in lookup.items()}
    nearest = np.full(len(data), np.nan)
    for index in np.flatnonzero(data.price.to_numpy() > 1):
        values = possible.get(keys[index])
        if values is not None:
            nearest[index] = values[np.argmin(np.abs(data.price.iat[index] - values))]
    return data.assign(support=nearest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--export-rule", nargs=2, metavar=("DETECTOR", "PRICER"),
                        help="Apply a chosen rule to the full-data model's test_components.csv.gz")
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo.resolve()))
    train = pd.read_csv(args.repo / "data/trainingset.csv")
    features = [f"feature{i}" for i in range(1, 19)]
    keys = pd.util.hash_pandas_object(train[features], index=False).to_numpy()
    if args.export_rule:
        test = pd.read_csv(args.repo / "data/testset.csv")
        frame = pd.read_csv(args.results / "test_components.csv.gz", float_precision="round_trip")
        assert np.array_equal(frame.rowIndex, test.rowIndex) and frame.rowIndex.is_unique
        positives = train.ClaimAmount.gt(0)
        query_keys = pd.util.hash_pandas_object(test[features], index=False).to_numpy()
        frame = attach_support(frame, zip(keys[positives], train.loc[positives, "ClaimAmount"]), query_keys)
        output = predict_rule(frame, *args.export_rule)
        assert np.isfinite(output).all() and (output >= 0).all()
        path = args.results / "test_predictions.csv"
        pd.DataFrame(dict(rowIndex=frame.rowIndex, ClaimAmount=output)).to_csv(path, index=False)
        print(f"Wrote {path}: {len(output)} predictions, {(output > 0).sum()} positive.")
        return
    group = pd.Series(np.where(keys % 10 < 6, "selection", "check"), index=train.rowIndex)
    records = []
    id_keys = pd.Series(keys, index=train.rowIndex)
    pricers = PRICERS + ("snap10", "snap25", "snap25_high")
    fresh_seeds = [s for s in range(151,156) if (args.results / f"components-{s}.csv.gz").exists()]
    for seed in [*range(41, 46), *fresh_seeds]:
        data = pd.read_csv(args.results / f"components-{seed}.csv.gz", float_precision="round_trip")
        assert len(data) == 14000 and data.rowIndex.is_unique
        fitting_paid = (~train.rowIndex.isin(data.rowIndex)) & (train.ClaimAmount > 0)
        data = attach_support(data, zip(keys[fitting_paid], train.loc[fitting_paid, "ClaimAmount"]),
                              data.rowIndex.map(id_keys).to_numpy())
        prefix = "fresh_" if seed in fresh_seeds else ""
        for cohort, frame in data.groupby(data.rowIndex.map(group)):
            records.extend(dict(cohort=prefix+cohort, **r) for r in score(frame, pricers))
        all_scores = score(data, pricers)
        records.extend(dict(cohort=prefix+"all", **r) for r in all_scores)
        pd.DataFrame(all_scores).to_csv(args.results/f"scores-{seed}.csv", index=False)
    scores = pd.DataFrame(records)
    scores.to_csv(args.results / "per_seed.csv", index=False)
    means = scores.groupby(["cohort", "detector", "pricer"])[["mae", "f1", "flagged"]].mean().reset_index()
    means.to_csv(args.results / "summary.csv", index=False)
    dev = means[means.cohort == "selection"]
    base = dev[(dev.detector == "original") & (dev.pricer == "original")].iloc[0]
    highest = dev.sort_values(["f1", "mae"], ascending=[False, True]).iloc[0]
    lowest = dev.sort_values(["mae", "f1"], ascending=[True, False]).iloc[0]
    protected = dev[dev.f1 >= base.f1 - 1e-12].sort_values(["mae", "f1"], ascending=[True, False]).iloc[0]
    same_flags = dev[dev.detector == "original"].sort_values("mae").iloc[0]
    for name, candidate in [("baseline", base), ("lowest MAE", lowest), ("highest F1", highest), ("lowest MAE with baseline F1", protected), ("keep every original flag", same_flags)]:
        print(name, candidate.detector, candidate.pricer)
        print(means[(means.detector == candidate.detector) & (means.pricer == candidate.pricer)].to_string(index=False))


if __name__ == "__main__":
    main()
