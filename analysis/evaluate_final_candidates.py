"""Score pricing and flag-volume variants of the cached timeline models.

Reads per-row outputs written by analysis/cache_final_model_outputs.py, applies each
variant, and reports five-seed means of F1 and MAE. Variants are ranked by mean F1,
then by mean MAE, so a pricing change that keeps every flag ranks by its MAE alone.

Example: python -m analysis.evaluate_final_candidates --cohort matched_20 --seeds 41 42 43 44 45
"""

import argparse
from itertools import product

import numpy as np
import pandas as pd

from analysis.cache_final_model_outputs import OUTPUT_DIRECTORY, cache_path
from src.feature_schema import ROW_INDEX_COLUMN
from src.reproduce_timeline_models import SEEDS, claim_scores

# Counts as a claim for F1 but changes MAE by at most $1. Matches the timeline detector's floor.
NEAR_ZERO_PRICE = 1.0
MODELS = ("pt", "cr")  # pairwise + timeline, count retrieval + timeline
LOW_CUTS = (None, .40, .45, .50, .55, .60)  # below this risk a flagged row gets NEAR_ZERO_PRICE
MULTIPLIERS = (.9, 1., 1.1)  # scales the remaining real prices
VOLUMES = (.95, 1., 1.05, 1.10, 1.15)  # scales the number of flags


def bagged_frame(cohort, seed, model_seeds):
    """Average count-retrieval outputs of fits that differ only in model seed.

    A majority vote keeps each fit's own flag logic, while averaged risks and prices
    smooth out fold-to-fold noise in the stacked calibrators.
    """
    frames = [pd.read_csv(cache_path(OUTPUT_DIRECTORY, cohort, seed, model_seed)) for model_seed in model_seeds]
    if any(not frame[ROW_INDEX_COLUMN].equals(frames[0][ROW_INDEX_COLUMN]) for frame in frames):
        raise ValueError(f"Cached rows differ across model seeds for split seed {seed}.")
    votes = np.sum([frame.cr_flag.astype(bool) for frame in frames], axis=0)
    return frames[0].assign(bag_risk=np.mean([frame.cr_risk for frame in frames], axis=0),
                            bag_price=np.mean([frame.cr_price for frame in frames], axis=0),
                            bag_flag=votes * 2 > len(frames))


def adjust_volume(risk, flags, volume):
    """Scale the flag count, adding the riskiest unflagged rows or dropping the least risky flags."""
    target, current = int(round(flags.sum() * volume)), int(flags.sum())
    adjusted = flags.copy()
    if target > current:
        candidates = np.flatnonzero(~flags)
        adjusted[candidates[np.argsort(-risk[candidates], kind="stable")][:target - current]] = True
    elif target < current:
        flagged = np.flatnonzero(flags)
        adjusted[flagged[np.argsort(risk[flagged], kind="stable")][:current - target]] = False
    return adjusted


def variant_predictions(frame, model, low_cut, multiplier, volume):
    risk = frame[f"{model}_risk"].to_numpy()
    flags = frame[f"{model}_flag"].to_numpy().astype(bool)
    prices = frame[f"{model}_price"].to_numpy() * multiplier
    adjusted = adjust_volume(risk, flags, volume)
    # Added rows sit below the F1 cutoff, so a real amount would usually cost MAE.
    near_zero = adjusted & ~flags
    if low_cut is not None:
        near_zero |= risk < low_cut
    return np.where(adjusted, np.where(near_zero, NEAR_ZERO_PRICE, prices), 0.)


def score_variant(frames, settings):
    rows = []
    for frame in frames:
        actual = frame.actual.to_numpy()
        f1, mae = claim_scores(actual, variant_predictions(frame, *settings))
        rows.append((f1, mae, 1 - mae / np.abs(actual).mean()))
    f1, mae, improvement = np.mean(rows, axis=0)
    return dict(zip(("model", "low_cut", "multiplier", "volume"), settings),
                mean_f1=f1, mean_mae=mae, improvement=improvement, worst_seed_f1=min(r[0] for r in rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", default="matched_20")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--bag-model-seeds", nargs="+", type=int,
                        help="Also score a majority-vote blend of fits with these model seeds.")
    args = parser.parse_args(argv)
    models = MODELS
    if args.bag_model_seeds:
        frames = [bagged_frame(args.cohort, seed, args.bag_model_seeds) for seed in args.seeds]
        models = (*MODELS, "bag")
    else:
        frames = [pd.read_csv(cache_path(OUTPUT_DIRECTORY, args.cohort, seed)) for seed in args.seeds]
    results = pd.DataFrame([score_variant(frames, settings)
                            for settings in product(models, LOW_CUTS, MULTIPLIERS, VOLUMES)])
    results = results.sort_values(["mean_f1", "mean_mae"], ascending=[False, True])
    seeds = "-".join(map(str, args.seeds)) + ("_bag" if args.bag_model_seeds else "")
    results.to_csv(OUTPUT_DIRECTORY / f"{args.cohort}_variants_{seeds}.csv", index=False)
    baseline = results[results.low_cut.isna() & results.multiplier.eq(1.) & results.volume.eq(1.)]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f"As-is models ({args.cohort}, seeds {seeds}):\n{baseline.round(4).to_string(index=False)}")
        print(f"\nTop {args.top} variants:\n{results.head(args.top).round(4).to_string(index=False)}")


if __name__ == "__main__":
    main()
