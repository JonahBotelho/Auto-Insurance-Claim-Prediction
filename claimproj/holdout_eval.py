"""Score the ten submission settings on fresh holdout splits of the training data.

For each split: hold out 20% of trainingset.csv, train on the other 80%, predict the held-out rows with each of the
ten settings in gen_submissions.py, and compute MAE. Because the held-out rows have real ClaimAmount values, this is a
true out-of-sample score (unlike the Hub's test set, which you only see after submitting).

Usage:  python holdout_eval.py [--splits 3]
Output: results/holdout_summary.csv and a printed table.
  holdout_mae  = average MAE of that setting over the splits (noisy: a few huge claims swing it between splits)
  gain_vs_zeros = holdout_mae of that setting MINUS holdout_mae of predicting 0 for everyone, on the same rows.
                  Negative = better than all zeros. This is the number to compare settings with.
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from common import (load_train, make_model, group_keys, fit_group_stats, add_group_feats,
                    fit_group_amounts, residual_dist, decide_group)

# same settings as gen_submissions.py: file number -> (model, threshold, amount scale); None = all zeros
PLAN = {1: None, 2: ("lgbm", .55, 1.0), 3: ("lgbm", .55, .9), 4: ("lgbm", .55, .8), 5: ("lgbm", .50, .8),
        6: ("lgbm", .50, 1.0), 7: ("lgbm", .60, 1.0), 8: ("cat", .60, .8), 9: ("blend", .50, .8),
        10: ("blend", .60, .8)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=3)
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)
    X, y = load_train(); b = (y > 0).astype(int); keys = group_keys(X)
    res = {i: [] for i in PLAN}
    for s in range(a.splits):
        seed = 101 + s
        tr, va = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=seed).split(X, b))
        st = fit_group_stats(keys[tr], b[tr])  # group history comes from the 80% only
        Xtr = add_group_feats(X.iloc[tr], keys[tr], st, own_b=b[tr])
        Xva = add_group_feats(X.iloc[va], keys[va], st)
        p = {n: make_model(n, seed).fit(Xtr, b[tr]).predict_proba(Xva)[:, 1] for n in ("lgbm", "cat")}
        p["blend"] = (p["lgbm"] + p["cat"]) / 2
        logs = fit_group_amounts(keys[tr], y[tr]); resid = residual_dist(logs)
        pos = np.sort(y[tr][y[tr] > 0])
        for i, spec in PLAN.items():
            pred = np.zeros(len(va)) if spec is None else decide_group(
                p[spec[0]], keys[va], logs, resid, pos, spec[1], scale=spec[2])
            res[i].append((np.abs(y[va] - pred).mean(), int((pred > 0).sum())))
    rows = [dict(file=f"1_2_{i}", setting=str(PLAN[i]),
                 holdout_mae=np.mean([r[0] for r in rs]), best_split=np.min([r[0] for r in rs]),
                 worst_split=np.max([r[0] for r in rs]), avg_flagged=np.mean([r[1] for r in rs]))
            for i, rs in res.items()]
    df = pd.DataFrame(rows)
    df["gain_vs_zeros"] = df["holdout_mae"] - df.loc[0, "holdout_mae"]
    df.to_csv("results/holdout_summary.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()