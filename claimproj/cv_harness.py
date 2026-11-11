"""Repeated stratified CV scored with the real metric (MAE).
Usage: python cv_harness.py [--repeats 3] [--models lgbm,cat] [--no-group]
Writes results/cv_summary.csv and results/oof_<model>.npy.
amount_rule: 'group' predicts flagged rows from the group's past claims; 'global' uses a global quantile.
"""
import argparse, os
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from common import (load_train, make_model, decide, decide_group, group_keys, fit_group_stats,
                    add_group_feats, fit_group_amounts, residual_dist)

THRESHOLDS = [0.5, 0.55, 0.6, 0.7]
SCALES = [0.8, 0.9, 1.0]  # multiplier on the predicted amount for flagged rows (group rule only)


def run_folds(name, X, y, b, keys, seed, use_group=True, folds=5):
    oof = np.zeros(len(b)); info = []
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, b):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        if use_group:
            st = fit_group_stats(keys[tr], b[tr])
            Xtr = add_group_feats(Xtr, keys[tr], st, own_b=b[tr])
            Xva = add_group_feats(Xva, keys[va], st)
        oof[va] = make_model(name, seed).fit(Xtr, b[tr]).predict_proba(Xva)[:, 1]
        logs = fit_group_amounts(keys[tr], y[tr])
        info.append((va, logs, residual_dist(logs), np.sort(y[tr][y[tr] > 0])))
    return oof, info


def calibrate_oof(p, b, seed):
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    return cross_val_predict(iso, p, b, cv=StratifiedKFold(5, shuffle=True, random_state=seed))


def mae_for(p, y, keys, info, t, rule, scale=1.0):
    pred = np.zeros(len(y))
    for va, logs, resid, pos in info:
        if rule == "group":
            pred[va] = decide_group(p[va], keys[va], logs, resid, pos, t, scale=scale)
        else:
            pred[va] = decide(p[va], pos, t)
    return np.abs(y - pred).mean(), int((pred > 0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--models", default="lgbm,cat")
    ap.add_argument("--no-group", action="store_true")
    a = ap.parse_args()
    os.makedirs("results", exist_ok=True)

    X, y = load_train(); b = (y > 0).astype(int); keys = group_keys(X)
    base = np.abs(y).mean()
    print(f"all-zero baseline MAE: {base:.4f}  (positives: {b.sum()} of {len(b)})")
    names = a.models.split(",")
    rows, store = [], {n: [] for n in names}
    for seed in range(a.repeats):
        ps, info = {}, None
        for n in names:
            ps[n], info = run_folds(n, X, y, b, keys, seed, not a.no_group)  # same seed -> same folds for all models
            store[n].append(ps[n])
            print(f"seed {seed} {n}: AUC {roc_auc_score(b, ps[n]):.4f}  logloss {log_loss(b, ps[n]):.5f}  max p {ps[n].max():.3f}")
        if len(names) > 1:
            ps["blend"] = np.mean([ps[n] for n in names], axis=0)
        for n, p in ps.items():
            for calib, pp in (("raw", p), ("iso", calibrate_oof(p, b, seed))):
                for rule, scales in (("group", SCALES), ("global", [1.0])):
                    for t in THRESHOLDS:
                        for sc in scales:
                            m, nf = mae_for(pp, y, keys, info, t, rule, sc)
                            rows.append(dict(model=n, calib=calib, amount_rule=rule, threshold=t, scale=sc, mae=m, n_flagged=nf))
    df = pd.DataFrame(rows)
    summ = (df.groupby(["model", "calib", "amount_rule", "threshold", "scale"])
              .agg(mae=("mae", "mean"), mae_worst=("mae", "max"), n_flagged=("n_flagged", "mean"))
              .reset_index())
    summ["vs_baseline"] = summ["mae"] - base
    summ = summ.sort_values("mae")
    summ.to_csv("results/cv_summary.csv", index=False)
    for n in names:
        np.save(f"results/oof_{n}.npy", np.mean(store[n], axis=0))
    print("\nBest 15 settings (negative vs_baseline beats all-zero):")
    print(summ.head(15).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
