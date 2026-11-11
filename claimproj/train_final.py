"""Fit the final model on all of trainingset.csv and save model.joblib.
Usage from the repository root: python -m claimproj.train_final [--threshold 0.55]
"""
import argparse
import joblib
import numpy as np
from claimproj.common import (load_train, make_model, group_keys, fit_group_stats, add_group_feats,
                    fit_group_amounts, residual_dist)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.55,
                    help="only predict a nonzero claim when P(claim) exceeds this (>= 0.5)")
    a = ap.parse_args()
    X, y = load_train()
    b = (y > 0).astype(int)
    keys = group_keys(X)
    stats = fit_group_stats(keys, b)
    Xg = add_group_feats(X, keys, stats, own_b=b)  # leave-one-out on the training rows
    model = make_model("lgbm").fit(Xg, b)
    logs = fit_group_amounts(keys, y)
    joblib.dump(dict(model=model, stats=stats, logs=logs, resid=residual_dist(logs), pos_sorted=np.sort(y[y > 0]),
                     threshold=a.threshold, columns=list(X.columns)), "model.joblib")
    print(f"saved model.joblib (threshold {a.threshold}, {len(stats)} feature groups)")


if __name__ == "__main__":
    main()
