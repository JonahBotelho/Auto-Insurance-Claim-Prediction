"""Write predictions for a feature file using the saved model.
Usage: python predict.py [--input data/testset.csv] [--output predictions.csv] [--zeros]
--zeros writes the all-zero baseline instead (a safe first submission).
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from common import load_test, group_keys, add_group_feats, decide_group, ID_COL, TARGET


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/testset.csv")
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--zeros", action="store_true")
    a = ap.parse_args()
    ids, X = load_test(a.input)
    if a.zeros:
        pred = np.zeros(len(X))
    else:
        art = joblib.load("model.joblib")
        X = X[art["columns"]]
        Xg = add_group_feats(X, group_keys(X), art["stats"])
        p = art["model"].predict_proba(Xg)[:, 1]
        pred = decide_group(p, group_keys(X), art["logs"], art["resid"], art["pos_sorted"], art["threshold"])
    pd.DataFrame({ID_COL: ids, TARGET: pred}).to_csv(a.output, index=False)
    print(f"wrote {a.output}: {len(pred)} rows, {(pred > 0).sum()} nonzero")


if __name__ == "__main__":
    main()
