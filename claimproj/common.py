"""Shared helpers: data loading, model factories, and the MAE decision rule."""
import numpy as np
import pandas as pd

TARGET = "ClaimAmount"
ID_COL = "rowIndex"
SEED = 42


def load_train(path="data/trainingset.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=[ID_COL, TARGET])
    return X, df[TARGET].to_numpy()


def load_test(path="data/testset.csv"):
    df = pd.read_csv(path)
    return df[ID_COL].to_numpy(), df.drop(columns=[ID_COL])


def make_model(name, seed=SEED):
    """Deliberately conservative settings: the signal is weak and positives are rare."""
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=700, learning_rate=0.015, num_leaves=8,
            min_child_samples=100, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=10.0, random_state=seed,
            n_jobs=1, verbose=-1)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=3,
            min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
            reg_lambda=10.0, random_state=seed, n_jobs=1, tree_method="hist",
            eval_metric="logloss")
    if name == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=400, learning_rate=0.04, depth=4, l2_leaf_reg=10,
            random_seed=seed, verbose=0, thread_count=1, allow_writing_files=False)
    if name == "logit":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=500))
    raise ValueError(name)


def decide(p, pos_sorted, threshold):
    """MAE-optimal prediction given P(claim>0)=p.

    The median of the claim distribution is 0 unless p > 0.5. Above that, it is the
    ((p - 0.5) / p) quantile of the positive amounts. `threshold` (>= 0.5) adds a safety
    margin because the estimated p is noisy.
    """
    p = np.asarray(p, dtype=float)
    pred = np.zeros_like(p)
    flag = p > threshold
    if flag.any():
        q = np.clip((p[flag] - 0.5) / p[flag], 0.0, 1.0)
        pred[flag] = np.quantile(pos_sorted, q)
    return pred


# ---------- group (repeat-customer) features ----------
# 83% of test rows have an exact feature match in the training set, and matching rows tend to
# claim together, so a group's claim history is the strongest signal we found (AUC 0.71 -> 0.87).

def group_keys(X):
    return X.astype(str).agg("|".join, axis=1).to_numpy()


def fit_group_stats(keys, b):
    """key -> (claims, rows) using labelled rows only."""
    df = pd.DataFrame({"k": keys, "b": b}).groupby("k").b.agg(["sum", "size"])
    return {k: (int(s), int(n)) for k, s, n in zip(df.index, df["sum"], df["size"])}


def add_group_feats(X, keys, stats, own_b=None):
    """Append g_claims / g_n / g_rate. Pass own_b for the rows the stats were fitted on so each
    row's own label is removed (leave-one-out); leave it None for unseen rows."""
    s = np.array([stats.get(k, (0, 0))[0] for k in keys], dtype=float)
    n = np.array([stats.get(k, (0, 0))[1] for k in keys], dtype=float)
    if own_b is not None:
        s, n = s - own_b, n - 1
    out = X.copy()
    out["g_claims"], out["g_n"] = s, n
    out["g_rate"] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out


# ---------- group-aware claim AMOUNT (the biggest MAE lever) ----------
# Claims from the same group are very similar in size (corr of log amounts ~0.86), so for rows we flag
# we predict from the group's past claims instead of a global quantile.

def fit_group_amounts(keys, y):
    df = pd.DataFrame({"k": keys, "y": y})
    df = df[df.y > 0]
    return {k: np.log(g.to_numpy()) for k, g in df.groupby("k").y}


def residual_dist(logs):
    """Sorted (log claim - median log of the group's OTHER claims), over groups with >= 2 claims."""
    r = []
    for arr in logs.values():
        if len(arr) < 2:
            continue
        for i in range(len(arr)):
            r.append(arr[i] - np.median(np.delete(arr, i)))
    return np.sort(np.array(r))


def decide_group(p, keys, logs, resid, pos_sorted, threshold, scale=1.0):
    # scale multiplies the predicted amount; 1.0 was best at threshold 0.55 in CV (0.8-0.9 was best at 0.50)
    p = np.asarray(p, dtype=float)
    pred = np.zeros_like(p)
    for i in np.where(p > threshold)[0]:
        q = float(np.clip((p[i] - 0.5) / p[i], 0.0, 1.0))
        arr = logs.get(keys[i])
        if arr is not None and len(resid):
            pred[i] = scale * np.exp(np.median(arr) + np.quantile(resid, q))
        else:
            pred[i] = scale * np.quantile(pos_sorted, q)
    return pred
