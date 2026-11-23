"""Masked-profile boosting with cross-fitted claim history and scaled median prices.

Run with the repository environment: python alternative.py --repo PATH --out PATH.
Both outer cohorts use only trainingset.csv. Timeline geometry uses the 18 public
covariates of the prediction batch; only labels in the fitting fold enter history.
"""
import argparse
import json
import os
from pathlib import Path
import sys

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[var] = "2"

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import StratifiedKFold, train_test_split
from threadpoolctl import threadpool_limits


def geometry(frame):
    from src.bonus_malus_timeline import _summarize_profiles, canonical_profile_keys
    from src.bonus_malus_timeline_v3 import build_timeline_graph_v3

    keys = canonical_profile_keys(frame)
    profiles = _summarize_profiles(frame, keys)
    graph, _, _ = build_timeline_graph_v3(profiles, competing_discounts=True)
    graph_columns = ["tier", "bm_ratio", "delay", "overlap_months", "exact_bm_rule",
                     "feature1_n", "n_portfolio_n", "jump_link_count", "bwd_exp", "fwd_exp"]
    graph_features = graph.reindex(keys)[graph_columns].to_numpy(dtype=float)
    immutable = np.column_stack([frame.to_numpy(), graph_features])
    _, profile_id = np.unique(keys, return_inverse=True)
    return immutable, profile_id


def history_features(history_idx, query_idx, ids, amounts):
    n_profiles = ids.max() + 1
    fit_counts = np.bincount(ids[history_idx], minlength=n_profiles)
    query_counts = np.bincount(ids[query_idx], minlength=n_profiles)
    positive_idx = history_idx[amounts[history_idx] > 0]
    bags = {}
    for key, value in zip(ids[positive_idx], amounts[positive_idx]):
        bags.setdefault(key, []).append(value)
    extra = np.zeros((n_profiles, 14))
    anchor = np.full(n_profiles, np.median(amounts[positive_idx]))
    for key, bag in bags.items():
        observed = np.asarray(bag)
        whole = np.abs(observed - np.round(observed, 2)) < 1e-7
        basic = np.round(np.where(whole, observed, observed / 1.178247734) * 100)
        values, copies = np.unique(basic, return_counts=True)
        direct = np.unique(basic[whole])
        inflated = np.unique(basic[~whole])
        anchor[key] = np.median(basic) / 100
        extra[key] = (len(bag), whole.sum(), len(values), len(direct), len(inflated),
                      len(set(direct).intersection(inflated)), copies.max(), np.sum(copies >= 2),
                      bool(len(inflated) and direct.sum() in inflated),
                      np.log1p(values.min()), np.log1p(values.max()), np.log1p(anchor[key]),
                      np.log(values.max() / max(1, values.min())), np.sum(np.round(values/100)*100 == values))
    h = ids[query_idx]
    return np.column_stack([fit_counts[h], query_counts[h], extra[h]]), anchor[h]


def fitted_scores(immutable, ids, y, train, test):
    known_rows, known_anchor, known_position = [], [], []
    # Every learning row sees only the outcomes of other folds when constructing
    # its history. Baseline covariates and graph geometry contain no labels.
    for ref, query in StratifiedKFold(n_splits=5, shuffle=True, random_state=733).split(train, y[train]>0):
        query_idx = train[query]
        hist, anchor = history_features(train[ref], query_idx, ids, y)
        known_rows.append(np.column_stack([immutable[query_idx], hist]))
        known_anchor.append(anchor)
        known_position.append(query_idx)
    xfit = np.vstack(known_rows)
    tfit = y[np.concatenate(known_position)]
    base = np.concatenate(known_anchor)
    hist, anchor = history_features(train, test, ids, y)
    xquery = np.column_stack([immutable[test], hist])
    # Fit a deliberately small tree ensemble to regularize the rare event.
    settings = dict(n_estimators=150, num_leaves=9, learning_rate=.045,
                    min_child_samples=40, reg_lambda=8, n_jobs=2, verbosity=-1,
                    deterministic=True, force_col_wise=True, random_state=733)
    clf = LGBMClassifier(**settings).fit(xfit,tfit>0)
    prob = clf.predict_proba(xquery)[:,1]
    # The median of the entire distribution, including zeros, minimizes MAE.
    # Express each value relative to a profile's typical observed payout.
    warm_fit = xfit[:,-14] > 0
    warm_query = xquery[:,-14] > 0
    price = np.full(len(test), .01)
    reg = LGBMRegressor(objective="quantile",alpha=.5,**settings)
    reg.fit(xfit[warm_fit], np.log1p(tfit[warm_fit]/base[warm_fit]))
    price[warm_query] = np.maximum(.01, anchor[warm_query]*np.expm1(reg.predict(xquery[warm_query])))
    return prob,price


def score(y, probability, price, threshold):
    flagged = probability >= threshold
    pred = np.where(flagged, price, 0.)
    f1 = 2*np.sum(flagged & (y>0))/(np.sum(flagged)+np.sum(y>0))
    return dict(f1=float(f1), mae=float(np.mean(np.abs(pred-y))), flagged=int(flagged.sum()))


def run(args):
    sys.path.insert(0,str(args.repo))
    frame = pd.read_csv(args.repo/"data/trainingset.csv")
    y = frame.ClaimAmount.to_numpy()
    immutable, ids = geometry(frame[[f"feature{i}" for i in range(1,19)]])
    args.out.mkdir(parents=True,exist_ok=True)
    all_rows = []
    chosen = None
    for name, seeds in [("development",range(41,46)),("confirmation",range(151,156))]:
        for seed in seeds:
            train,test = train_test_split(np.arange(len(y)),stratify=y>0,test_size=.2,random_state=seed)
            p, price = fitted_scores(immutable, ids, y, train, test)
            for cut in ([.20,.24,.28,.32,.36] if chosen is None else [chosen]):
                row = dict(cohort=name, seed=seed, threshold=cut, **score(y[test],p,price,cut))
                all_rows.append(row)
                print(json.dumps(row),flush=True)
            pd.DataFrame(all_rows).to_csv(args.out/"alternative_scores.csv",index=False)
        if chosen is None:
            means = pd.DataFrame(all_rows).groupby("threshold")[["f1","mae"]].mean()
            chosen = float(means.sort_values(["f1","mae"],ascending=[False,True]).index[0])
            print("FROZEN THRESHOLD",chosen,flush=True)
    chosen_scores = pd.DataFrame(all_rows).query("threshold == @chosen")
    print(chosen_scores.groupby("cohort")[["f1","mae"]].mean().to_string(),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    with threadpool_limits(limits=2):
        run(parser.parse_args())
