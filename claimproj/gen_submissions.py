"""Generate the 10 checkpoint submissions 1_2_1.csv ... 1_2_10.csv.
threshold = claim probability needed before predicting a nonzero amount; scale = multiplier on the predicted amount."""
import numpy as np, pandas as pd
from common import (load_train, load_test, make_model, group_keys, fit_group_stats, add_group_feats,
                    fit_group_amounts, residual_dist, decide_group, ID_COL, TARGET)

X, y = load_train(); b = (y > 0).astype(int); pos = np.sort(y[y > 0])
ids, Xt = load_test()
keys, tkeys = group_keys(X), group_keys(Xt)
stats = fit_group_stats(keys, b)
Xg, Xtg = add_group_feats(X, keys, stats, own_b=b), add_group_feats(Xt, tkeys, stats)
logs = fit_group_amounts(keys, y); resid = residual_dist(logs)

p = {n: make_model(n).fit(Xg, b).predict_proba(Xtg)[:, 1] for n in ("lgbm", "cat")}
p["blend"] = (p["lgbm"] + p["cat"]) / 2
plan = {1: None,
        2: ("lgbm", 0.55, 1.0), 3: ("lgbm", 0.55, 0.9), 4: ("lgbm", 0.55, 0.8),
        5: ("lgbm", 0.50, 0.8), 6: ("lgbm", 0.50, 1.0), 7: ("lgbm", 0.60, 1.0),
        8: ("cat", 0.60, 0.8), 9: ("blend", 0.50, 0.8), 10: ("blend", 0.60, 0.8)}
for i, spec in plan.items():
    pred = np.zeros(len(Xt)) if spec is None else decide_group(p[spec[0]], tkeys, logs, resid, pos, spec[1], scale=spec[2])
    pd.DataFrame({ID_COL: ids, TARGET: pred}).to_csv(f"1_2_{i}.csv", index=False)
    print(f"1_2_{i}.csv  {spec}  nonzero={(pred > 0).sum()}")
