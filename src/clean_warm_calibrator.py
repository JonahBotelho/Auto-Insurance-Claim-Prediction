"""Non-memorizing warm probability calibrator (Stratified EM + LGBM + CatBoost) for L1 pricing."""

from collections import Counter, defaultdict

from catboost import CatBoostClassifier
from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.claim_probability import log_odds
from src.dashboard_model import profile_keys
from src.exact_em_deconvolution import (
    ExactEMDeconvolution,
    canonical_state,
    profile_stratum,
    source_supports,
)
from src.feature_schema import RANDOM_SEED
from src.masked_payout_history import aggregate_history, masked_states
from src.profile_count_survival import COUNT, possible_scoring_counts


class CleanWarmCalibrator:
    """Calibrate warm claim probabilities without high-cardinality profile memorization."""

    def __init__(self, alpha=0.80, random_seed=RANDOM_SEED):
        self.alpha = alpha
        self.random_seed = random_seed

    def _feature_row(self, seen_zero, seen_paid, total_rows, exposure, veh_age, bonus_malus,
                     supports, em_global, em_strats, own_full_paid=None):
        whole_sup, scaled_sup, sum_sup, split_sup = supports
        cent = {b: n for (b, w), n in seen_paid.items() if w}
        scaled = {b: n for (b, w), n in seen_paid.items() if not w}
        sum_w, top_s = sum(cent.keys()), max(scaled.keys(), default=0)
        anchor = top_s or max(cent.keys(), default=0)
        nw, ns = sum(cent.values()), sum(scaled.values())
        nq = total_rows - seen_zero - nw - ns
        reconciled = float(len(scaled) == 1 and top_s == sum_w)
        gap = max(0, top_s - sum_w)
        rem_copies_2 = sum(max(0, 2 - n) for n in seen_paid.values())

        if own_full_paid is not None:
            own_c = {b: n for (b, w), n in own_full_paid.items() if w}
            own_s = {b: n for (b, w), n in own_full_paid.items() if not w}
            ow = max(0, whole_sup[anchor] - int(anchor in own_c))
            osc = max(0, scaled_sup[anchor] - int(anchor in own_s))
            own_sum = anchor == sum(own_c) and anchor in own_s
            oconf = max(0, sum_sup[anchor] - int(own_sum))
            ospl = max(0, split_sup[anchor] - int(own_sum and len(own_c) > 1))
            ogap = max(0, whole_sup[gap] - int(gap in own_c)) if gap > 0 else 0
        else:
            ow, osc = whole_sup[anchor], scaled_sup[anchor]
            oconf, ospl = sum_sup[anchor], split_sup[anchor]
            ogap = whole_sup[gap] if gap > 0 else 0

        state = canonical_state(seen_zero, seen_paid)
        pg, dg = em_global.predict_state(state, nq)
        pg_sm = (dg * pg + 5e-4 * 0.45) / (dg + 5e-4)
        stratum = profile_stratum(seen_paid, whole_sup, scaled_sup, split_sup)
        ps, ds = em_strats[stratum].predict_state(state, nq, fallback=pg_sm) if stratum in em_strats else (pg_sm, 0.0)
        ps_sm = (ds * ps + 5e-4 * pg_sm) / (ds + 5e-4)
        euros = anchor / 100.0
        return (
            seen_zero, nw, ns, len(cent), len(scaled),
            sum(n >= 2 for n in cent.values()), sum(n >= 2 for n in scaled.values()),
            float(ns == 0), float(nw == 0), reconciled, float(top_s > sum_w > 0),
            float(reconciled > 0 and rem_copies_2 == 0), rem_copies_2,
            min(1.0, rem_copies_2 / max(nq, 1)) if reconciled > 0 else 1.0,
            total_rows, nq, (nw + ns) / max(0.5, seen_zero + nw + ns),
            np.log1p(euros), float(euros < 250.0), float(euros > 1500.0), float(anchor % 100 == 0),
            np.log1p(gap / 100.0), gap / max(top_s, 1.0),
            np.log1p(ow), np.log1p(osc), np.log1p(oconf), np.log1p(ospl), np.log1p(ogap),
            np.log((1.0 + ow) / (1.0 + osc)), osc / max(1.0, ow + osc),
            pg_sm, ps_sm, log_odds(np.clip(pg_sm, 0.01, 0.99)), log_odds(np.clip(ps_sm, 0.01, 0.99)),
            float(exposure), float(exposure >= 0.999), float(veh_age),
            float(bonus_malus == 50), min(float(bonus_malus), 150.0),
        )

    def _fit_em_pair(self, keys, supports, global_iter, strat_iter):
        global_obs, strat_obs = Counter(), defaultdict(Counter)
        for key in keys:
            state = canonical_state(self.zeros_[key], self.paid_[key])
            global_obs[state] += 1
            strat_obs[profile_stratum(self.paid_[key], supports[0], supports[1], supports[3])][state] += 1
        em_g = ExactEMDeconvolution(alpha=self.alpha, n_iter=global_iter).fit_counts(global_obs)
        em_s = {
            name: ExactEMDeconvolution(alpha=self.alpha, n_iter=strat_iter).fit_counts(
                cnt, states=em_g.states_, prior_pi=em_g.pi_, prior_strength=120.0
            )
            for name, cnt in strat_obs.items()
        }
        return em_g, em_s

    def fit(self, features, amounts):
        keys, self.zeros_, self.paid_ = aggregate_history(features, amounts)
        self.supports_ = source_supports(self.paid_)
        warm_keys = np.array(list(self.paid_.keys()))
        if len(warm_keys) < 6:
            self.lgb_, self.cb_, self.em_g_, self.em_s_ = None, None, None, {}
            return self

        first_pos = {keys[i]: i for i in np.flatnonzero(~pd.Index(keys).duplicated()) if keys[i] in self.paid_}
        # Cross-fit EM priors by profile so masked-state trees do not overfit EM on training profiles.
        fold_em = {}
        for tr_idx, va_idx in GroupKFold(n_splits=3).split(warm_keys, groups=warm_keys):
            tr_keys = warm_keys[tr_idx]
            tr_supports = source_supports({k: self.paid_[k] for k in tr_keys})
            em_pair = self._fit_em_pair(tr_keys, tr_supports, global_iter=70, strat_iter=45)
            for key in warm_keys[va_idx]:
                fold_em[key] = em_pair
        self.em_g_, self.em_s_ = self._fit_em_pair(warm_keys, self.supports_, global_iter=80, strat_iter=50)

        ret_grid = (max(0.65, self.alpha - 0.05), min(0.92, self.alpha + 0.07))
        exp_arr = features["feature1"].to_numpy(dtype=float)
        veh_arr = features["feature3"].to_numpy(dtype=float)
        bm_arr = features["feature10"].to_numpy(dtype=float)

        x_rows, y_rows, w_rows = [], [], []
        for key in warm_keys:
            pos, (em_g, em_s) = first_pos[key], fold_em[key]
            z0, p0, agg = self.zeros_[key], self.paid_[key], {}
            for ret in ret_grid:
                for sz, sp, neg, pos_m in masked_states(z0, p0, ret):
                    if not sp or neg + pos_m < 1e-8:
                        continue
                    paid_left = sum(p0.values()) - sum(sp.values())
                    sizes, sel, outcomes = possible_scoring_counts(z0 - sz + paid_left, paid_left)
                    for size, vol, law in zip(sizes, sel, outcomes):
                        if vol <= 1e-8:
                            continue
                        state_key = (sz, tuple(sorted(sp.items())), sz + sum(sp.values()) + int(size))
                        mass = (neg + pos_m) * vol / len(ret_grid)
                        rate = float(np.dot(law, COUNT)) / float(size)
                        prev = agg.get(state_key, (0.0, 0.0))
                        agg[state_key] = (prev[0] + mass, prev[1] + mass * rate)
            for (sz, sp_items, tot), (m_sum, mr_sum) in agg.items():
                if m_sum >= 1e-8:
                    x_rows.append(self._feature_row(
                        sz, dict(sp_items), tot, exp_arr[pos], veh_arr[pos], bm_arr[pos],
                        self.supports_, em_g, em_s, own_full_paid=p0,
                    ))
                    y_rows.append(mr_sum / m_sum)
                    w_rows.append(m_sum)

        if not x_rows:
            self.lgb_, self.cb_ = None, None
            return self
        X = np.asarray(x_rows, dtype=float)
        y = np.asarray(y_rows, dtype=float)
        w = np.asarray(w_rows, dtype=float)
        self.lgb_ = LGBMRegressor(
            objective="cross_entropy", n_estimators=220, num_leaves=10, min_child_samples=90,
            learning_rate=0.03, reg_lambda=50.0, subsample=0.8, colsample_bytree=0.8,
            random_state=self.random_seed, n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True,
        ).fit(X, y, sample_weight=w)
        x_bin = np.vstack([X, X])
        y_bin = np.r_[np.ones(len(X), dtype=int), np.zeros(len(X), dtype=int)]
        w_bin = np.r_[w * y, w * (1.0 - y)]
        keep = w_bin > 1e-8
        self.cb_ = CatBoostClassifier(
            loss_function="Logloss", iterations=350, depth=4, learning_rate=0.035,
            l2_leaf_reg=35.0, random_seed=self.random_seed, thread_count=2,
            allow_writing_files=False, verbose=False,
        ).fit(x_bin[keep], y_bin[keep], sample_weight=w_bin[keep])
        return self

    def predict(self, query):
        n_rows = len(query)
        probabilities = np.zeros(n_rows, dtype=float)
        if n_rows == 0 or self.lgb_ is None or self.cb_ is None:
            return probabilities
        keys = profile_keys(query)
        query_counts = Counter(keys)
        exp_arr = query["feature1"].to_numpy(dtype=float)
        veh_arr = query["feature3"].to_numpy(dtype=float)
        bm_arr = query["feature10"].to_numpy(dtype=float)

        warm_idx, feats = [], []
        for idx, key in enumerate(keys):
            paid = self.paid_.get(key)
            if paid is None:
                continue
            tot = self.zeros_[key] + sum(paid.values()) + query_counts[key]
            warm_idx.append(idx)
            feats.append(self._feature_row(
                self.zeros_[key], paid, tot, exp_arr[idx], veh_arr[idx], bm_arr[idx],
                self.supports_, self.em_g_, self.em_s_,
            ))
        if warm_idx:
            Xq = np.asarray(feats, dtype=float)
            p_lgb = np.clip(self.lgb_.predict(Xq), 1e-4, 1.0 - 1e-4)
            p_cb = np.clip(self.cb_.predict_proba(Xq)[:, 1], 1e-4, 1.0 - 1e-4)
            probabilities[warm_idx] = 0.5 * p_lgb + 0.5 * p_cb
        return probabilities
