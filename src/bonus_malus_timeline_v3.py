"""Upgraded French Bonus-Malus reference-window cold detector with anniversary-split interactions."""

from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.bonus_malus_timeline import (
    RELAXED_DIFF_COLUMNS,
    STATIC_POLICY_COLUMNS,
    _find_backward_same_bm,
    _find_tier_jumps,
    _numeric_frame,
    _summarize_profiles,
)
from src.claim_probability import log_odds
from src.feature_schema import PROFILE_COLUMNS, RANDOM_SEED
from src.local_payout_tail import LocalPayoutTail

BM_V3_FEATURES = (
    "tier0", "tier1", "overlap_months", "log_overlap", "overlap_exp", "log_overlap_exp",
    "overlap_eff", "log_overlap_eff", "overlap_ge_2", "overlap_ge_5",
    "overlap_share", "exposure", "effective_exposure",
    "n_fit", "log_n_fit", "is_unseen", "is_fit1", "is_fit2", "is_fit3_plus",
    "n_query", "n_total", "fit_share", "n_total_odd",
    "delay", "abs_delay", "delta_licence", "bm_ratio", "exact_bm_rule", "exact_125", "exact_1125", "exact_two_claims",
    "bm_at_floor", "next_exposure", "log_next_portfolio",
    "prior_claim_explained", "prior_same_bm_exp", "fwd_claim_explained", "fwd_same_bm_exp",
    "next_fit_claimed", "jump_link_count",
    "exp_sum", "abs_exp_sum_m1", "exp_sum_near1",
    "ov_x_exp_sum_near1", "ov_x_unseen", "ov_x_fit1", "ov_x_exact_bm", "tier0_x_exp_sum_near1",
    "exp_sum_near1_x_unseen", "exp_sum_near1_x_fit1",
)


def _find_forward_same_bm(profiles):
    """Find immediately following same-BM terms that compete inside the reference window."""
    block_id = pd.util.hash_pandas_object(profiles[list(STATIC_POLICY_COLUMNS)], index=False)
    sizes = block_id.value_counts()
    subset = profiles[block_id.isin(sizes[(sizes >= 2) & (sizes <= 150)].index)].assign(_bid=block_id)
    if subset.empty:
        return pd.DataFrame(columns=["fwd_exp"]), {}
    pairs = subset.merge(subset, on="_bid", suffixes=("_t", "_f"))
    pairs = pairs[pairs["pid_t"] != pairs["pid_f"]]
    delta_lic = pairs["feature2_f"] - pairs["feature2_t"]
    valid = (
        delta_lic.between(0, 15)
        & (delta_lic - 12.0 * pairs["feature1_t"]).abs().le(2.5)
        & (pairs["feature8_f"] - pairs["feature8_t"]).isin([0, 1])
        & (pairs["feature3_f"] - pairs["feature3_t"]).isin([0, 1])
        & (pairs["feature10_f"] == pairs["feature10_t"])
    )
    fwd = pairs.loc[valid, ["key_t", "key_f", "feature1_f"]]
    if fwd.empty:
        return pd.DataFrame(columns=["fwd_exp"]), {}
    return fwd.groupby("key_t").agg(fwd_exp=("feature1_f", "sum")), fwd.groupby("key_t")["key_f"].apply(list).to_dict()


def build_timeline_graph_v3(profiles):
    """Build purely unsupervised Tier 0/1/2 forward BM-jump graph with forward/backward same-BM context."""
    frames = [_find_tier_jumps(profiles, STATIC_POLICY_COLUMNS, tier=0, max_group=150)]
    no_riskvar = tuple(c for c in STATIC_POLICY_COLUMNS if c != "feature17")
    frames.append(_find_tier_jumps(profiles, no_riskvar, tier=1, max_group=80, diff_col="feature17"))
    for diff_col in RELAXED_DIFF_COLUMNS:
        cols = tuple(c for c in STATIC_POLICY_COLUMNS if c != diff_col)
        frames.append(_find_tier_jumps(profiles, cols, tier=2, max_group=80, diff_col=diff_col))
    frames = [frame for frame in frames if not frame.empty]
    base = profiles.set_index("key")[["n_portfolio", "feature1", "feature10"]]
    if not frames:
        return base.assign(tier=np.nan), {}, {}
    jumps = pd.concat(frames, ignore_index=True)
    jumps["sort_key"] = jumps["tier"] * 100.0 - jumps["overlap_months"] + 0.01 * jumps["delay"].abs()
    best = jumps.sort_values("sort_key").drop_duplicates("key_t").set_index("key_t")
    link_counts = jumps.groupby("key_t")["key_n"].nunique().rename("jump_link_count")
    bwd_stats, bwd_map = _find_backward_same_bm(profiles)
    fwd_stats, fwd_map = _find_forward_same_bm(profiles)
    keep = ["key_n", "tier", "bm_ratio", "delay", "delta_licence", "overlap_months",
            "exact_bm_rule", "exact_two_claims", "feature1_n", "n_portfolio_n"]
    return base.join(best[keep]).join(link_counts).join(bwd_stats).join(fwd_stats), bwd_map, fwd_map


def _bm_v3_row(key, n_fit, n_query, graph, bwd_map, fwd_map, fit_positives):
    row = graph.loc[key]
    tier, overlap = float(row["tier"]), float(row["overlap_months"])
    exposure, next_exp = float(row["feature1"]), float(row["feature1_n"])
    exp_sum = exposure + next_exp
    abs_exp_m1 = abs(exp_sum - 1.0)
    exp_near1 = float(abs_exp_m1 <= 0.025)
    overlap_exp = max(0.0, 12.0 * exposure - 2.0)
    # Complementary 2004 anniversary splits terminate at L_jump, avoiding integer LicAge rounding jitter.
    overlap_eff = overlap_exp if exp_near1 > 0.5 else overlap
    log_ov = float(np.log1p(overlap))
    log_ov_exp = float(np.log1p(overlap_exp))
    log_ov_eff = float(np.log1p(overlap_eff))
    delay = float(row["delay"])
    bwd_exp = 0.0 if pd.isna(row["bwd_exp"]) else float(row["bwd_exp"])
    fwd_exp = 0.0 if pd.isna(row["fwd_exp"]) else float(row["fwd_exp"])
    prior_pos = float(sum(fit_positives.get(prev, 0) for prev in bwd_map.get(key, ())) > 0)
    fwd_pos = float(sum(fit_positives.get(nxt, 0) for nxt in fwd_map.get(key, ())) > 0)
    next_pos = float(fit_positives.get(row["key_n"], 0) > 0)
    bt = int(row["feature10"])
    bn = int(round(bt * float(row["bm_ratio"])))
    ex125 = float((bn >= (bt * 125 // 100) - 1) and (bn <= (bt * 125 // 100) + 1) and (bn > bt))
    ex1125 = float((bn >= (bt * 1125 // 1000) - 1) and (bn <= (bt * 1125 // 1000) + 1) and (bn > bt))
    ex_bm, n_total = float(row["exact_bm_rule"]), n_fit + n_query
    is_unseen, is_fit1, tier0 = float(n_fit == 0), float(n_fit == 1), float(tier == 0.0)
    return (
        tier0, float(tier == 1.0), overlap, log_ov, overlap_exp, log_ov_exp,
        overlap_eff, log_ov_eff, float(overlap_eff >= 2.0), float(overlap_eff >= 5.0),
        overlap_eff / max(overlap_eff + 12.0 * (bwd_exp + fwd_exp), 1.0),
        exposure, max(0.0, exposure - 0.167),
        n_fit, float(np.log1p(n_fit)), is_unseen, is_fit1, float(n_fit == 2), float(n_fit >= 3),
        n_query, n_total, n_fit / max(n_total, 1.0), float(int(n_total) % 2),
        delay, abs(delay), float(row["delta_licence"]), float(row["bm_ratio"]),
        ex_bm, ex125, ex1125, float(row["exact_two_claims"]),
        float(bt == 50), next_exp, float(np.log1p(float(row["n_portfolio_n"]))),
        prior_pos, bwd_exp, fwd_pos, fwd_exp, next_pos, float(row["jump_link_count"]),
        exp_sum, abs_exp_m1, exp_near1,
        log_ov_eff * exp_near1, log_ov_eff * is_unseen, log_ov_eff * is_fit1, log_ov_eff * ex_bm, tier0 * exp_near1,
        exp_near1 * is_unseen, exp_near1 * is_fit1,
    )


def _fit_analytical_v3(fit_keys, fit_y, graph, bwd_map, fwd_map, alpha, regularization_c=0.20):
    counts = Counter(fit_keys)
    positives = Counter(k for k, yv in zip(fit_keys, fit_y) if yv)
    jump_keys = [k for k in counts if k in graph.index and pd.notna(graph.at[k, "tier"])]
    if not jump_keys:
        return None
    ret_grid = (max(0.50, alpha - 0.10), alpha, min(0.92, alpha + 0.05))
    records, targets, weights = [], [], []
    for key in jump_keys:
        n_tot, k_pos = counts[key], positives[key]
        n_zero = n_tot - k_pos
        for ret in ret_grid:
            p_hidden = (1.0 - ret) ** k_pos
            for z_seen in range(n_zero + 1):
                unseen = (n_zero - z_seen) + k_pos
                if unseen == 0:
                    continue
                weight = binom.pmf(z_seen, n_zero, ret) * p_hidden * unseen / len(ret_grid)
                if weight < 1e-8:
                    continue
                records.append(_bm_v3_row(key, float(z_seen), float(unseen), graph, bwd_map, fwd_map, positives))
                targets.append(k_pos / unseen)
                weights.append(weight)
    if not records:
        return None
    design = pd.DataFrame(records, columns=BM_V3_FEATURES)
    y_arr, w_arr = np.asarray(targets, dtype=float), np.asarray(weights, dtype=float)
    x_bin = pd.concat([design, design], ignore_index=True)
    y_bin = np.r_[np.ones(len(design)), np.zeros(len(design))]
    w_bin = np.r_[w_arr * y_arr, w_arr * (1.0 - y_arr)]
    keep = w_bin > 1e-9
    if np.unique(y_bin[keep]).size < 2:
        return None
    return make_pipeline(StandardScaler(), LogisticRegression(C=regularization_c, max_iter=500)).fit(
        x_bin.loc[keep], y_bin[keep], logisticregression__sample_weight=w_bin[keep]
    )


class BonusMalusTimelineV3Detector:
    """Calibrated cold-profile detector using French Bonus-Malus anniversary transitions."""

    def __init__(self, random_seed=RANDOM_SEED, alpha=0.80, regularization_c=0.20,
                 high_confidence_quote=0.50, low_confidence_floor=1.0):
        self.random_seed = random_seed
        self.alpha = alpha
        self.regularization_c = regularization_c
        self.high_confidence_quote = high_confidence_quote
        self.low_confidence_floor = low_confidence_floor

    def fit(self, features, amounts):
        numeric = _numeric_frame(features)
        keys = pd.util.hash_pandas_object(numeric, index=False).to_numpy()
        labels = (np.asarray(amounts, dtype=float) > 0.0).astype(int)
        self.fit_counts_ = Counter(keys)
        self.fit_positives_ = Counter(k for k, yv in zip(keys, labels) if yv)
        fit_profiles = _summarize_profiles(numeric, keys)
        self.reference_profiles_ = fit_profiles.drop(columns=["pid"])
        self.tail_ = LocalPayoutTail().fit(features, amounts)
        graph, bwd_map, fwd_map = build_timeline_graph_v3(fit_profiles)
        self.model_ = _fit_analytical_v3(keys, labels, graph, bwd_map, fwd_map, self.alpha, self.regularization_c)
        self.calibrator_ = self._fit_calibrator(features, keys, labels, graph, bwd_map, fwd_map)
        return self

    def _fit_calibrator(self, features, keys, labels, graph, bwd_map, fwd_map):
        if self.model_ is None or np.unique(labels).size < 2:
            return None
        oof_scores, oof_targets = [], []
        for rep in range(2):
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed * 10 + rep)
            for tr_idx, va_idx in splitter.split(features, labels):
                tk, ty, vk, vy = keys[tr_idx], labels[tr_idx], keys[va_idx], labels[va_idx]
                fold_model = _fit_analytical_v3(tk, ty, graph, bwd_map, fwd_map, self.alpha, self.regularization_c)
                if fold_model is None:
                    continue
                tc, tp, vc = Counter(tk), Counter(k for k, yv in zip(tk, ty) if yv), Counter(vk)
                cand = np.array([(tp[k] == 0) and (k in graph.index) and pd.notna(graph.at[k, "tier"]) for k in vk])
                if not cand.any():
                    continue
                idxs = np.flatnonzero(cand)
                x_val = pd.DataFrame(
                    [_bm_v3_row(vk[i], float(tc[vk[i]]), float(vc[vk[i]]), graph, bwd_map, fwd_map, tp) for i in idxs],
                    columns=BM_V3_FEATURES,
                )
                oof_scores.append(fold_model.predict_proba(x_val)[:, 1])
                oof_targets.append(vy[cand])
        if not oof_scores:
            return None
        scores = np.clip(np.concatenate(oof_scores), 1e-4, 1.0 - 1e-4)
        targets = np.concatenate(oof_targets)
        if np.unique(targets).size < 2:
            return None
        return LogisticRegression(C=10.0).fit(log_odds(scores)[:, None], targets)

    def predict_cold_risk(self, query_features):
        n_rows = len(query_features)
        if n_rows == 0 or self.model_ is None:
            return np.zeros(n_rows, dtype=bool), np.zeros(n_rows, dtype=float)
        query_num = _numeric_frame(query_features)
        query_keys = pd.util.hash_pandas_object(query_num, index=False).to_numpy()
        query_counts = Counter(query_keys)
        query_profiles = _summarize_profiles(query_num, query_keys).drop(columns=["pid"])
        combined = pd.concat([self.reference_profiles_, query_profiles], ignore_index=True)
        merged = combined.groupby(list(PROFILE_COLUMNS), as_index=False).agg(
            n_portfolio=("n_portfolio", "sum"), key=("key", "first"),
        )
        merged["pid"] = np.arange(len(merged))
        graph, bwd_map, fwd_map = build_timeline_graph_v3(merged)
        cand = np.array(
            [(self.fit_positives_[k] == 0) and (k in graph.index) and pd.notna(graph.at[k, "tier"]) for k in query_keys],
            dtype=bool,
        )
        probabilities = np.zeros(n_rows, dtype=float)
        if not cand.any():
            return cand, probabilities
        idxs = np.flatnonzero(cand)
        design = pd.DataFrame(
            [_bm_v3_row(query_keys[i], float(self.fit_counts_[query_keys[i]]),
                        float(query_counts[query_keys[i]]), graph, bwd_map, fwd_map, self.fit_positives_)
             for i in idxs],
            columns=BM_V3_FEATURES,
        )
        raw = np.clip(self.model_.predict_proba(design)[:, 1], 1e-4, 1.0 - 1e-4)
        probabilities[cand] = self.calibrator_.predict_proba(log_odds(raw)[:, None])[:, 1] if self.calibrator_ is not None else raw
        return cand, probabilities

    def quote_cold_claims(self, query_features, cold_risk):
        tail_quotes = self.tail_.predict(query_features)
        return np.where(cold_risk >= self.high_confidence_quote, tail_quotes, self.low_confidence_floor)
