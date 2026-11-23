"""Detect cold-profile claims from French Bonus-Malus reference-window transitions."""

from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.claim_probability import log_odds
from src.feature_schema import PROFILE_COLUMNS, RANDOM_SEED
from src.local_payout_tail import LocalPayoutTail

# Time-varying policy fields excluded when grouping records of the same policyholder/vehicle.
TIME_VARYING_COLUMNS = frozenset({"feature1", "feature2", "feature3", "feature8", "feature10"})
STATIC_POLICY_COLUMNS = tuple(c for c in PROFILE_COLUMNS if c not in TIME_VARYING_COLUMNS)
RELAXED_DIFF_COLUMNS = ("feature5", "feature7", "feature9", "feature18")

TIMELINE_FEATURE_COLUMNS = (
    "tier0", "tier1", "overlap_months", "log_overlap", "overlap_ge_2", "overlap_ge_5",
    "overlap_share", "exposure", "effective_exposure",
    "n_fit", "log_n_fit", "is_unseen", "is_fit1", "is_fit2", "is_fit3_plus",
    "n_query", "n_total", "fit_share", "n_total_odd",
    "delay", "abs_delay", "delta_licence", "bm_ratio", "exact_bm_rule", "exact_two_claims",
    "bm_at_floor", "next_exposure", "log_next_portfolio", "prior_claim_explained", "prior_same_bm_exp",
    "jump_link_count",
)


def _numeric_frame(features):
    """Return the 18 profile columns coerced to consistent float64/int64 numeric types."""
    numeric = pd.DataFrame(index=features.index)
    for col in PROFILE_COLUMNS:
        series = pd.to_numeric(features[col])
        numeric[col] = series.astype(np.float64 if col == "feature1" else np.int64)
    return numeric


def canonical_profile_keys(features):
    """Hash the 18 profile columns using canonical numeric dtypes (immune to category dtype drift)."""
    return pd.util.hash_pandas_object(_numeric_frame(features), index=False).to_numpy()


def _summarize_profiles(numeric_df, keys, weights=None):
    """Aggregate unique profiles with portfolio row counts and canonical keys."""
    frame = numeric_df.copy()
    frame["_key"] = keys
    frame["_w"] = 1 if weights is None else weights
    profiles = frame.groupby(list(PROFILE_COLUMNS), as_index=False).agg(
        n_portfolio=("_w", "sum"),
        key=("_key", "first"),
    )
    profiles["pid"] = np.arange(len(profiles))
    return profiles


def _find_tier_jumps(profiles, block_cols, tier, max_group=120, diff_col=None):
    """Find forward Bonus-Malus jumps and compute French [L_jump-14, L_jump-2] window overlap."""
    block_id = pd.util.hash_pandas_object(profiles[list(block_cols)], index=False)
    sizes = block_id.value_counts()
    subset = profiles[block_id.isin(sizes[(sizes >= 2) & (sizes <= max_group)].index)].assign(_bid=block_id)
    if subset.empty:
        return pd.DataFrame()
    pairs = subset.merge(subset, on="_bid", suffixes=("_t", "_n"))
    pairs = pairs[pairs["pid_t"] != pairs["pid_n"]].copy()
    if diff_col is not None:
        pairs = pairs[pairs[f"{diff_col}_t"] != pairs[f"{diff_col}_n"]]
    if pairs.empty:
        return pd.DataFrame()

    delta_lic = pairs["feature2_n"] - pairs["feature2_t"]
    term_months = 12.0 * pairs["feature1_t"]
    bm_t, bm_n = pairs["feature10_t"], pairs["feature10_n"]
    bm_ratio = bm_n / bm_t
    # In French motor insurance, an anniversary BM increase at L_jump is triggered by
    # at-fault claims occurring in the 12-month window ending 2 months prior: [L_jump-14, L_jump-2].
    overlap = np.maximum(0.0, np.minimum(term_months, delta_lic - 2.0) - np.maximum(0.0, delta_lic - 14.0))
    delay = delta_lic - term_months
    exact_125 = ((bm_n >= (bm_t * 125 // 100) - 1) & (bm_n <= (bm_t * 125 // 100) + 1) & (bm_n > bm_t)).astype(float)
    exact_1125 = ((bm_n >= (bm_t * 1125 // 1000) - 1) & (bm_n <= (bm_t * 1125 // 1000) + 1) & (bm_n > bm_t)).astype(float)
    exact_156 = ((bm_n >= ((bm_t * 125 // 100) * 125 // 100) - 2) & (bm_n <= ((bm_t * 125 // 100) * 125 // 100) + 2) & (bm_n > bm_t)).astype(float)
    exact_rule = np.maximum(exact_125, np.maximum(exact_1125, exact_156))

    valid = (
        (delta_lic >= 1) & (delta_lic <= 18) &
        delay.between(-2.0, 8.0 if tier == 0 else 2.0) &
        (pairs["feature8_n"] - pairs["feature8_t"]).isin([0, 1]) &
        (pairs["feature3_n"] - pairs["feature3_t"]).isin([0, 1]) &
        (bm_n > bm_t) & bm_ratio.between(1.08, 1.75)
    )
    if tier == 1:
        valid &= (pairs["feature17_n"] > pairs["feature17_t"]) & (delta_lic >= 2)
    elif tier == 2:
        valid &= (exact_rule > 0) & (delta_lic >= 2) & (overlap >= 2.0)
    elif tier == 4:
        valid &= (
            (pairs["feature17_n"] < pairs["feature17_t"]) & (exact_rule > 0)
            & (delta_lic >= 2) & (overlap >= 2.0)
            & (pairs["feature1_t"] + pairs["feature1_n"] - 1.).abs().le(.025)
        )
    if not valid.any():
        return pd.DataFrame()

    selected = pairs.loc[valid, ["key_t", "key_n", "feature1_n", "n_portfolio_n"]].copy()
    selected["tier"] = float(tier)
    selected["bm_ratio"] = bm_ratio.loc[valid].to_numpy()
    selected["delay"] = delay.loc[valid].to_numpy()
    selected["delta_licence"] = delta_lic.loc[valid].to_numpy()
    selected["overlap_months"] = overlap.loc[valid].to_numpy()
    selected["exact_bm_rule"] = exact_rule.loc[valid].to_numpy()
    selected["exact_two_claims"] = exact_156.loc[valid].to_numpy()
    return selected


def _find_backward_same_bm(profiles):
    """Find immediately preceding terms with the same BM that compete to explain a jump."""
    block_id = pd.util.hash_pandas_object(profiles[list(STATIC_POLICY_COLUMNS)], index=False)
    sizes = block_id.value_counts()
    subset = profiles[block_id.isin(sizes[(sizes >= 2) & (sizes <= 150)].index)].assign(_bid=block_id)
    if subset.empty:
        return pd.DataFrame(columns=["bwd_exp"]), {}
    pairs = subset.merge(subset, on="_bid", suffixes=("_t", "_p"))
    pairs = pairs[pairs["pid_t"] != pairs["pid_p"]]
    delta_lic = pairs["feature2_t"] - pairs["feature2_p"]
    valid = (
        delta_lic.between(0, 15) &
        (delta_lic - 12.0 * pairs["feature1_p"]).abs().le(2.5) &
        (pairs["feature8_t"] - pairs["feature8_p"]).isin([0, 1]) &
        (pairs["feature3_t"] - pairs["feature3_p"]).isin([0, 1]) &
        (pairs["feature10_p"] == pairs["feature10_t"])
    )
    bwd = pairs.loc[valid, ["key_t", "key_p", "feature1_p"]]
    if bwd.empty:
        return pd.DataFrame(columns=["bwd_exp"]), {}
    return bwd.groupby("key_t").agg(bwd_exp=("feature1_p", "sum")), bwd.groupby("key_t")["key_p"].apply(list).to_dict()


def build_timeline_graph(profiles):
    """Build purely unsupervised French Bonus-Malus jump table from summarized profiles."""
    frames = [_find_tier_jumps(profiles, STATIC_POLICY_COLUMNS, tier=0, max_group=150)]
    no_riskvar = tuple(c for c in STATIC_POLICY_COLUMNS if c != "feature17")
    frames.append(_find_tier_jumps(profiles, no_riskvar, tier=1, max_group=80, diff_col="feature17"))
    for diff_col in RELAXED_DIFF_COLUMNS:
        cols = tuple(c for c in STATIC_POLICY_COLUMNS if c != diff_col)
        frames.append(_find_tier_jumps(profiles, cols, tier=2, max_group=80, diff_col=diff_col))
    frames = [frame for frame in frames if not frame.empty]

    base = profiles.set_index("key")[["n_portfolio", "feature1", "feature10"]]
    if not frames:
        return base.assign(tier=np.nan), {}
    jumps = pd.concat(frames, ignore_index=True)
    jumps["sort_key"] = jumps["tier"] * 100.0 - jumps["overlap_months"] + 0.01 * jumps["delay"].abs()
    best = jumps.sort_values("sort_key").drop_duplicates("key_t").set_index("key_t")
    link_counts = jumps.groupby("key_t")["key_n"].nunique().rename("jump_link_count")
    bwd_stats, bwd_map = _find_backward_same_bm(profiles)
    keep = ["tier", "bm_ratio", "delay", "delta_licence", "overlap_months",
            "exact_bm_rule", "exact_two_claims", "feature1_n", "n_portfolio_n"]
    return base.join(best[keep]).join(link_counts).join(bwd_stats), bwd_map


def _row_record(key, n_fit, n_query, graph, bwd_map, fit_positives):
    """Assemble numeric model features for one cold jump candidate profile."""
    row = graph.loc[key]
    tier = float(row["tier"])
    overlap = float(row["overlap_months"])
    delay = float(row["delay"])
    bwd_exp = 0.0 if pd.isna(row["bwd_exp"]) else float(row["bwd_exp"])
    prior_pos = float(sum(fit_positives.get(prev, 0) for prev in bwd_map.get(key, ())) > 0)
    exposure = float(row["feature1"])
    n_total = n_fit + n_query
    return (
        float(tier == 0.0), float(tier == 1.0), overlap, np.log1p(overlap),
        float(overlap >= 2.0), float(overlap >= 5.0), overlap / max(overlap + 12.0 * bwd_exp, 1.0),
        exposure, max(0.0, exposure - 0.167),
        n_fit, np.log1p(n_fit), float(n_fit == 0), float(n_fit == 1), float(n_fit == 2), float(n_fit >= 3),
        n_query, n_total, n_fit / max(n_total, 1.0), float(int(n_total) % 2),
        delay, abs(delay), float(row["delta_licence"]), float(row["bm_ratio"]),
        float(row["exact_bm_rule"]), float(row["exact_two_claims"]), float(row["feature10"] == 50),
        float(row["feature1_n"]), np.log1p(float(row["n_portfolio_n"])), prior_pos, bwd_exp,
        float(row["jump_link_count"]),
    )


def _fit_analytical_lr(fit_keys, fit_y, graph, bwd_map, alpha):
    """Fit regularized logistic regression via exact analytical Binomial thinning of jump profiles."""
    counts = Counter(fit_keys)
    positives = Counter(k for k, yv in zip(fit_keys, fit_y) if yv)
    jump_keys = [k for k in counts if k in graph.index and pd.notna(graph.at[k, "tier"])]
    if not jump_keys:
        return None
    ret_grid = (max(0.55, alpha - 0.10), alpha, min(0.92, alpha + 0.05))
    records, targets, weights = [], [], []
    for key in jump_keys:
        n_tot, k_pos = counts[key], positives[key]
        n_zero = n_tot - k_pos
        for ret in ret_grid:
            prob_all_pos_hidden = (1.0 - ret) ** k_pos
            for z_seen in range(n_zero + 1):
                unseen = (n_zero - z_seen) + k_pos
                if unseen == 0:
                    continue
                weight = binom.pmf(z_seen, n_zero, ret) * prob_all_pos_hidden * unseen / len(ret_grid)
                if weight < 1e-8:
                    continue
                records.append(_row_record(key, float(z_seen), float(unseen), graph, bwd_map, positives))
                targets.append(k_pos / unseen)
                weights.append(weight)
    if not records:
        return None
    design = pd.DataFrame(records, columns=TIMELINE_FEATURE_COLUMNS)
    y_arr = np.asarray(targets, dtype=float)
    w_arr = np.asarray(weights, dtype=float)
    # Split fractional cross-entropy targets into weighted binary rows for exact logistic fitting.
    x_bin = pd.concat([design, design], ignore_index=True)
    y_bin = np.r_[np.ones(len(design)), np.zeros(len(design))]
    w_bin = np.r_[w_arr * y_arr, w_arr * (1.0 - y_arr)]
    keep = w_bin > 1e-9
    if np.unique(y_bin[keep]).size < 2:
        return None
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.20, max_iter=500))
    model.fit(x_bin.loc[keep], y_bin[keep], logisticregression__sample_weight=w_bin[keep])
    return model


class BonusMalusTimelineDetector:
    """Calibrated cold-profile detector using French Bonus-Malus transition windows."""

    def __init__(self, random_seed=RANDOM_SEED, alpha=0.80, high_confidence_quote=0.50, low_confidence_floor=1.0):
        self.random_seed = random_seed
        self.alpha = alpha
        self.high_confidence_quote = high_confidence_quote
        self.low_confidence_floor = low_confidence_floor

    def fit(self, features, amounts):
        numeric = _numeric_frame(features)
        keys = pd.util.hash_pandas_object(numeric, index=False).to_numpy()
        labels = (np.asarray(amounts, dtype=float) > 0.0).astype(int)
        self.fit_counts_ = Counter(keys)
        self.fit_positives_ = Counter(k for k, yv in zip(keys, labels) if yv)
        # Store only unique numeric profiles with their fitting counts for compact serialization.
        fit_profiles = _summarize_profiles(numeric, keys)
        self.reference_profiles_ = fit_profiles.drop(columns=["pid"])
        self.tail_ = LocalPayoutTail().fit(features, amounts)

        graph, bwd_map = build_timeline_graph(fit_profiles)
        self.model_ = _fit_analytical_lr(keys, labels, graph, bwd_map, self.alpha)
        self.calibrator_ = self._fit_platt_calibrator(features, keys, labels, graph, bwd_map)
        return self

    def _fit_platt_calibrator(self, features, keys, labels, graph, bwd_map):
        """Correct analytical double-thinning bias via out-of-fold Platt scaling inside fitting."""
        if self.model_ is None or np.unique(labels).size < 2:
            return None
        oof_scores, oof_targets = [], []
        for rep in range(2):
            splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed * 10 + rep)
            for tr_idx, va_idx in splitter.split(features, labels):
                tk, ty = keys[tr_idx], labels[tr_idx]
                vk, vy = keys[va_idx], labels[va_idx]
                fold_model = _fit_analytical_lr(tk, ty, graph, bwd_map, self.alpha)
                if fold_model is None:
                    continue
                tc = Counter(tk)
                tp = Counter(k for k, yv in zip(tk, ty) if yv)
                vc = Counter(vk)
                cand = np.array([(tp[k] == 0) and (k in graph.index) and pd.notna(graph.at[k, "tier"]) for k in vk])
                if not cand.any():
                    continue
                idxs = np.flatnonzero(cand)
                x_val = pd.DataFrame(
                    [_row_record(vk[i], float(tc[vk[i]]), float(vc[vk[i]]), graph, bwd_map, tp) for i in idxs],
                    columns=TIMELINE_FEATURE_COLUMNS,
                )
                oof_scores.append(fold_model.predict_proba(x_val)[:, 1])
                oof_targets.append(vy[cand])
        if not oof_scores:
            return None
        scores = np.clip(np.concatenate(oof_scores), 1e-4, 1.0 - 1e-4)
        targets = np.concatenate(oof_targets)
        if np.unique(targets).size < 2:
            return None
        calibrator = LogisticRegression(C=10.0)
        calibrator.fit(log_odds(scores)[:, None], targets)
        return calibrator

    def predict_cold_risk(self, query_features):
        """Return (candidate_mask, calibrated_probability) for cold query rows with a BM jump."""
        n_rows = len(query_features)
        if n_rows == 0 or self.model_ is None:
            return np.zeros(n_rows, dtype=bool), np.zeros(n_rows, dtype=float)
        query_num = _numeric_frame(query_features)
        query_keys = pd.util.hash_pandas_object(query_num, index=False).to_numpy()
        query_counts = Counter(query_keys)
        query_profiles = _summarize_profiles(query_num, query_keys).drop(columns=["pid"])
        # Merge stored fitting profiles and unlabeled query profiles (summing row counts).
        combined = pd.concat([self.reference_profiles_, query_profiles], ignore_index=True)
        merged = combined.groupby(list(PROFILE_COLUMNS), as_index=False).agg(
            n_portfolio=("n_portfolio", "sum"),
            key=("key", "first"),
        )
        merged["pid"] = np.arange(len(merged))
        graph, bwd_map = build_timeline_graph(merged)

        cand = np.array([
            (self.fit_positives_[k] == 0) and (k in graph.index) and pd.notna(graph.at[k, "tier"])
            for k in query_keys
        ], dtype=bool)
        probabilities = np.zeros(n_rows, dtype=float)
        if not cand.any():
            return cand, probabilities
        idxs = np.flatnonzero(cand)
        design = pd.DataFrame(
            [_row_record(query_keys[i], float(self.fit_counts_[query_keys[i]]),
                         float(query_counts[query_keys[i]]), graph, bwd_map, self.fit_positives_)
             for i in idxs],
            columns=TIMELINE_FEATURE_COLUMNS,
        )
        raw = np.clip(self.model_.predict_proba(design)[:, 1], 1e-4, 1.0 - 1e-4)
        if self.calibrator_ is not None:
            probabilities[cand] = self.calibrator_.predict_proba(log_odds(raw)[:, None])[:, 1]
        else:
            probabilities[cand] = raw
        return cand, probabilities

    def augment_predictions(self, query_features, base_risk, base_prices, base_flags, base_cutoff):
        """Add calibrated cold BM-window detections and MAE-aware quotes to base predictions."""
        cand, cold_risk = self.predict_cold_risk(query_features)
        if not cand.any():
            return base_risk, base_prices, base_flags
        # The Bayes-optimal F1 cutoff is F1/2, closely tracked by the batch's base_cutoff.
        cutoff = float(np.clip(base_cutoff, 0.12, 0.265))
        added = cand & (cold_risk >= cutoff) & ~base_flags
        if not added.any():
            return np.where(cand, np.maximum(base_risk, cold_risk), base_risk), base_prices, base_flags
        tail_quotes = self.tail_.predict(query_features)
        # Quote LocalPayoutTail when p >= 0.50 (reduces MAE); use minimal positive floor when p < 0.50.
        added_quotes = np.where(cold_risk >= self.high_confidence_quote, tail_quotes, self.low_confidence_floor)
        new_risk = np.where(cand, np.maximum(base_risk, cold_risk), base_risk)
        new_prices = np.where(added, added_quotes, base_prices)
        new_flags = base_flags | added
        return new_risk, new_prices, new_flags
