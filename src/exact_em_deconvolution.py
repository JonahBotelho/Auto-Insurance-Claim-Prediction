"""Exact Binomial-thinning EM deconvolution for reconciled warm-profile calibration."""

from collections import Counter, defaultdict
from itertools import product
from math import comb

import numpy as np

from src.dashboard_model import profile_keys
from src.masked_payout_history import aggregate_history


def canonical_state(zeros, paid):
    """Map (n_zeros, paid_counter) to canonical (z, w_mults, s_mults, structural_tag)."""
    cent = {base: count for (base, whole), count in paid.items() if whole}
    scaled = {base: count for (base, whole), count in paid.items() if not whole}
    whole_mults = tuple(sorted(cent.values()))
    scaled_mults = tuple(sorted(scaled.values()))
    if not scaled:
        tag = "W"
    elif not cent:
        tag = "S"
    elif len(scaled) == 1 and max(scaled) == sum(cent):
        tag = "R"
    elif max(scaled) > sum(cent):
        tag = "G"
    else:
        tag = "A"
    return (int(zeros), whole_mults, scaled_mults, tag)


def state_size(state):
    return state[0] + sum(state[1]) + sum(state[2])


def state_claims(state):
    return sum(state[1]) + sum(state[2])


def source_supports(paid_dict):
    """Count independent whole-cent, scaled, reconciled-sum, and split-sum occurrences across profiles."""
    whole_sup, scaled_sup, sum_sup, split_sup = Counter(), Counter(), Counter(), Counter()
    for observed in paid_dict.values():
        cent = {base: count for (base, whole), count in observed.items() if whole}
        scaled = {base: count for (base, whole), count in observed.items() if not whole}
        whole_sup.update(cent.keys())
        scaled_sup.update(scaled.keys())
        if cent and sum(cent) in scaled:
            sum_sup[sum(cent)] += 1
            split_sup[sum(cent)] += int(len(cent) > 1)
    return whole_sup, scaled_sup, sum_sup, split_sup


def profile_stratum(paid, whole_sup, scaled_sup, split_sup):
    """Assign a warm profile to a payout-magnitude or forfait stratum using fitting-only supports."""
    cent = {base: count for (base, whole), count in paid.items() if whole}
    scaled = {base: count for (base, whole), count in paid.items() if not whole}
    anchor = max(scaled, default=0) or max(cent, default=0)
    other_whole = whole_sup[anchor] - int(anchor in cent)
    other_scaled = scaled_sup[anchor] - int(anchor in scaled)
    other_splits = split_sup[anchor] - int(anchor == sum(cent) and anchor in scaled and len(cent) > 1)
    if other_splits >= 2 or (other_scaled >= 5 and other_whole == 0):
        return "forfait_scaled_split"
    if other_whole >= 5 or other_scaled >= 5:
        return "forfait_common"
    euros = anchor / 100.0
    if euros < 250.0:
        return "small"
    return "large" if euros > 1500.0 else "mid"


def _enumerate_thinnings(state, alpha):
    z, whole_mults, scaled_mults, tag = state
    total_claims = sum(whole_mults) + sum(scaled_mults)
    visibility = 1.0 - (1.0 - alpha) ** total_claims
    obs_probs = defaultdict(float)
    p_z = [comb(z, kz) * (alpha ** kz) * ((1.0 - alpha) ** (z - kz)) for kz in range(z + 1)]
    p_w = [[comb(m, kw) * (alpha ** kw) * ((1.0 - alpha) ** (m - kw)) for kw in range(m + 1)] for m in whole_mults]
    p_s = [[comb(m, ks) * (alpha ** ks) * ((1.0 - alpha) ** (m - ks)) for ks in range(m + 1)] for m in scaled_mults]

    for kz in range(z + 1):
        pz = p_z[kz]
        if pz == 0.0:
            continue
        for kw_tup in product(*(range(m + 1) for m in whole_mults)):
            pw = pz
            for idx, kw in enumerate(kw_tup):
                pw *= p_w[idx][kw]
            if pw == 0.0:
                continue
            for ks_tup in product(*(range(m + 1) for m in scaled_mults)):
                if sum(kw_tup) + sum(ks_tup) == 0:
                    continue
                prob = pw
                for idx, ks in enumerate(ks_tup):
                    prob *= p_s[idx][ks]
                if prob == 0.0:
                    continue
                kept_w = tuple(sorted(k for k in kw_tup if k > 0))
                kept_s = tuple(sorted(k for k in ks_tup if k > 0))
                if not kept_s:
                    obs_tag = "W"
                elif not kept_w:
                    obs_tag = "S"
                elif tag == "R":
                    obs_tag = "R" if len(kept_w) == len(whole_mults) else "G"
                elif tag == "G":
                    obs_tag = "G"
                else:
                    obs_tag = "A"
                obs_probs[(kz, kept_w, kept_s, obs_tag)] += prob
    return obs_probs, visibility


class ExactEMDeconvolution:
    """Binomial-thinning EM deconvolution over empirically observed canonical warm profile states."""

    def __init__(self, alpha=0.80, n_iter=80, max_n=8):
        self.alpha = alpha
        self.n_iter = n_iter
        self.max_n = max_n

    def fit_counts(self, obs_counter, states=None, prior_pi=None, prior_strength=0.0):
        obs_list = [st for st in obs_counter if state_size(st) <= self.max_n]
        if not obs_list:
            self.states_, self.pi_, self.lookup_ = [], {}, {}
            return self
        counts = np.array([obs_counter[st] for st in obs_list], dtype=float)
        obs_idx = {st: i for i, st in enumerate(obs_list)}
        self.states_ = list(states) if states is not None else sorted(set(obs_list))
        n_states, n_obs = len(self.states_), len(obs_list)
        trans = np.zeros((n_states, n_obs), dtype=float)
        visibility = np.zeros(n_states, dtype=float)
        transitions = {}
        for ci, st in enumerate(self.states_):
            obs_probs, vis = _enumerate_thinnings(st, self.alpha)
            visibility[ci] = vis
            transitions[st] = obs_probs
            for obs_st, prob in obs_probs.items():
                oi = obs_idx.get(obs_st)
                if oi is not None:
                    trans[ci, oi] = prob

        if prior_pi is not None:
            base = np.array([prior_pi.get(st, 0.0) for st in self.states_], dtype=float)
            base /= max(base.sum(), 1e-12)
            pi = base.copy()
        else:
            pi = np.array([obs_counter.get(st, 0.0) + 1e-3 for st in self.states_], dtype=float)
            pi /= pi.sum()
            base = pi.copy()

        for _ in range(self.n_iter):
            denom = pi @ trans
            safe_denom = np.where(denom > 1e-15, denom, 1.0)
            # Horvitz-Thompson correction accounts for claiming profiles hidden when all positives are unobserved.
            expected_total = (pi * (trans @ (counts / safe_denom))) / np.maximum(visibility, 1e-6)
            if prior_strength > 0.0:
                expected_total += prior_strength * base
            pi = expected_total / max(expected_total.sum(), 1e-15)

        self.pi_ = dict(zip(self.states_, pi))
        num_rate, denom_map = defaultdict(float), defaultdict(float)
        for st, pi_c in self.pi_.items():
            if pi_c < 1e-12:
                continue
            n_c, k_c = state_size(st), state_claims(st)
            for obs_st, prob in transitions[st].items():
                n_query = n_c - state_size(obs_st)
                if n_query > 0:
                    weight = pi_c * prob
                    denom_map[(obs_st, n_query)] += weight
                    num_rate[(obs_st, n_query)] += weight * ((k_c - state_claims(obs_st)) / n_query)
        self.lookup_ = {key: (num_rate[key] / d, d) for key, d in denom_map.items() if d > 1e-15}
        return self

    def predict_state(self, obs_state, n_query, fallback=0.45):
        return self.lookup_.get((obs_state, int(n_query)), (fallback, 0.0))


class EMReconciledVeto:
    """Compact fitting-only EM summary that adapts retention alpha to each scored batch."""

    def __init__(self, default_alpha=0.80, min_batch_for_alpha=200):
        self.default_alpha = default_alpha
        self.min_batch_for_alpha = min_batch_for_alpha

    def fit(self, features, amounts):
        _, self.zeros_, self.paid_ = aggregate_history(features, amounts)
        self.n_fit_rows_ = len(features)
        self.obs_counts_ = Counter(canonical_state(self.zeros_[k], paid) for k, paid in self.paid_.items())
        return self

    def batch_alpha(self, n_query_rows):
        if n_query_rows < self.min_batch_for_alpha:
            return self.default_alpha
        return float(np.clip(self.n_fit_rows_ / max(1, self.n_fit_rows_ + n_query_rows), 0.55, 0.92))

    def predict(self, query):
        n_rows = len(query)
        warm = np.zeros(n_rows, dtype=bool)
        is_reconciled = np.zeros(n_rows, dtype=bool)
        p_em = np.zeros(n_rows, dtype=float)
        em_mass = np.zeros(n_rows, dtype=float)
        if n_rows == 0 or not self.obs_counts_:
            return warm, is_reconciled, p_em, em_mass
        keys = profile_keys(query)
        query_counts = Counter(keys)
        if not any(k in self.paid_ for k in query_counts):
            return warm, is_reconciled, p_em, em_mass
        alpha = self.batch_alpha(n_rows)
        em = ExactEMDeconvolution(alpha=alpha, n_iter=80).fit_counts(self.obs_counts_)
        for idx, key in enumerate(keys):
            paid = self.paid_.get(key)
            if paid is None:
                continue
            warm[idx] = True
            state = canonical_state(self.zeros_[key], paid)
            is_reconciled[idx] = state[3] == "R"
            p_em[idx], em_mass[idx] = em.predict_state(state, query_counts[key])
        return warm, is_reconciled, p_em, em_mass
