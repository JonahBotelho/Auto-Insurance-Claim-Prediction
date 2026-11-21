"""Reweight claim calibration by the histories present in the scoring batch."""

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from src.claim_probability import log_odds
from src.payout_history_groups import GROUPS, evidence_groups
from src.claim_logit_offset import fit_logit_offset
from src.profile_rank_gate import gate_inputs
from src.feature_schema import RANDOM_SEED


def history_design(rows, peers):
    groups = evidence_groups(rows)
    counts = rows[["prior_positives", "prior_zeros", "whole_count", "scaled_count"]].to_numpy(dtype=float)
    context = rows[["gap_fraction", "price_log", "exposure", "vehicle_age_code"]].to_numpy(dtype=float)
    return np.column_stack([counts, peers, context, np.eye(len(GROUPS))[groups]])


def propensity_weights(fitting, scoring, train_peers, score_peers):
    train = history_design(fitting, train_peers)
    query = history_design(scoring, score_peers)
    design = np.vstack([train, query])
    group = np.r_[fitting.profile.to_numpy(), scoring.profile.to_numpy()]
    query_label = np.r_[np.zeros(len(train)), np.ones(len(query))]
    estimated = np.empty(len(design))
    scales = np.empty(len(design))
    folds = GroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED)
    for fit, held in folds.split(design, query_label, group):
        classifier = HistGradientBoostingClassifier(max_iter=115, max_leaf_nodes=7, min_samples_leaf=45,
                                                    learning_rate=.055, l2_regularization=8., random_state=RANDOM_SEED)
        classifier.fit(design[fit], query_label[fit])
        estimated[held] = classifier.predict_proba(design[held])[:, 1]
        probability = query_label[fit].mean()
        scales[held] = (1 - probability) / probability
    ratio = estimated[:len(train)] / np.maximum(1e-5, 1 - estimated[:len(train)]) * scales[:len(train)]
    return np.clip(ratio, .35, 3.), roc_auc_score(query_label, estimated)


def count_weights(fitting, scoring, train_peers, score_peers):
    def keys(rows, peers):
        return pd.MultiIndex.from_arrays([
            evidence_groups(rows), rows.prior_positives.clip(upper=3), rows.prior_zeros.clip(upper=3),
            np.minimum(peers, 3),
        ])

    training, query = keys(fitting, train_peers), keys(scoring, score_peers)
    known, live = training.value_counts(), query.value_counts()
    ratio = len(query) / len(training)
    # Small evidence cells borrow strength from the overall batch-size ratio.
    local = live.reindex(training, fill_value=0).to_numpy() + 12. * ratio
    denominator = (known.reindex(training).to_numpy() + 12.) * ratio
    return np.clip(local / denominator, .35, 3.)


def calibrate_for_batch(fitting, scoring, training, query, original):
    first, second, train_peers = training
    live_first, live_second, query_peers = query
    anchor = (first + second) / 2
    train = gate_inputs(fitting, first, second, train_peers)
    held = gate_inputs(scoring, live_first, live_second, query_peers)
    scaler = StandardScaler().fit(train)
    train, held = [np.c_[np.ones(len(item)), np.clip(scaler.transform(item), -5., 5.)] for item in (train, held)]
    relevance = .22 + np.exp(-((anchor - .25) / .15) ** 2)
    target = fitting.actual.gt(0).to_numpy()
    weight, auc = propensity_weights(fitting, scoring, train_peers, query_peers)
    binned = count_weights(fitting, scoring, train_peers, query_peers)
    coefficients = [fit_logit_offset(train, log_odds(anchor), target, relevance * values, penalty=.05)
                    for values in (np.ones(len(weight)), np.sqrt(weight), np.sqrt(binned))]
    ordinary, weighted, cell = [np.clip(held @ coefficient, -.70, .70) for coefficient in coefficients]
    base = log_odds(original)
    adjustment = {
        "history_equal": .5 * ordinary,
        "history_importance": .5 * weighted,
        "history_importance_delta": weighted - ordinary,
        "history_cell_delta": cell - ordinary,
    }
    return {name: expit(base + shift) for name, shift in adjustment.items()}, float(auc)
