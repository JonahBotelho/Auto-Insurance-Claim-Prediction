"""Estimate claim presence and the positive fraction of the scoring profile."""

from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np

from src.feature_schema import RANDOM_SEED
from src.payout_source_calibration import payout_source_design


def group_frame(rows):
    groups = ["profile", "fold"]
    grouped = rows.groupby(groups, sort=False)
    totals = grouped.size().to_frame("peers")
    if "actual" in rows:
        totals["hits"] = rows.actual.gt(0).groupby([rows[name] for name in groups], sort=False).sum()
    return rows.drop_duplicates(groups).set_index(groups).join(totals)


def group_inputs(rows):
    return np.column_stack([payout_source_design(rows), rows.peers, rows.probability, rows.hist_oof,
                            rows.template_single, rows.template_batch, rows.aggregate_choice])


def fit_presence(fitting):
    fitting = fitting.loc[fitting.prior_positives.gt(0)]
    model_args = dict(iterations=220, depth=3, learning_rate=.045, l2_leaf_reg=42,
                      random_seed=RANDOM_SEED, thread_count=2, verbose=False, allow_writing_files=False)
    relevance = .35 + np.exp(-((fitting.probability - .25) / .16) ** 2)
    input_rows = group_inputs(fitting)
    any_claim = CatBoostClassifier(**model_args).fit(input_rows, fitting.hits.gt(0), sample_weight=relevance)
    multi = fitting.peers.gt(1) & fitting.hits.gt(0)
    fraction_model = CatBoostRegressor(**{**model_args, "iterations": 160}).fit(
        input_rows[multi], (fitting.hits / fitting.peers).loc[multi], sample_weight=fitting.peers.loc[multi],
    )
    return any_claim, fraction_model


def score_presence(models, query):
    known = query.prior_positives.gt(0)
    updated = query.probability_template_expected_f1.copy()
    if not known.any():
        return updated
    held = query.loc[known]
    held_input = group_inputs(held)
    any_claim, fraction_model = models
    probability = any_claim.predict_proba(held_input)[:, 1]
    fraction = np.clip(fraction_model.predict(held_input), 1 / held.peers.to_numpy(), 1.)
    updated.loc[known] = probability * fraction
    return updated


def presence_forecast(fitting, query):
    return score_presence(fit_presence(fitting), query)


