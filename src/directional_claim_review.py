"""Fitting-only count, payout composition, and analogue review model."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.claim_count_mixture import MAX_ROWS
from src.claim_history_leaf_retrieval import ClaimHistoryLeafRetrieval
from src.claim_probability import log_odds
from src.dashboard_config import build_dashboard_model
from src.dashboard_model import profile_keys
from src.feature_schema import RANDOM_SEED
from src.masked_payout_composition import MaskedPayoutComposition
from src.masked_payout_history import aggregate_history, roster_keys
from src.profile_count_hurdle import ProfileCountHurdle
from src.profile_count_survival import COUNT, ProfileCountSurvival


def count_evidence(features, target, query):
    keys = profile_keys(query)
    _, first, inverse, sizes = np.unique(keys, return_index=True, return_inverse=True, return_counts=True)
    fallback = np.full(len(query), np.mean(target > 0.))
    point, continuation, warm = ProfileCountSurvival().fit(features, target).predict(query, fallback)
    _, hurdle, _ = ProfileCountHurdle().fit(features, target).predict(query, fallback)
    rates = [((law @ COUNT) / np.minimum(sizes, MAX_ROWS))[inverse] for law in (continuation, hurdle)]
    _, zeros, paid = aggregate_history(features, target)
    whole, scaled = np.array([
        [sum(n for (_, cent), n in paid.get(key, {}).items() if cent == mark) for mark in (True, False)]
        for key in keys[first]
    ]).T
    prior = log_odds(point)
    controls = np.column_stack([prior, np.log1p(sizes[inverse]), np.log1p(whole[inverse]), np.log1p(scaled[inverse]),
                                np.log1p([zeros[key] for key in keys])])
    first_shift, second_shift = (np.clip(log_odds(rate) - prior, -1., 1.) for rate in rates)
    count_features = np.column_stack([first_shift, second_shift,
                                     (first_shift + second_shift) * (sizes[inverse] > 1),
                                     (first_shift + second_shift) * (scaled[inverse] > 0)])
    return point, warm, controls, count_features


def evidence(features, amounts, query):
    point, warm, controls, counts = count_evidence(features, amounts, query)
    direct, typed, marked, _ = MaskedPayoutComposition().fit(
        features, amounts, roster_keys(features, query)
    ).predict(query, point)
    blocks = [controls, counts, marked, log_odds(direct), log_odds(typed)]
    risks = [point, direct, typed]
    for ranking in (False, True):
        estimates, supported = ClaimHistoryLeafRetrieval(ranking=ranking).fit(features, amounts).predict(query)
        risk, effective, confidence, prior = estimates.T
        risk = np.where(supported, risk, point)
        risks.append(risk)
        blocks.extend([log_odds(risk), np.log1p(np.where(supported, effective, 0.)),
                       np.where(supported, confidence, 0.), log_odds(np.where(supported, prior, point))])
    # Retain enough low-risk cases to learn the transition around the F1 cutoff.
    eligible = warm & (np.maximum.reduce(risks) >= .09) & (point <= .85)
    return np.column_stack(blocks), eligible, point


TARGET_RISK = .275
REVIEW_BAND = .10
REVIEW_MARGIN = .02


def panel(settings, fitting, amounts, query):
    # A refitted version of the source-tail classifier supplies the inner decision.
    # Training on its actual mistakes makes the review task match prediction time.
    anchor = build_dashboard_model(settings, settings["threshold"]).fit(fitting, amounts)
    risk = anchor.predict_claim_probability(query)
    boundary = anchor.adaptive_claim_threshold(risk)
    extra, warm, _ = evidence(fitting, amounts, query)
    offset = log_odds(risk) - log_odds(boundary)
    design = np.column_stack([extra, offset, np.abs(offset)])
    return design, risk >= boundary, warm & (np.abs(risk - boundary) <= REVIEW_BAND)


def review_models(settings, features, amounts):
    outcome = amounts > 0.
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    designs, decisions, observed = [], [], []
    for learn, validate in folds.split(features, outcome):
        branch = panel(settings, features.iloc[learn], amounts[learn], features.iloc[validate])
        design, flagged, eligible = branch
        designs.append(design[eligible])
        decisions.append(flagged[eligible])
        observed.append(outcome[validate][eligible])
    design, flagged, target = map(np.concatenate, (designs, decisions, observed))
    models = []
    for subset in (np.ones(len(flagged), dtype=bool), ~flagged, flagged):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=.06, max_iter=300))
        models.append(model.fit(design[subset], target[subset]))
    return models
