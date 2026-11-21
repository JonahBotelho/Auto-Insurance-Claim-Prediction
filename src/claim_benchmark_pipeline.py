"""Rebuild the count, analogue, pricing, and review stages of the claim benchmark."""

import numpy as np
from scipy.special import expit

from src.claim_count_mixture import MAX_ROWS
from src.claim_history_leaf_retrieval import ClaimHistoryLeafRetrieval
from src.claim_probability import log_odds
from src.dashboard_model import profile_keys
from src.directional_claim_review import REVIEW_BAND, REVIEW_MARGIN, TARGET_RISK, panel, review_models
from src.local_payout_tail import LocalPayoutTail
from src.local_tail_refinement import refine_low_quotes
from src.payout_collision_evidence import PayoutCollisionEvidence
from src.profile_count_hurdle import ProfileCountHurdle
from src.profile_count_survival import COUNT, ProfileCountSurvival
from src.source_tail_pricing import SourceTailPricing


def count_quotes(features, target, query, anchor, boundary, family, trimmed):
    _, inverse, peers = np.unique(profile_keys(query), return_inverse=True, return_counts=True)
    point, continuation, warm = ProfileCountSurvival().fit(features, target).predict(query, anchor)
    _, hurdle, _ = ProfileCountHurdle().fit(features, target).predict(query, anchor)
    sizes = np.minimum(peers, MAX_ROWS)
    baseline, center = log_odds(point), log_odds(anchor)
    decisions = []
    for law in (continuation, hurdle):
        rate = ((law @ COUNT) / sizes)[inverse]
        shift = np.clip(log_odds(rate) - baseline, -.6, .6)
        decisions.append(expit(center + shift / 2) >= boundary)
    first, second = decisions
    near = warm & (np.abs(anchor - boundary) <= .075)
    flags = np.where(near & (first == second), second, family > 0.)
    adds = SourceTailPricing().fit(features, target).support(query)[1]
    quotes = np.where(family > 0., trimmed, adds)
    return np.where(flags, quotes, 0.), adds


def supported_admission(evidence, near):
    risk, support, confidence, _ = evidence.T
    return near & (support >= 10.) & (confidence >= .90) & (risk >= .33)


def refined_quotes(features, target, query, anchor, boundary, original, addition_quote, tail):
    ranked, warm = ClaimHistoryLeafRetrieval(ranking=True).fit(features, target).predict(query)
    regular, _ = ClaimHistoryLeafRetrieval().fit(features, target).predict(query)
    near = warm & (np.abs(anchor - boundary) <= .075)
    added = (original == 0.) & supported_admission(ranked, near) & supported_admission(regular, near)
    amounts = np.where(added, addition_quote, original)
    low = (amounts > 0.) & (amounts <= tail.raw_floor_) & (ranked[:, 0] < .40)
    candidate = np.maximum(.8 * amounts, np.minimum(amounts, tail.predict(query)))
    amounts = np.where(added | low, candidate, amounts)
    collision = PayoutCollisionEvidence().fit(features, target)
    base, full, _ = collision.predict(query, anchor)
    return refine_low_quotes(tail, query, amounts, collision.paid_, np.maximum(base, full) < .36)


def benchmark_predictions(settings, features, target, query, reference):
    anchor = reference["anchor_score"].to_numpy()
    boundary = reference["cutoff"].iloc[0]
    family = reference["family_veto"].to_numpy()
    trimmed = reference["risk_supported_trim"].to_numpy()
    tail = LocalPayoutTail().fit(features, target)
    counted, addition_quote = count_quotes(features, target, query, anchor, boundary, family, trimmed)
    amounts = refined_quotes(features, target, query, anchor, boundary, counted, addition_quote, tail)
    pooled, missed, _ = review_models(settings, features, target)
    design, surrogate, eligible = panel(settings, features, target, query)
    # Apply both review models only to missed-claim candidates accepted by the
    # archived anchor's review band and rejected by the refitted surrogate.
    eligible &= (amounts == 0.) & ~surrogate & (np.abs(anchor - boundary) <= REVIEW_BAND)
    high = TARGET_RISK + REVIEW_MARGIN
    added = eligible & (pooled.predict_proba(design)[:, 1] >= high) & (missed.predict_proba(design)[:, 1] >= high)
    return {"collision_price": amounts, "review_additions": np.where(added, tail.predict(query), amounts)}
