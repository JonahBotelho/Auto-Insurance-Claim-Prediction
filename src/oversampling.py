"""Uniform classifier-only duplication and its claim-prior correction."""

import numpy as np
from sklearn.utils import _safe_indexing

from src.feature_schema import (
    MAX_OVERSAMPLING_PERCENT, MIN_OVERSAMPLING_PERCENT, RANDOM_SEED,
)


def validate_oversampling_percent(percent):
    if type(percent) is not int or not MIN_OVERSAMPLING_PERCENT <= percent <= MAX_OVERSAMPLING_PERCENT:
        raise ValueError(
            f"Oversampling percentage must be a whole number from "
            f"{MIN_OVERSAMPLING_PERCENT} to {MAX_OVERSAMPLING_PERCENT}; 0 disables it."
        )
    return percent


def oversample_claims(features, labels, percent):
    """Return sampled inputs, labels and original-to-sampled prior odds ratio."""
    validate_oversampling_percent(percent)
    labels = np.asarray(labels, dtype=int)
    positive_indices = np.flatnonzero(labels)
    positives = len(positive_indices)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("Claim classification requires both zero and positive claims.")
    if percent == 0:
        return features, labels, 1.0
    target_positives = max(positives, round(negatives * percent / (100 - percent)))
    if target_positives == positives:
        return features, labels, 1.0
    duplicates = np.random.default_rng(RANDOM_SEED).choice(
        positive_indices, target_positives - positives, replace=True,
    )
    indices = np.r_[np.arange(len(labels)), duplicates]
    # Negative rows are unchanged, so their counts cancel in the prior odds.
    return _safe_indexing(features, indices), labels[indices], positives / target_positives


def correct_claim_probabilities(probabilities, odds_ratio):
    """Approximate natural-rate probabilities after uniform positive duplication."""
    if odds_ratio == 1.0:
        return probabilities
    return odds_ratio * probabilities / (1 - probabilities + odds_ratio * probabilities)
