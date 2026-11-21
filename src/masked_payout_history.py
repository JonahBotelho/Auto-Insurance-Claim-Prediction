"""Describe partially observed payout histories and simulate masked outcomes."""

from collections import Counter
from itertools import product
from math import comb

import numpy as np
import pandas as pd

from src.claim_rating_features import actuarial_features
from src.dashboard_model import profile_keys
from src.feature_schema import PAYOUT_FACTOR
from src.payout_pricing import payout_signature


def describe_history(zero, paid, total):
    cent = {base: count for (base, whole), count in paid.items() if whole}
    scaled = {base: count for (base, whole), count in paid.items() if not whole}
    maximum = max(scaled, default=0)
    component_sum = sum(cent)
    whole_rows, scaled_rows = sum(cent.values()), sum(scaled.values())
    peers = total - zero - whole_rows - scaled_rows
    return (
        zero, whole_rows, scaled_rows, len(cent), len(scaled), sum(value >= 2 for value in cent.values()),
        maximum > 0 and maximum == component_sum, maximum > component_sum > 0, maximum < component_sum and maximum > 0,
        np.log1p(max(maximum, component_sum) / 100), np.log1p(max(0, maximum - component_sum) / 100),
        total, peers, (whole_rows + scaled_rows) / max(.5, zero + whole_rows + scaled_rows),
    )


HISTORY_COLUMNS = (
    "past_zeros", "past_whole", "past_scaled", "whole_families", "scaled_families", "repeated_components",
    "reconciled_total", "partial_total", "anomalous_total", "largest_total", "total_gap", "roster_size", "unseen_rows",
    "historical_claim_share",
)


def aggregate_history(features, amounts):
    keys = profile_keys(features)
    zero, paid = Counter(), {}
    for key, amount in zip(keys, amounts):
        if amount > 0:
            paid.setdefault(key, Counter())[payout_signature(amount)] += 1
        else:
            zero[key] += 1
    return keys, zero, paid


def masked_states(zero, paid, retention):
    signatures = tuple(paid)
    counts = tuple(paid.values())
    for kept in product(range(zero + 1), *(range(value + 1) for value in counts)):
        negatives_left = zero - kept[0]
        positives_left = sum(counts) - sum(kept[1:])
        unseen = negatives_left + positives_left
        if unseen == 0:
            continue
        ways = comb(zero, kept[0])
        for count, included in zip(counts, kept[1:]):
            ways *= comb(count, included)
        probability = ways * retention ** sum(kept) * (1 - retention) ** (unseen - 1)
        observed = {signature: included for signature, included in zip(signatures, kept[1:]) if included}
        yield kept[0], observed, probability * negatives_left, probability * positives_left


def history_design(features, histories):
    covariates = actuarial_features(features, np.zeros(len(features))).drop(columns="zero_history")
    return pd.concat([covariates.reset_index(drop=True), pd.DataFrame(histories, columns=HISTORY_COLUMNS)], axis=1)


def roster_keys(fitting, query):
    return pd.Series(np.r_[profile_keys(fitting), profile_keys(query)]).value_counts(sort=False)


def signature_amount(signature):
    cents, whole = signature
    return cents * (1. if whole else PAYOUT_FACTOR) / 100.


def observed_support(paid):
    whole = sum(cents for cents, cent in paid if cent)
    scaled = max((cents for cents, cent in paid if not cent), default=0)
    values = [signature_amount(signature) for signature in paid]
    if whole > 0:
        values.append(whole * PAYOUT_FACTOR / 100.)
    if scaled > whole > 0:
        values.append((scaled - whole) / 100.)
    return np.unique(values)
