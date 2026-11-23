"""Learn a remaining-claims distribution from exactly thinned payout histories."""

from functools import lru_cache
from functools import partial

from lightgbm import LGBMRegressor
import numpy as np
import pandas as pd
from scipy.stats import binom, hypergeom

from src.masked_payout_history import (
    HISTORY_COLUMNS, PAYOUT_SHAPE_COLUMNS, history_design, aggregate_history, describe_history, masked_states, payout_shape,
)
from src.claim_count_mixture import MAX_ROWS
from src.dashboard_model import profile_keys
from src.feature_schema import RANDOM_SEED

RETENTION = (.78, .92)
COUNT = np.arange(MAX_ROWS + 1)
PARAMETERS = dict(n_estimators=185, num_leaves=11, min_child_samples=85, learning_rate=.035,
                  reg_lambda=48, random_state=RANDOM_SEED, n_jobs=2, verbosity=-1, deterministic=True,
                  force_col_wise=True)


@lru_cache(maxsize=256)
def possible_scoring_counts(missing_rows, missing_claims):
    sizes = np.arange(1, min(missing_rows, MAX_ROWS) + 1)
    # Force one of the unseen rows into scoring, then sample its siblings.
    # The count law given that size is hypergeometric by exchangeability.
    draw = binom.pmf(sizes - 1, missing_rows - 1, .55)
    outcome = hypergeom.pmf(COUNT, missing_rows, missing_claims, sizes[:, None])
    return sizes, draw, outcome


def masked_scoring_examples(zero, payments, *, zero_only=False, retention=RETENTION, rich_history=False):
    totals = {}
    for retained in retention:
        for seen_zero, seen_paid, negative, positive in masked_states(zero, payments, retained):
            supported = (seen_zero > 0 and not seen_paid) if zero_only else bool(seen_paid)
            if not supported or negative + positive < 1e-8:
                continue
            paid_left = sum(payments.values()) - sum(seen_paid.values())
            unseen = zero - seen_zero + paid_left
            sizes, selection, outcomes = possible_scoring_counts(unseen, paid_left)
            for size, volume, law in zip(sizes, selection, outcomes, strict=True):
                if volume <= 1e-8:
                    continue
                total = seen_zero + sum(seen_paid.values()) + size
                inputs = describe_history(seen_zero, seen_paid, total)
                if rich_history:
                    inputs += payout_shape(seen_paid)
                totals.setdefault(inputs, np.zeros(len(COUNT)))
                totals[inputs] += (negative + positive) * volume * law / len(retention)
    return totals.items()


def continuation_rows(covariates, outcome):
    survival = np.cumsum(outcome[:, ::-1], axis=1)[:, ::-1]
    group, step = np.nonzero(survival[:, :-1] > 1e-8)
    weights = survival[group, step]
    reaches_step = survival[group, step + 1] / weights
    predictors = covariates.iloc[group].assign(claim_step=step + 1)
    return predictors, reaches_step, weights


class ProfileCountSurvival:
    """Estimate both marginal risk and a distribution of exact profile counts."""

    def __init__(self, retention=RETENTION, rich_history=False):
        self.retention = retention
        self.rich_history = rich_history

    def history_columns(self):
        return HISTORY_COLUMNS + PAYOUT_SHAPE_COLUMNS if getattr(self, "rich_history", False) else HISTORY_COLUMNS

    def fit(self, features, target):
        simulator = partial(masked_scoring_examples, retention=self.retention, rich_history=self.rich_history)
        design, outcome = self.training_panel(features, target, simulator)
        sample_weight = outcome.sum(axis=1)
        target = (outcome @ COUNT / sample_weight) / design.unseen_rows
        self.point_ = LGBMRegressor(objective="cross_entropy", **PARAMETERS).fit(design, target, sample_weight=sample_weight)
        self.fit_distribution(design, outcome)
        return self

    def training_panel(self, features, target, simulator):
        keys, self.zero_, self.paid_ = aggregate_history(features, target)
        positions, histories, counts = [], [], []
        for position in np.flatnonzero(~pd.Index(keys).duplicated()):
            key = keys[position]
            if key not in self.paid_:
                continue
            for history, mass in simulator(self.zero_[key], self.paid_[key]):
                positions.append(position)
                histories.append(history)
                counts.append(mass)
        design = history_design(features.iloc[positions], histories, self.history_columns())
        return design, np.asarray(counts)

    def fit_distribution(self, design, outcome):
        matrix, target, weight = continuation_rows(design, outcome)
        self.survival_ = LGBMRegressor(objective="cross_entropy", **PARAMETERS).fit(matrix, target, sample_weight=weight)

    def distributions(self, design, sizes):
        positions, steps = np.nonzero(COUNT[1:] <= sizes[:, None])
        matrix = design.iloc[positions].assign(claim_step=steps + 1)
        continuation = np.zeros((len(design), MAX_ROWS))
        continuation[positions, steps] = self.survival_.predict(matrix)
        survivor = np.column_stack([np.ones(len(design)), np.cumprod(continuation, axis=1)])
        return survivor - np.column_stack([survivor[:, 1:], np.zeros(len(design))])

    def predict(self, features, fallback):
        keys, first, inverse, peers = np.unique(profile_keys(features), return_index=True, return_inverse=True, return_counts=True)
        eligible = np.fromiter((key in self.paid_ for key in keys), dtype=bool) & (peers <= MAX_ROWS)
        sizes = np.minimum(peers, MAX_ROWS)
        probability = np.asarray(fallback)[first].copy()
        law = binom.pmf(COUNT, sizes[:, None], probability[:, None])
        if eligible.any():
            histories = []
            for key, remaining in zip(keys[eligible], peers[eligible], strict=True):
                paid, zero = self.paid_[key], self.zero_[key]
                state = describe_history(zero, paid, zero + sum(paid.values()) + remaining)
                histories.append(state + payout_shape(paid) if getattr(self, "rich_history", False) else state)
            design = history_design(features.iloc[first[eligible]], histories, self.history_columns())
            probability[eligible] = self.point_.predict(design)
            law[eligible] = self.distributions(design, sizes[eligible])
        return probability[inverse], law, eligible[inverse]
