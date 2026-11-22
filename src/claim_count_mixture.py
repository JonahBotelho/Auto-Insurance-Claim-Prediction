"""Infer remaining profile claims from a mixture of censored count distributions."""

from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import binom

from src.dashboard_model import profile_keys
from src.feature_schema import CENT_TOLERANCE

MAX_ROWS = 10
REFERENCE_POPULATION = 100_000
ITERATIONS = 60
CELL_SHRINKAGE = 300.
STATES = np.array([state for state in product(range(MAX_ROWS + 1), repeat=3) if sum(state) <= MAX_ROWS])
STATE_LOOKUP = np.full((MAX_ROWS + 1,) * 3, -1)
STATE_LOOKUP[tuple(STATES.T)] = np.arange(len(STATES))
TOTAL_ROWS = STATES.sum(axis=1)


def cells(rows):
    # Exposure and feature3 have different relationships with whole-cent and
    # fractional-cent claims. Coarse cells give the rare count states support.
    exposure = np.clip((4 * rows.feature1).astype(int), 0, 3)
    rating = np.searchsorted([3, 7], rows.feature3.to_numpy())
    return (3 * exposure + rating).to_numpy()


def count_states(features, amounts):
    amounts = np.asarray(amounts)
    claim = amounts > 0
    whole_cent = np.abs(amounts - np.round(amounts, 2)) < CENT_TOLERANCE
    observed = pd.DataFrame({
        "zero": ~claim, "cent": claim & whole_cent, "fractional": claim & ~whole_cent,
        "cell": cells(features),
    }, index=profile_keys(features))
    grouped = observed.groupby(level=0, sort=False)
    counts = grouped[["zero", "cent", "fractional"]].sum()
    return counts, grouped.cell.first().to_numpy()


def fit_seen_mixture(counts, response, prior, shrinkage):
    mixture = prior.copy()
    penalty = shrinkage * prior
    seen = counts > 0
    observed = response[seen]
    observed_counts = counts[seen]
    # EM responsibilities are estimated on profiles seen at least once.
    # Dividing by the chance of being seen recovers the unconditional mixture.
    for _ in range(ITERATIONS):
        expected = mixture * (observed.T @ (observed_counts / np.maximum(observed @ mixture, 1e-300)))
        mixture = expected + penalty
        mixture /= mixture.sum()
    return mixture


def regional_priors(observations, regions, response, initial):
    seen = 1. - response[0]
    conditional = response[1:] / seen
    size = response.shape[0]
    counts = np.bincount(observations, minlength=size)[1:]
    global_mixture = fit_seen_mixture(counts, conditional, initial, 15.)
    priors = []
    for cell in range(12):
        local = observations[regions == cell]
        histogram = np.bincount(local, minlength=size)[1:]
        mixture = fit_seen_mixture(histogram, conditional, global_mixture, CELL_SHRINKAGE)
        priors.append(mixture / seen)
    return priors


def reference_rates(weights, missing_rows, missing_claims):
    numerator = np.sum(weights * missing_claims, axis=1)
    denominator = np.sum(weights * missing_rows, axis=1)
    return numerator / np.maximum(1e-300, denominator)


def batch_rates(weights, missing_rows, missing_claims, peers, fraction):
    conditional = weights * binom.pmf(peers, missing_rows, fraction)
    probability = missing_claims / np.maximum(missing_rows, 1)
    return np.sum(conditional * probability, axis=1) / np.maximum(1e-300, conditional.sum(axis=1))


class ClaimCountMixture:
    """Fit count priors by binomial thinning of zero, whole-cent and scaled payouts."""

    def fit(self, features, amounts):
        self.counts_, observed_cells = count_states(features, amounts)
        observed_sizes = self.counts_.sum(axis=1)
        use = observed_sizes.le(MAX_ROWS)
        observed_states = STATE_LOOKUP[tuple(self.counts_.loc[use].to_numpy().T)]
        self.population_ = max(REFERENCE_POPULATION, len(features) / .8)
        self.q_ = len(features) / self.population_
        self.response_ = np.prod(binom.pmf(STATES[:, None, :], STATES[None, 1:, :], self.q_), axis=2)
        remaining = np.maximum(0, STATES[None, 1:, :] - STATES[:, None, :])
        self.remaining_rows_ = remaining.sum(axis=2)
        self.remaining_claims_ = remaining[:, :, 1:].sum(axis=2)
        # Initialize every latent count so rare states can receive evidence in EM.
        initial = np.exp(-.22 * TOTAL_ROWS[1:] - 1.5 * STATES[1:, 1:].sum(axis=1))
        initial /= initial.sum()
        self.priors_ = regional_priors(observed_states, observed_cells[use], self.response_, initial)
        return self

    def predict(self, evaluation, fallback):
        keys = profile_keys(evaluation)
        count = self.counts_.reindex(keys, fill_value=0).to_numpy()
        size = count.sum(axis=1)
        valid = size < MAX_ROWS
        with_self = pd.Series(keys).map(pd.Series(keys).value_counts()).to_numpy()
        state = STATE_LOOKUP[tuple(np.minimum(count, MAX_ROWS).T)]
        cell = cells(evaluation)
        singleton = np.array(fallback, copy=True)
        batched = singleton.copy()
        q_eval = len(evaluation) / (self.population_ * (1 - self.q_))
        for region in range(12):
            matched = valid & (cell == region)
            weights = self.response_ * self.priors_[region]
            index = state[matched]
            singleton[matched] = reference_rates(weights, self.remaining_rows_, self.remaining_claims_)[index]
            if q_eval > 1:
                batched[matched] = singleton[matched]
                continue
            for peers in np.unique(with_self[matched]):
                group = matched & (with_self == peers) & (size + peers <= MAX_ROWS)
                index = state[group]
                batched[group] = batch_rates(weights[index], self.remaining_rows_[index],
                                            self.remaining_claims_[index], peers, q_eval)
        return singleton, batched
