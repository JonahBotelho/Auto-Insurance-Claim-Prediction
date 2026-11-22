"""Infer holdout claim counts from thinned payout templates and observed scaled totals."""

from collections import Counter, defaultdict
from itertools import product

import numpy as np
import pandas as pd
from scipy.stats import binom

from src.claim_count_mixture import MAX_ROWS, REFERENCE_POPULATION, batch_rates, cells, reference_rates, regional_priors
from src.dashboard_model import profile_keys
from src.payout_pricing import payout_signature

TEMPLATES = np.array([state for state in product(range(MAX_ROWS + 1), range(MAX_ROWS + 1), range(6), range(3))
                      if state[0] + state[1] + 2 * state[2] + state[3] <= MAX_ROWS])
OBSERVATIONS = np.array([(*state, proof) for state in TEMPLATES
                         for proof in ((1, 2) if state[3] and state[1] + state[2] else (0,))])
OBSERVATION_INDEX = {tuple(state): index for index, state in enumerate(OBSERVATIONS)}


def observed_template(zeros, amounts):
    cent = {amount: count for (amount, is_cent), count in amounts.items() if is_cent}
    scaled = {amount: count for (amount, is_cent), count in amounts.items() if not is_cent}
    if len(scaled) > 1 or max(cent.values(), default=0) > 2:
        return -1
    proof = 0
    if cent and scaled:
        exposed = sum(cent)
        total = next(iter(scaled))
        if exposed > total:
            return -1
        proof = 1 if exposed == total else 2
    counts = Counter(cent.values())
    return OBSERVATION_INDEX.get((zeros, counts[1], counts[2], sum(scaled.values()), proof), -1)


def amount_family_likelihood(single, double, fraction):
    probability = np.zeros((MAX_ROWS + 1, 6))
    probability[0, 0] = 1.
    for _ in range(single):
        updated = (1 - fraction) * probability
        updated[1:] += fraction * probability[:-1]
        probability = updated
    for _ in range(double):
        updated = (1 - fraction) ** 2 * probability
        updated[1:] += 2 * fraction * (1 - fraction) * probability[:-1]
        updated[:, 1:] += fraction ** 2 * probability[:, :-1]
        probability = updated
    return probability


def observation_likelihood(fraction):
    obs, latent = OBSERVATIONS, TEMPLATES[1:]
    response = binom.pmf(obs[:, :1], latent[:, 0], fraction)
    response *= binom.pmf(obs[:, 3:4], latent[:, 3], fraction)
    amount_probability = np.empty_like(response)
    for single, double in np.unique(latent[:, 1:3], axis=0):
        same_counts = (latent[:, 1] == single) & (latent[:, 2] == double)
        distribution = amount_family_likelihood(single, double, fraction)
        amount_probability[:, same_counts] = distribution[obs[:, 1], obs[:, 2], None]
    response *= amount_probability
    all_seen = (obs[:, 1:3].sum(axis=1)[:, None] == latent[:, 1:3].sum(axis=1))
    # Keep some mass for departures from the composition pattern.
    agreement = (obs[:, 4:5] == 1) == all_seen
    response *= np.where(obs[:, 4:5] == 0, 1., np.where(agreement, .98, .02))
    return response


def tally_profiles(fitting, amounts):
    keys = profile_keys(fitting)
    history = defaultdict(Counter)
    zero = Counter()
    regions = {}
    for key, amount, cell in zip(keys, amounts, cells(fitting)):
        regions[key] = cell
        if amount > 0:
            history[key][payout_signature(amount)] += 1
        else:
            zero[key] += 1
    return {key: observed_template(zero[key], history.get(key, {})) for key in regions}, np.array(list(regions.values()))


class ClaimTemplateMixture:
    def fit(self, fitting, amounts):
        self.population_ = max(REFERENCE_POPULATION, len(fitting) / .8)
        self.q_ = len(fitting) / self.population_
        self.observation_, regions = tally_profiles(fitting, amounts)
        observed = np.fromiter(self.observation_.values(), dtype=int)
        self.response_ = observation_likelihood(self.q_)
        observed_claims = OBSERVATIONS[:, 1] + 2 * OBSERVATIONS[:, 2] + OBSERVATIONS[:, 3]
        latent_claims = TEMPLATES[1:, 1] + 2 * TEMPLATES[1:, 2] + TEMPLATES[1:, 3]
        observed_rows = observed_claims + OBSERVATIONS[:, 0]
        latent_rows = latent_claims + TEMPLATES[1:, 0]
        self.rows_left_ = np.maximum(0, latent_rows - observed_rows[:, None])
        self.claims_left_ = np.maximum(0, latent_claims - observed_claims[:, None])
        self.observed_rows_ = observed_rows
        prior = np.exp(-.22 * latent_rows - 1.5 * latent_claims)
        valid = observed >= 0
        self.priors_ = regional_priors(observed[valid], regions[valid], self.response_, prior / prior.sum())
        return self

    def predict(self, evaluation, fallback):
        keys = profile_keys(evaluation)
        observations = np.fromiter((self.observation_.get(key, 0) for key in keys), dtype=int)
        observed_size = self.observed_rows_[np.maximum(0, observations)]
        valid = (observations >= 0) & (observed_size < MAX_ROWS)
        peers = pd.Series(keys).map(pd.Series(keys).value_counts()).to_numpy()
        regions = cells(evaluation)
        individual = np.array(fallback, copy=True)
        batch = individual.copy()
        q_eval = len(evaluation) / (self.population_ * (1 - self.q_))
        for region, prior in enumerate(self.priors_):
            in_region = valid & (regions == region)
            weights = self.response_ * prior
            individual[in_region] = reference_rates(weights, self.rows_left_, self.claims_left_)[observations[in_region]]
            if q_eval > 1:
                batch[in_region] = individual[in_region]
                continue
            for count in np.unique(peers[in_region]):
                in_batch = in_region & (peers == count) & (observed_size + peers <= MAX_ROWS)
                selected = observations[in_batch]
                batch[in_batch] = batch_rates(weights[selected], self.rows_left_[selected], self.claims_left_[selected],
                                             count, q_eval)
        return individual, batch
