"""Describe the fitted payout choices available for each profile."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.preprocessing import StandardScaler

from src.claim_probability import log_odds
from src.claim_logit_offset import fit_logit_offset
from src.profile_rank_gate import ProfileRankGate
from src.dashboard_model import profile_keys
from src.feature_schema import PAYOUT_FACTOR
from src.payout_pricing import payout_signature, whole_cent
from src.relative_payout_distribution import RelativePayoutDistribution

NAMES = (
    "positive_choice_mass", "observed_payout_mass", "inferred_payment_mass", "whole_payment_share",
    "largest_choice_share", "choice_entropy", "positive_median", "positive_spread",
    "low_amount_share", "closest_amount_share", "observed_range_share",
)


def weighted_quantile(values, mass, fractions):
    running = np.cumsum(mass)
    return values[np.minimum(np.searchsorted(running, fractions), len(running) - 1)]


def summarize(pricer, contract, context, start, stop, weight, history):
    kind = contract.kind[start:stop]
    keep = kind != "zero"
    positive_mass = weight[keep].sum()
    conditional = weight[keep] / positive_mass
    repeats = np.isin(kind, ("repeat_cent", "repeat_scaled"))
    observed = np.array([repeat and payout_signature(amount) in history
                         for repeat, amount in zip(repeats, contract.amount[start:stop])])
    whole = np.isin(kind, ("new_cent", "repeat_cent"))
    amount, mass = pricer.outcome_arrays(contract, weight, context, start, stop)
    paid = amount > 0.
    candidates = amount[paid]
    mass = mass[paid] / mass[paid].sum()
    low, center, high = weighted_quantile(candidates, mass, [.10, .50, .90])
    recorded = [base / 100 if cent else base / 100 * PAYOUT_FACTOR for base, cent in history]
    observed_min, observed_max = min(recorded), max(recorded)
    nearest = np.min(np.abs(np.log(candidates[:, None] / np.asarray(recorded))), axis=1)
    return (
        positive_mass, weight[observed].sum() / positive_mass, weight[repeats & ~observed].sum() / positive_mass,
        weight[whole].sum() / positive_mass, conditional.max(),
        -np.dot(conditional, np.log(np.maximum(conditional, 1e-12))),
        np.log1p(center), np.log(high / low), mass[candidates <= pricer.model_.floor_].sum(),
        mass[nearest <= np.log(1.05)].sum(), mass[(candidates >= observed_min) & (candidates <= observed_max)].sum(),
    )


class PayoutDistributionEvidence(RelativePayoutDistribution):
    def transform(self, features):
        keys, first, inverse = np.unique(profile_keys(features), return_index=True, return_inverse=True)
        profiles = features.iloc[first]
        alternatives = self.model_.alternatives(profiles)
        count = len(alternatives.positions)
        output = np.zeros((len(profiles), len(NAMES)))
        if count and self.model_.coefficients_ is not None:
            weights, _, _ = alternatives.probabilities(self.model_.coefficients_)
            contexts = self.contexts(profiles, alternatives.positions)
            stops = np.r_[alternatives.starts[1:], len(weights)]
            for position, context, start, stop in zip(alternatives.positions, contexts, alternatives.starts, stops):
                output[position] = summarize(self, alternatives, context, start, stop, weights[start:stop],
                                             self.model_.positive_[keys[position]])
        return pd.DataFrame(output[inverse], columns=NAMES, index=features.index)


def mark_offsets(design, probability, whole, actual, weights):
    prior = state_prior(probability, whole)
    label = np.where(actual <= 0, 0, np.where(whole_cent(actual), 1, 2))
    target = np.eye(3)[label]
    row = np.arange(len(actual))
    weight = weights / weights.sum()
    ridge = .08

    def objective(flat):
        coefficients = flat.reshape(design.shape[1], 2)
        scores = prior.copy()
        scores[:, 1:] += design @ coefficients
        norm = logsumexp(scores, axis=1)
        mass = np.exp(scores - norm[:, None])
        loss = np.dot(weight, norm - scores[row, label]) + ridge / 2 * np.dot(flat, flat)
        gradient = design.T @ ((mass - target)[:, 1:] * weight[:, None])
        return loss, gradient.ravel() + ridge * flat

    fitted = minimize(objective, np.zeros(design.shape[1] * 2), jac=True, method="L-BFGS-B")
    if not fitted.success:
        raise RuntimeError(f"Marked payout-distribution calibration failed. {fitted.message}")
    return fitted.x.reshape(design.shape[1], 2)


def state_prior(probability, whole):
    risk = np.clip(probability, .002, .998)
    share = np.clip(whole, .02, .98)
    return np.log(np.column_stack([1. - risk, risk * share, risk * (1. - share)]))


class DistributionRisk:
    def design(self, rows):
        return np.clip(self.scale_.transform(rows.loc[:, list(NAMES)]), -5., 5.)

    def fit(self, rows, probability, *, ranking=False):
        actual = rows.actual.to_numpy()
        weight = .23 + np.exp(-((probability - .25) / .15) ** 2)
        self.scale_ = StandardScaler().fit(rows.loc[:, list(NAMES)])
        inputs = self.design(rows)
        self.binary_ = fit_logit_offset(inputs, log_odds(probability), actual > 0, weight, penalty=.075)
        self.marked_ = mark_offsets(inputs, probability, rows.whole_payment_share.to_numpy(), actual, weight)
        if ranking:
            self.pairwise_ = ProfileRankGate.fit_rank(inputs, probability, actual > 0, weight)
        return self

    def predict(self, rows, baseline):
        inputs = self.design(rows)
        logit = log_odds(baseline)
        binary = np.clip(inputs @ self.binary_, -.8, .8)
        marked_prior = state_prior(baseline, rows.whole_payment_share.to_numpy())
        positives = marked_prior[:, 1:]
        adjustment = logsumexp(positives + inputs @ self.marked_, axis=1) - logsumexp(positives, axis=1)
        return expit(logit + binary), expit(logit + binary / 2), expit(logit + np.clip(adjustment, -.8, .8))

    def rank(self, rows, baseline):
        inputs = self.design(rows)
        pointwise = np.clip(inputs @ self.binary_, -.8, .8)
        pairwise = np.clip(inputs @ self.pairwise_, -.8, .8)
        logit = log_odds(baseline)
        return expit(logit + pairwise), expit(logit + pairwise / 2), expit(logit + (pairwise + pointwise) / 2)
