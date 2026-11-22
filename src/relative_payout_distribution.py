"""Estimate a profile-relative distribution for payouts absent from its exact history."""

from dataclasses import dataclass

import numpy as np

from src.aggregate_payout_choices import AggregatePayoutChoices, payout_bases
from src.claim_payout_choices import REFERENCE_POINTS
from src.dashboard_model import profile_keys
from src.feature_schema import PAYOUT_FACTOR
from src.payout_pricing import MEDIAN_PRICING_CAP, payout_signature

NOVEL = ("new_cent", "new_scaled")
SCALE_BANDS = np.array([250., 1000., 2500.])


@dataclass
class PayoutContext:
    cent: float
    scaled: float
    cent_group: int
    scaled_group: int

    def for_kind(self, kind):
        if kind == "new_cent":
            return self.cent, self.cent_group
        return self.scaled, self.scaled_group


def anchor_context(history):
    whole, scaled = payout_bases(history)
    subtotal = sum(whole) / 100
    total = max(scaled, default=0) / 100
    if total <= 0:
        return PayoutContext(max(25., subtotal), max(25., subtotal * PAYOUT_FACTOR), 0, 0)
    if subtotal < total:
        cent_reference, cent_group = total - subtotal, 1
    else:
        cent_reference, cent_group = total, 2
    return PayoutContext(max(25., cent_reference), max(25., total * PAYOUT_FACTOR), cent_group, 1)


class RelativePayoutDistribution:
    """Shrink relative novel-payout estimates toward the unconditional samples."""

    def fit(self, features, amounts, *, fitted=None):
        actual = np.asarray(amounts)
        self.model_ = fitted if fitted is not None else AggregatePayoutChoices().fit(features, actual)
        alternatives = self.model_.alternatives(features, actual)
        contexts = self.contexts(features, alternatives.positions, actual)
        selected = alternatives.kind[alternatives.selected]
        observations = {}
        for amount, kind, context in zip(actual[alternatives.positions], selected, contexts):
            if kind not in NOVEL:
                continue
            anchor, group = context.for_kind(kind)
            band = np.searchsorted(SCALE_BANDS, anchor)
            observations.setdefault((kind, group, band), []).append(float(np.log(amount / anchor)))
        self.distributions_ = {key: (len(values), np.quantile(values, REFERENCE_POINTS))
                               for key, values in observations.items()}
        self.tail_cap_ = max(MEDIAN_PRICING_CAP, float(np.quantile(actual[actual > 0], .97)))
        return self

    def contexts(self, features, positions, amounts=None):
        for position, key in zip(positions, profile_keys(features.iloc[positions])):
            history = self.model_.positive_[key].copy()
            if amounts is not None and amounts[position] > 0:
                own = payout_signature(amounts[position])
                history[own] -= 1
                if not history[own]:
                    del history[own]
            yield anchor_context(history)

    def novel_samples(self, context, kind):
        fallback = self.model_.fallback_payouts_[kind]
        anchor, group = context.for_kind(kind)
        count, ratio = self.distributions_.get((kind, group, np.searchsorted(SCALE_BANDS, anchor)), (0, None))
        if count < 15:
            return fallback, np.full(len(fallback), 1 / len(fallback))
        share = count / (count + 75)
        estimates = anchor * np.exp(ratio)
        probabilities = np.r_[np.full(len(fallback), (1 - share) / len(fallback)),
                              np.full(len(estimates), share / len(estimates))]
        return np.r_[fallback, estimates], probabilities

    def outcome_arrays(self, contract, weights_for_row, context, start, stop):
        values, weights = [], []
        for kind, amount, mass in zip(contract.kind[start:stop], contract.amount[start:stop], weights_for_row):
            if kind in NOVEL:
                amounts, fraction = self.novel_samples(context, kind)
                values.extend(amounts)
                weights.extend(mass * fraction)
            else:
                values.append(amount)
                weights.append(mass)
        order = np.argsort(values)
        return np.asarray(values)[order], np.asarray(weights)[order]

    def source_weighted_outcomes(self, contract, weights, context, start, stop, probability, scaled_share):
        kinds = contract.kind[start:stop]
        scaled = (kinds == "repeat_scaled") | (kinds == "new_scaled")
        cent = (kinds == "repeat_cent") | (kinds == "new_cent")
        local = weights.copy()
        scaled_mass, cent_mass = local[scaled].sum(), local[cent].sum()
        # Retain the choice model's relative weights within each payout source.
        # Pool its source share with the independently calibrated source share.
        target_share = (scaled_mass / (scaled_mass + cent_mass) + scaled_share) / 2
        local[scaled] *= probability * target_share / scaled_mass
        local[cent] *= probability * (1 - target_share) / cent_mass
        local[~(scaled | cent)] = 1 - probability
        return self.outcome_arrays(contract, local, context, start, stop)

    @staticmethod
    def reweight_positive(values, weights, probability):
        positive = values > 0
        mass = weights[positive].sum()
        if mass <= 0:
            return weights.copy()
        updated = weights.copy()
        updated[positive] *= probability / mass
        updated[~positive] = 1 - probability
        return updated

    def median(self, values, weights, cap):
        position = min(np.searchsorted(np.cumsum(weights), .5), len(values) - 1)
        return float(np.clip(values[position], min(self.model_.floor_, cap), cap))

    def predict(self, features, probability, fallback, *, scaled_share=None):
        alternatives = self.model_.alternatives(features)
        raw_price = np.array(fallback, copy=True)
        calibrated_price = raw_price.copy()
        tail_price = raw_price.copy()
        # Pool novel amounts on their conditional relative scale, then recalibrate
        # the zero mass separately from the severity mixture.
        cost = raw_price.copy()
        if self.model_.coefficients_ is None or not len(alternatives.starts):
            return raw_price, calibrated_price, tail_price, cost
        raw_weights, _, _ = alternatives.probabilities(self.model_.coefficients_)
        contexts = self.contexts(features, alternatives.positions)
        ends = np.r_[alternatives.starts[1:], len(raw_weights)]
        for position, context, start, stop in zip(alternatives.positions, contexts, alternatives.starts, ends):
            weights_for_row = raw_weights[start:stop]
            values, weights = self.outcome_arrays(alternatives, weights_for_row, context, start, stop)
            calibrated = self.reweight_positive(values, weights, probability[position])
            raw_price[position] = self.median(values, weights, MEDIAN_PRICING_CAP)
            if scaled_share is not None and np.isfinite(scaled_share[position]):
                values, calibrated = self.source_weighted_outcomes(
                    alternatives, weights_for_row, context, start, stop, probability[position], scaled_share[position],
                )
            calibrated_price[position] = self.median(values, calibrated, MEDIAN_PRICING_CAP)
            tail_price[position] = self.median(values, calibrated, self.tail_cap_)
            amount = calibrated_price[position]
            cost[position] = amount - 2 * np.dot(calibrated, np.minimum(values, amount))
        return raw_price, calibrated_price, tail_price, cost
