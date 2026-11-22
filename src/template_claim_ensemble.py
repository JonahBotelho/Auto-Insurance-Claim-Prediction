"""Combine payout templates, adaptive F1 decisions and calibrated payout distributions."""

import numpy as np
from scipy.special import logit

from src.aggregate_payout_choices import AggregatePayoutChoices
from src.claim_template_mixture import ClaimTemplateMixture
from src.profile_ensemble_model import PROFILE_STACK_CUTOFF, ProfileHistoryEnsemble, history_inputs
from src.relative_payout_distribution import RelativePayoutDistribution


def aggregate_stack_inputs(rows, *, batch=True, template=True):
    evidence = history_inputs(rows, batch=batch)
    names = ["aggregate_choice", "template_single", "template_batch"] if template else ["aggregate_choice"]
    logits = logit(np.clip(rows[names].to_numpy(), .005, .995))
    return np.column_stack([evidence, logits])


def aggregate_stack_weights(rows):
    center = (rows.hist_oof.to_numpy() + rows.aggregate_choice.to_numpy()) / 2
    return .25 + np.exp(-((center - PROFILE_STACK_CUTOFF) / .2) ** 2)


def expected_f1_cutoff(probability, prevalence):
    ranked = np.sort(probability)[::-1]
    if not len(ranked):
        return np.inf
    ends = np.flatnonzero(np.r_[ranked[1:] != ranked[:-1], True])
    # Estimate the claim count from the fitting prevalence and expected true
    # positives from calibrated scores. Treat tied scores as one decision.
    f1 = 2 * np.cumsum(ranked)[ends] / (ends + 1 + prevalence * len(ranked))
    return float(ranked[ends[np.argmax(f1)]])


class TemplateClaimEnsemble(ProfileHistoryEnsemble):
    def fit(self, features, amounts):
        super().fit(features, amounts)
        self.prevalence_ = float(np.mean(np.asarray(amounts) > 0))
        aggregate = self.components_[3]
        self.relative_pricer_ = RelativePayoutDistribution().fit(features, amounts, fitted=aggregate)
        return self

    def _fit_components(self, features, amounts, *, price):
        base = super()._fit_components(features, amounts, price=price)
        aggregate = base[0].aggregate_payout_model_
        if aggregate is None:
            aggregate = AggregatePayoutChoices().fit(features, amounts)
        template = ClaimTemplateMixture().fit(features, amounts)
        return (*base, aggregate, template)

    def _scores(self, components, features):
        raw, prices = super()._scores(components[:3], features)
        aggregate, template = components[3:]
        probability = raw.hist_oof.to_numpy()
        aggregate_score, _ = aggregate.predict(features, probability, prices)
        single, batch = template.predict(features, probability)
        return raw.assign(aggregate_choice=aggregate_score, template_single=single, template_batch=batch), prices

    def _forecast(self, features):
        scores, base = self._scores(self.components_, features)
        return self._stack_probability(scores), base

    def predict_components(self, features):
        probability, fallback = self._forecast(features)
        _, _, tail, _ = self.relative_pricer_.predict(features, probability, fallback)
        return probability, np.where(fallback > 0, tail, 0.)

    def predict_claim_probability(self, features):
        return self._forecast(features)[0]

    def adaptive_claim_threshold(self, probabilities):
        if len(probabilities) * self.prevalence_ < 10:
            # At small volumes, expected F1 is unstable and can flag even a
            # single low-risk row. Use the configured reference threshold.
            return self.claim_probability_threshold
        return expected_f1_cutoff(probabilities, self.prevalence_)

    def predict(self, features):
        probability, prices = self.predict_components(features)
        return np.where(probability >= self.adaptive_claim_threshold(probability), prices, 0.)

    stack_inputs = staticmethod(aggregate_stack_inputs)
    stack_weights = staticmethod(aggregate_stack_weights)
