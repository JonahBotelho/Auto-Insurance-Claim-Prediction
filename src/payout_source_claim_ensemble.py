"""Blend template forecasts with a separate classifier for each payout source."""

import numpy as np

from src.cold_profile_risk import ColdProfileRisk
from src.dashboard_model import profile_keys
from src.exposure_hazard import MarkedExposureHazard
from src.payout_source_calibration import PayoutSourceCalibration
from src.payout_source_evidence import PayoutSourceEvidence
from src.profile_ensemble_model import fit_history_stack
from src.source_tail_pricing import PayoutSourceQuantiles, SourceTailPricing
from src.template_claim_ensemble import TemplateClaimEnsemble


class PayoutSourceClaimEnsemble(TemplateClaimEnsemble):
    def _fit_components(self, features, amounts, *, price):
        base = super()._fit_components(features, amounts, price=price)
        source = PayoutSourceEvidence(structural=getattr(self, "structural_source_features", False))
        return (*base, source.fit(features, amounts))

    def _scores(self, components, features):
        scores, prices = super()._scores(components[:-1], features)
        return scores.join(components[-1].score_inputs(features)), prices

    def _fit_calibrators(self, data):
        super()._fit_calibrators(data)
        self.source_calibration_ = PayoutSourceCalibration().fit(data, self._fold_calibration_forecasts(data))

    def _fold_calibration_forecasts(self, data):
        forecasts = data.hist_oof.to_numpy(copy=True)
        known = data.prior_positives.gt(0)
        for fold in data.fold.unique():
            training = data.loc[known & data.fold.ne(fold)]
            held = known & data.fold.eq(fold)
            if held.any() and training.claim.nunique() == 2:
                calibration = fit_history_stack(self.stack_inputs(training), training.claim, self.stack_weights(training))
                forecasts[held.to_numpy()] = calibration.predict_proba(self.stack_inputs(data.loc[held]))[:, 1]
        return forecasts

    def _payout_source_forecast(self, features):
        scores, fallback = self._scores(self.components_, features)
        reference = self._stack_probability(scores)
        marked, scaled_share = self.source_calibration_.predict(scores, reference)
        return (reference + marked) / 2, reference, scaled_share, fallback

    def predict_claim_probability(self, features):
        return self._payout_source_forecast(features)[0]

    def predict_components(self, features):
        probability, reference, scaled_share, fallback = self._payout_source_forecast(features)
        _, _, tail, _ = self.relative_pricer_.predict(features, reference, fallback, scaled_share=scaled_share)
        return probability, np.where(fallback > 0, tail, 0.)


class SourceTailClaimEnsemble(PayoutSourceClaimEnsemble):
    def fit(self, features, amounts):
        super().fit(features, amounts)
        self.source_tail_ = SourceTailPricing().fit(features, amounts)
        return self

    def predict_components(self, features):
        probability, prices = super().predict_components(features)
        return probability, self.source_tail_.predict(features, prices)


class LocalTailClaimEnsemble(SourceTailClaimEnsemble):
    def fit(self, features, amounts):
        from src.local_tail_refinement import LocalTailRefinement

        super().fit(features, amounts)
        self.local_tail_ = LocalTailRefinement().fit(features, amounts)
        return self

    def predict_components(self, features):
        probability, prices = super().predict_components(features)
        return probability, self.local_tail_.predict(features, prices)


class ColdProfileClaimEnsemble(SourceTailClaimEnsemble):
    def fit(self, features, amounts):
        super().fit(features, amounts)
        self.cold_profile_risk_ = ColdProfileRisk(self.random_seed).fit(features, amounts)
        return self

    def _payout_source_forecast(self, features):
        forecast, reference, share, fallback = super()._payout_source_forecast(features)
        if self._unfamiliar_batch(features):
            forecast = self.cold_profile_risk_.predict(features)
        return forecast, reference, share, fallback

    def _unfamiliar_batch(self, features):
        observed = self.components_[4].observation_
        return len(features) > 0 and not any(key in observed for key in profile_keys(features))


class ColdPayoutTailClaimEnsemble(ColdProfileClaimEnsemble):
    def predict_components(self, features):
        probability, prices = super().predict_components(features)
        if self._unfamiliar_batch(features):
            tail = self.source_tail_.pooled_tail_
            if self.baseline_model.amount_cap is not None:
                tail = min(tail, self.baseline_model.amount_cap)
            # Low-confidence predictions lack a matching payout history. The
            # lower observed quote limits their absolute error without changing flags.
            prices = np.where(probability < .5, np.minimum(prices, tail * self.baseline_model.severity_scale), prices)
        return probability, prices


class ColdExposureHazardClaimEnsemble(ColdPayoutTailClaimEnsemble):
    def fit(self, features, amounts):
        super().fit(features, amounts)
        self.exposure_hazard_ = MarkedExposureHazard(flexible=True).fit_profiles(features, amounts)
        self.cold_source_quantiles_ = PayoutSourceQuantiles().fit(amounts, (.02,))
        return self

    def predict_components(self, features):
        probability, prices = super().predict_components(features)
        if not self._unfamiliar_batch(features):
            return probability, prices
        marks = self.exposure_hazard_.predict_profiles(features)[:, 1:]
        risk = marks.sum(axis=1)
        cutoff = self.adaptive_claim_threshold(risk)
        quote, = self.cold_source_quantiles_.predict(marks[:, 1] / risk)
        if self.baseline_model.amount_cap is not None:
            quote = np.minimum(quote, self.baseline_model.amount_cap)
        prices = np.where((probability < .5) & (prices > 0.), quote * self.baseline_model.severity_scale, prices)
        # A zero quote vetoes the claim. The base classifier still supplies
        # the risk and threshold used by the standard prediction path.
        return probability, np.where(risk >= cutoff, prices, 0.)
