"""Pairwise claim reranking with fitting-derived payout and settlement evidence."""

from collections import Counter

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits

from src.claim_probability import log_odds
from src.dashboard_model import profile_keys
from src.history_importance_calibration import calibrate_for_batch
from src.payout_cap_pricing import supported_tails
from src.payout_distribution_evidence import DistributionRisk, PayoutDistributionEvidence
from src.payout_residue_calibration import fit_residue_shift, predict_residue_shift
from src.payout_residue_evidence import PayoutResidueEvidence
from src.payout_source_claim_ensemble import PayoutSourceClaimEnsemble
from src.profile_claim_presence import fit_presence, group_frame, score_presence
from src.profile_rank_gate import ProfileRankGate, head_forecasts
from src.source_tail_pricing import SourceTailPricing


def decision_cost(probability, prices, cutoff):
    near = np.abs(probability - cutoff) <= .06
    benefit = (2 * probability - 1) * prices
    scale = max(50., float(np.median(prices[near]))) if near.any() else 50.
    return np.where(near, probability + .01 * np.clip(benefit / scale, -1., 1.), probability)


def fixed_volume(probability, flags, keys):
    requested = flags.sum()
    if not requested:
        return flags.copy(), np.inf
    order = np.lexsort((keys, -probability))
    ends = np.flatnonzero(np.r_[probability[order[1:]] != probability[order[:-1]], True])
    position = ends[np.argmin(np.abs(ends + 1 - requested))]
    cutoff = probability[order[position]]
    return probability >= cutoff, cutoff


class PairwisePayoutEnsemble(PayoutSourceClaimEnsemble):
    """Fit once and score each portfolio using its observed profile counts."""

    def fit(self, features, amounts):
        with threadpool_limits(limits=2):
            super().fit(features, amounts)
            self.tail_ = SourceTailPricing().fit(features, amounts)
        return self

    def _fit_components(self, features, amounts, *, price):
        components = super()._fit_components(features, amounts, price=price)
        residue = PayoutResidueEvidence().fit(features, amounts)
        distribution = PayoutDistributionEvidence().fit(features, amounts, fitted=components[3])
        return (*components, residue, distribution)

    def _scores(self, components, features):
        scores, prices = super()._scores(components[:-2], features)
        scores = scores.assign(profile=profile_keys(features))
        return scores.join(components[-2].transform(features)).join(components[-1].transform(features)), prices

    def _fit_calibrators(self, rows):
        super()._fit_calibrators(rows)
        calibration = rows.assign(probability=self._fold_calibration_forecasts(rows))
        first, second, peers = head_forecasts(calibration)
        self.rank_ = ProfileRankGate().fit(calibration, (first, second, peers))
        self.presence_ = fit_presence(group_frame(calibration))
        warm = calibration.prior_positives.gt(0).to_numpy()
        supported = calibration.residue_supported.gt(0).to_numpy()
        self.residue_ = fit_residue_shift(calibration.loc[supported], first[supported]) if supported.any() else None
        self.weight_learning_ = calibration.loc[warm]
        self.fitting_heads_ = first[warm], second[warm], peers[warm]
        self.distribution_risk_ = DistributionRisk().fit(self.weight_learning_, (first[warm] + second[warm]) / 2)

    def _history_scores(self, rows, first, prior, warm):
        context = rows.assign(fold=0, probability=prior, probability_template_expected_f1=prior)
        groups = group_frame(context)
        keys = pd.MultiIndex.from_frame(context[["profile", "fold"]])
        second = score_presence(self.presence_, groups).reindex(keys).to_numpy()
        peers = groups.peers.reindex(keys).to_numpy()
        _, base = self.rank_.predict(context, first, second, peers)
        supported = rows.residue_supported.gt(0).to_numpy()
        if self.residue_ is not None and supported.any():
            shift = predict_residue_shift(rows.loc[supported], self.residue_)
            base[supported] = expit(log_odds(base[supported]) + shift)
        # This estimates the covariate shift from scored histories only. The
        # calibration targets are the stored out-of-fold fitting outcomes.
        if np.unique(rows.profile.to_numpy()[warm]).size >= 4:
            scores, _ = calibrate_for_batch(self.weight_learning_, rows.loc[warm], self.fitting_heads_,
                                             (first[warm], second[warm], peers[warm]), base[warm])
            base[warm] = scores["history_importance_delta"]
        return base

    def score_batch(self, features):
        with threadpool_limits(limits=2):
            return self._score_batch(features)

    def _score_batch(self, features):
        return self._ranked_batch(features)[:4]

    def _ranked_batch(self, features):
        if not len(features):
            empty = np.empty(0)
            return empty, empty, empty.astype(bool), self.claim_probability_threshold, empty
        rows, fallback = self._scores(self.components_, features)
        prior = self._stack_probability(rows)
        marked, share = self.source_calibration_.predict(rows, prior)
        first = (prior + marked) / 2
        _, _, prices, _ = self.relative_pricer_.predict(features, prior, fallback, scaled_share=share)
        prices = self.tail_.predict(features, prices)
        warm = rows.prior_positives.gt(0).to_numpy()
        if not warm.any():
            cutoff = self.adaptive_claim_threshold(first)
            return first, prices, first >= cutoff, cutoff, first
        base = self._history_scores(rows, first, prior, warm)
        cutoff = self.adaptive_claim_threshold(base)
        prices = supported_tails(self.relative_pricer_, features, prices, prior, fallback, share)["single_copy_tail"]
        shaped = base.copy()
        shaped[warm] = self.distribution_risk_.predict(rows.loc[warm], base[warm])[1]
        pricing_risk = decision_cost(shaped, prices, cutoff)
        adjusted = pricing_risk
        if getattr(self, "retention_adjusted_ranking", False):
            alpha = self.em_veto_.warm_batch_alpha(
                Counter(rows.profile.to_numpy()), getattr(self, "seen_profile_retention", False),
            )
            # Fewer retained rows make the distribution-derived score and quote less reliable.
            if alpha < .8:
                weight = max(0., (alpha - .75) / .05)
                adjusted = (1. - weight) * base + weight * adjusted
        flags, threshold = fixed_volume(adjusted, base >= cutoff, rows.profile.to_numpy())
        return adjusted, prices, flags, threshold, pricing_risk

    def predict_claim_probability(self, features):
        return self.score_batch(features)[0]

    def predict_components(self, features):
        risk, prices, _ = self.decision_components(features)
        return risk, prices

    def decision_components(self, features):
        risk, prices, _, cutoff = self.score_batch(features)
        return risk, prices, cutoff

    def predict(self, features):
        _, prices, flags, _ = self.score_batch(features)
        return np.where(flags, prices, 0.)
