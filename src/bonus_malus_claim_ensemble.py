"""Deployable pairwise and count-confirmed claim ensembles with Bonus-Malus timeline detection."""

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from src.bonus_malus_timeline import BonusMalusTimelineDetector
from src.bonus_malus_timeline_v3 import BonusMalusTimelineV3Detector
from src.claim_benchmark_pipeline import supported_admission
from src.claim_count_mixture import MAX_ROWS
from src.claim_history_leaf_retrieval import ClaimHistoryLeafRetrieval
from src.claim_probability import log_odds
from src.clean_warm_calibrator import CleanWarmCalibrator
from src.dashboard_model import profile_keys
from src.exact_em_deconvolution import EMReconciledVeto
from src.feature_schema import RANDOM_SEED
from src.pairwise_payout_ensemble import PairwisePayoutEnsemble
from src.profile_count_hurdle import ProfileCountHurdle
from src.profile_count_survival import COUNT, ProfileCountSurvival
from src.template_claim_ensemble import expected_f1_cutoff


def _align_risk_with_flags(risk, flags, cutoff):
    """Keep decision_components() consistent with flags so threshold CV matches predict()."""
    if not np.isfinite(cutoff):
        return risk
    above = np.nextafter(cutoff, np.inf)
    below = np.nextafter(cutoff, -np.inf)
    aligned = np.where(flags & (risk < cutoff), np.maximum(risk, above), risk)
    return np.where(~flags & (aligned >= cutoff), np.minimum(aligned, below), aligned)


class PairwiseTimelineClaimEnsemble(PairwisePayoutEnsemble):
    """Pairwise payout reranker augmented with French Bonus-Malus reference-window cold detection."""

    def fit(self, features, amounts):
        with threadpool_limits(limits=2):
            super().fit(features, amounts)
            self.timeline_ = BonusMalusTimelineDetector(random_seed=self.random_seed).fit(features, amounts)
        return self

    def _score_batch(self, features):
        risk, prices, flags, cutoff = super()._score_batch(features)
        if not len(features):
            return risk, prices, flags, cutoff
        risk, prices, flags = self.timeline_.augment_predictions(features, risk, prices, flags, cutoff)
        return _align_risk_with_flags(risk, flags, cutoff), prices, flags, cutoff


class CountRetrievalTimelineClaimEnsemble(PairwiseTimelineClaimEnsemble):
    """Pairwise ensemble with count-hurdle agreement, leaf-retrieval admissions, and BM timeline detection."""

    def fit(self, features, amounts):
        with threadpool_limits(limits=2):
            super().fit(features, amounts)
            self.count_survival_ = ProfileCountSurvival().fit(features, amounts)
            self.count_hurdle_ = ProfileCountHurdle().fit(features, amounts)
            self.leaf_ranked_ = ClaimHistoryLeafRetrieval(ranking=True).fit(features, amounts)
            self.leaf_regular_ = ClaimHistoryLeafRetrieval().fit(features, amounts)
        return self

    def _refine_warm_flags(self, features, anchor, prices, flags, cutoff):
        """Apply count-survival/hurdle agreement and dual leaf-retrieval admissions on warm profiles."""
        _, inverse, peers = np.unique(profile_keys(features), return_inverse=True, return_counts=True)
        point, continuation, warm = self.count_survival_.predict(features, anchor)
        if not warm.any():
            return prices, flags
        _, hurdle, _ = self.count_hurdle_.predict(features, anchor)
        sizes = np.minimum(peers, MAX_ROWS)
        baseline, center = log_odds(point), log_odds(anchor)
        decisions = []
        for law in (continuation, hurdle):
            rate = ((law @ COUNT) / sizes)[inverse]
            shift = np.clip(log_odds(rate) - baseline, -0.6, 0.6)
            decisions.append(expit(center + shift / 2.0) >= cutoff)
        first, second = decisions
        near = warm & (np.abs(anchor - cutoff) <= 0.075)
        counted = np.where(near & (first == second), second, flags)
        ranked, _ = self.leaf_ranked_.predict(features)
        regular, _ = self.leaf_regular_.predict(features)
        admitted = ~counted & supported_admission(ranked, near) & supported_admission(regular, near)
        updated_flags = counted | admitted
        # Quote newly admitted warm profiles using fitted source-tail support and local tail caps.
        added_warm = updated_flags & ~flags
        if added_warm.any():
            addition_quotes = self.tail_.support(features)[1]
            tail_caps = self.timeline_.tail_.predict(features)
            warm_quotes = np.where(admitted, np.maximum(0.8 * addition_quotes, np.minimum(addition_quotes, tail_caps)), addition_quotes)
            prices = np.where(added_warm, warm_quotes, prices)
        return prices, updated_flags

    def _score_batch(self, features):
        # Run PairwisePayoutEnsemble first, then warm count+leaf refinement, then cold BM-window detection.
        risk, prices, flags, cutoff = PairwisePayoutEnsemble._score_batch(self, features)
        if not len(features):
            return risk, prices, flags, cutoff
        prices, flags = self._refine_warm_flags(features, risk, prices, flags, cutoff)
        risk, prices, flags = self.timeline_.augment_predictions(features, risk, prices, flags, cutoff)
        return _align_risk_with_flags(risk, flags, cutoff), prices, flags, cutoff


class EMTimelineClaimEnsemble(CountRetrievalTimelineClaimEnsemble):
    """Count-retrieval ensemble with EM reconciled veto, BM timeline v3, and calibrated L1 pricing."""

    def __init__(self, baseline_model, claim_probability_threshold=0.25, n_splits=5, random_seed=RANDOM_SEED,
                 calibrated_l1_pricing=True, low_confidence_cutoff=0.48, low_confidence_floor=1.0,
                 em_support_min=0.003, clean_weight=0.50, default_alpha=0.80):
        super().__init__(baseline_model, claim_probability_threshold=claim_probability_threshold,
                         n_splits=n_splits, random_seed=random_seed)
        self.calibrated_l1_pricing = calibrated_l1_pricing
        self.low_confidence_cutoff = low_confidence_cutoff
        self.low_confidence_floor = low_confidence_floor
        self.em_support_min = em_support_min
        self.clean_weight = clean_weight
        self.default_alpha = default_alpha

    def fit(self, features, amounts):
        with threadpool_limits(limits=2):
            PairwisePayoutEnsemble.fit(self, features, amounts)
            self.timeline_ = BonusMalusTimelineV3Detector(
                random_seed=self.random_seed, alpha=self.default_alpha,
                low_confidence_floor=self.low_confidence_floor,
            ).fit(features, amounts)
            self.count_survival_ = ProfileCountSurvival().fit(features, amounts)
            self.count_hurdle_ = ProfileCountHurdle().fit(features, amounts)
            self.leaf_ranked_ = ClaimHistoryLeafRetrieval(ranking=True).fit(features, amounts)
            self.leaf_regular_ = ClaimHistoryLeafRetrieval().fit(features, amounts)
            self.em_veto_ = EMReconciledVeto(default_alpha=self.default_alpha).fit(features, amounts)
            self.warm_calibrator_ = (
                CleanWarmCalibrator(alpha=self.default_alpha, random_seed=self.random_seed).fit(features, amounts)
                if self.calibrated_l1_pricing else None
            )
        return self

    def _score_batch(self, features):
        anchor, prices, flags, base_cutoff = PairwisePayoutEnsemble._score_batch(self, features)
        if not len(features):
            return anchor, prices, flags, base_cutoff
        prices, flags = self._refine_warm_flags(features, anchor, prices, flags, base_cutoff)
        warm, is_reconciled, p_em, em_mass = self.em_veto_.predict(features)
        cand_cold, cold_risk = self.timeline_.predict_cold_risk(features)

        if warm.any() and len(features) * self.prevalence_ >= 10:
            unified = np.where(warm, anchor, np.where(cand_cold, cold_risk, 0.0))
            cutoff = float(np.clip(expected_f1_cutoff(unified, self.prevalence_), 0.12, 0.275))
        else:
            cutoff = float(np.clip(base_cutoff, 0.12, 0.265))

        vetoed = warm & is_reconciled & (em_mass >= self.em_support_min) & (p_em < cutoff)
        flags = flags & ~vetoed
        added_cold = cand_cold & (cold_risk >= cutoff) & ~flags
        if added_cold.any():
            cold_quotes = self.timeline_.quote_cold_claims(features, cold_risk)
            prices = np.where(added_cold, cold_quotes, prices)
        flags = flags | added_cold

        if self.calibrated_l1_pricing and self.warm_calibrator_ is not None:
            p_clean = self.warm_calibrator_.predict(features)
            cold_prob = np.where(cand_cold, np.maximum(anchor, cold_risk), anchor)
            p_l1 = np.where(warm, (1.0 - self.clean_weight) * anchor + self.clean_weight * p_clean, cold_prob)
            prices = np.where(p_l1 < self.low_confidence_cutoff, self.low_confidence_floor, prices)

        risk = np.where(cand_cold, np.maximum(anchor, cold_risk), anchor)
        return _align_risk_with_flags(risk, flags, cutoff), prices, flags, cutoff
