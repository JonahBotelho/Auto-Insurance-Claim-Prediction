"""Stack exact-profile count and payout-choice models using out-of-fold scores."""

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.claim_count_mixture import ClaimCountMixture
from src.claim_payout_choices import PayoutChoices
from src.feature_schema import PROFILE_HISTORY_COLUMNS, RANDOM_SEED

SCORE_COLUMNS = ("hist_oof", "payout_choices", "count_single", "count_batch")
PROFILE_STACK_CUTOFF = .25


def logit_scores(rows):
    return logit(np.clip(rows.loc[:, SCORE_COLUMNS].to_numpy(), .005, .995))


def history_inputs(rows, *, batch=True, interactions=False):
    scores = logit_scores(rows)
    if not batch:
        scores = scores[:, :-1]
    counts = np.column_stack([np.log1p(rows.prior_positives), np.log1p(rows.prior_zeros)])
    # A repeated payout and a positive count do not have the same implications
    # when a fitting history already contains several positive rows.
    if interactions:
        return np.column_stack([scores, counts, scores[:, :2] * counts[:, :1]])
    return np.column_stack([scores, counts])


def calibration_weights(rows):
    center = (rows.hist_oof.to_numpy() + rows.payout_choices.to_numpy()) / 2
    return .25 + np.exp(-((center - PROFILE_STACK_CUTOFF) / .2) ** 2)


def fit_history_stack(inputs, target, weight):
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=500))
    return model.fit(inputs, target, logisticregression__sample_weight=weight)


class ProfileHistoryEnsemble(RegressorMixin, BaseEstimator):
    def __init__(self, baseline_model, claim_probability_threshold=PROFILE_STACK_CUTOFF, n_splits=5,
                 random_seed=RANDOM_SEED):
        self.baseline_model = baseline_model
        self.claim_probability_threshold = claim_probability_threshold
        self.n_splits = n_splits
        self.random_seed = random_seed

    def fit(self, features, amounts):
        amounts = np.asarray(amounts)
        repetitions = getattr(self, "history_crossfit_repeats", 1)
        folds = [panel for repeat in range(repetitions)
                 for panel in self._calibration_folds(features, amounts, repeat)]
        self._fit_calibrators(pd.concat(folds, ignore_index=repetitions > 1))
        self.components_ = self._fit_components(features, amounts, price=True)
        self.n_features_in_ = features.shape[1]
        return self

    def _calibration_folds(self, features, amounts, repeat):
        claims = amounts > 0
        splitter = StratifiedKFold(self.n_splits, shuffle=True, random_state=self.random_seed + repeat)
        first_fold = self.n_splits * repeat + 1
        for fold, (training, validation) in enumerate(splitter.split(features, claims), first_fold):
            fitted = self._fit_components(features.iloc[training], amounts[training], price=False)
            scores, _ = self._scores(fitted, features.iloc[validation])
            yield scores.assign(claim=claims[validation], actual=amounts[validation], fold=fold)

    def _fit_calibrators(self, data):
        known = data.loc[data.prior_positives.gt(0)]
        self.history_stack_ = None
        if known.claim.nunique() == 2:
            self.history_stack_ = fit_history_stack(self.stack_inputs(known), known.claim, self.stack_weights(known))

    def _fit_components(self, features, amounts, *, price):
        baseline = clone(self.baseline_model)
        if not price:
            # The first stage for stack fitting consumes probabilities and counts.
            baseline.set_params(aggregate_payout_pricing=False, median_payout_pricing=False)
        fitted = baseline.fit(features, amounts)
        choices = PayoutChoices().fit(features, amounts)
        count = ClaimCountMixture().fit(features, amounts)
        return fitted, choices, count

    @staticmethod
    def _scores(components, features):
        baseline, choices, count = components
        probability, price = baseline.predict_components(features)
        choice, _ = choices.predict(features, probability, price)
        single, batch = count.predict(features, probability)
        row_count, positive_count, _ = PROFILE_HISTORY_COLUMNS
        history = baseline.history_.transform(features)
        return pd.DataFrame({
            "hist_oof": probability, "payout_choices": choice, "count_single": single, "count_batch": batch,
            "prior_positives": history[positive_count], "prior_zeros": history[row_count] - history[positive_count],
        }), price

    def predict_components(self, features):
        scores, prices = self._scores(self.components_, features)
        return self._stack_probability(scores), prices

    def _stack_probability(self, scores):
        probability = scores.hist_oof.to_numpy(copy=True)
        known = scores.prior_positives.gt(0)
        if self.history_stack_ is not None and known.any():
            probability[known] = self.history_stack_.predict_proba(self.stack_inputs(scores.loc[known]))[:, 1]
        return probability

    stack_inputs = staticmethod(history_inputs)
    stack_weights = staticmethod(calibration_weights)

    def predict_claim_probability(self, features):
        return self.predict_components(features)[0]

    def predict(self, features):
        probability, prices = self.predict_components(features)
        return np.where(probability >= self.claim_probability_threshold, prices, 0.)
