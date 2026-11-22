"""Learn which payout types remain in fitting-only simulated claim histories."""

from collections import defaultdict

from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np
import pandas as pd

from src.masked_payout_history import history_design, aggregate_history, describe_history, masked_states
from src.dashboard_model import profile_keys
from src.feature_schema import PAYOUT_FACTOR, RANDOM_SEED

NO_CLAIM, REPEAT_WHOLE, RESIDUAL_WHOLE, NEW_WHOLE, REPEAT_SCALED, RECONCILED_SCALED, NEW_SCALED = range(7)
RETENTION = (.82, .95)
PARAMETERS = dict(n_estimators=200, num_leaves=12, min_child_samples=70, learning_rate=.035, reg_lambda=50,
                  random_state=RANDOM_SEED, n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True)


def split_amounts(visible):
    whole = [amount for amount, cent in visible if cent]
    scaled = [amount for amount, cent in visible if not cent]
    total = sum(whole)
    residual = max(0, max(scaled, default=0) - total)
    return whole, scaled, total, residual


def classify_amount(signature, visible, total, residual):
    amount, cent = signature
    if signature in visible:
        return REPEAT_WHOLE if cent else REPEAT_SCALED
    if cent:
        return RESIDUAL_WHOLE if amount == residual else NEW_WHOLE
    return RECONCILED_SCALED if amount == total else NEW_SCALED


def composition_inputs(zero, visible, roster_size):
    whole, scaled, total, residual = split_amounts(visible)
    history = describe_history(zero, visible, roster_size)
    signature = np.array([total, residual, min(whole, default=0), min(scaled, default=0)], dtype=float)
    return (*history, *np.log1p(signature / 100.), residual / max(1, max(scaled, default=0)),
            len(whole) > 0, len(scaled) > 0, residual > 0)


def masked_examples(zero, paid, roster_size, retentions=RETENTION):
    for retention in retentions:
        for past_zero, past_paid, negatives, positives in masked_states(zero, paid, retention):
            if not past_paid or negatives + positives <= 1e-7:
                continue
            _, _, total, residual = split_amounts(past_paid)
            missing = {signature: count - past_paid.get(signature, 0) for signature, count in paid.items()}
            counts = np.zeros(7)
            counts[NO_CLAIM] = negatives
            if positives > 0:
                weight = positives / sum(missing.values())
                for signature, frequency in missing.items():
                    counts[classify_amount(signature, past_paid, total, residual)] += frequency * weight
            yield composition_inputs(past_zero, past_paid, roster_size), counts / len(retentions)


COMPOSITION = ("log_whole_total", "log_component_residual", "log_smallest_whole", "log_smallest_scaled",
               "relative_residual", "can_repeat_whole", "can_repeat_scaled", "can_complete_residual")


class MaskedPayoutComposition:
    """Compare a binary objective with recovery of seven mutually exclusive outcomes."""

    @staticmethod
    def design(features, histories):
        panel = np.asarray(histories)
        inputs = history_design(features, panel[:, :-len(COMPOSITION)])
        return inputs.join(pd.DataFrame(panel[:, -len(COMPOSITION):], columns=COMPOSITION))

    def fit(self, features, amounts, roster, *, coverage=None):
        keys, self.zero_, self.paid_ = aggregate_history(features, amounts)
        self.roster_, self.coverage_ = roster, coverage
        positions, inputs, weights = [], [], []
        for index in np.flatnonzero(~pd.Index(keys).duplicated()):
            key = keys[index]
            if key not in self.paid_:
                continue
            for design, count in self.examples(self.zero_[key], self.paid_[key], roster.loc[key]):
                positions.append(index)
                inputs.append(design)
                weights.append(count)
        design = self.add_coverage(self.design(features.iloc[positions], inputs), keys[positions])
        counts = np.asarray(weights)
        totals = counts.sum(axis=1)
        self.binary_ = LGBMRegressor(objective="cross_entropy", **PARAMETERS)
        self.binary_.fit(design, 1. - counts[:, NO_CLAIM] / totals, sample_weight=totals)
        examples, outcome = np.nonzero(counts > 1e-8)
        self.composition_ = LGBMClassifier(objective="multiclass", **PARAMETERS)
        self.composition_.fit(design.iloc[examples], outcome, sample_weight=counts[examples, outcome])
        self.amounts_ = self.fitted_amounts(self.paid_)
        self.calibration_profiles_ = len(np.unique(positions))
        return self

    @staticmethod
    def examples(zero, paid, roster_size):
        return masked_examples(zero, paid, roster_size)

    def add_coverage(self, design, keys):
        if self.coverage_ is not None:
            return design.join(self.coverage_.reindex(keys).reset_index(drop=True))
        return design

    @staticmethod
    def fitted_amounts(history):
        payments = defaultdict(list)
        for profile in history.values():
            for base, whole in profile:
                payments[whole].append(base * (1. if whole else PAYOUT_FACTOR) / 100.)
        return {whole: np.quantile(values, (.04, .12, .28, .50, .75, .90))
                for whole, values in payments.items()}

    def predict(self, features, fallback):
        keys = profile_keys(features)
        unique, location, repeat = np.unique(keys, return_index=True, return_inverse=True)
        warm = np.fromiter((key in self.paid_ for key in unique), dtype=bool)
        if not warm.any():
            return np.asarray(fallback).copy(), np.asarray(fallback).copy(), warm[repeat], {}
        selected = unique[warm]
        design = self.design(features.iloc[location[warm]], [
            composition_inputs(self.zero_[key], self.paid_[key], self.roster_.loc[key]) for key in selected
        ])
        design = self.add_coverage(design, selected)
        probabilities = np.zeros((len(selected), 7))
        probabilities[:, self.composition_.classes_] = self.composition_.predict_proba(design)
        probabilities[:, REPEAT_WHOLE] *= design.can_repeat_whole
        probabilities[:, RESIDUAL_WHOLE] *= design.can_complete_residual
        probabilities[:, REPEAT_SCALED] *= design.can_repeat_scaled
        probabilities[:, RECONCILED_SCALED] *= design.can_repeat_whole
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        direct, typed = np.asarray(fallback)[location].copy(), np.asarray(fallback)[location].copy()
        direct[warm] = self.binary_.predict(design)
        typed[warm] = 1 - probabilities[:, NO_CLAIM]
        panels = dict(zip(selected, probabilities))
        return direct[repeat], typed[repeat], warm[repeat], panels

    def quote(self, profile, probabilities, floor, cap):
        whole, scaled, total, residual = split_amounts(self.paid_[profile])
        p = probabilities.copy()
        if (total, False) in self.paid_[profile]:
            # A sum already observed belongs to the repeat-scaled mark.
            p[REPEAT_SCALED] += p[RECONCILED_SCALED]
            p[RECONCILED_SCALED] = 0.
        if p[NO_CLAIM] >= .5:
            return floor
        values, weights = [0.], [p[NO_CLAIM]]
        offers = {REPEAT_WHOLE: np.array(whole) / 100., RESIDUAL_WHOLE: [residual / 100.],
                  REPEAT_SCALED: np.array(scaled) * PAYOUT_FACTOR / 100.,
                  RECONCILED_SCALED: [total * PAYOUT_FACTOR / 100.],
                  NEW_WHOLE: self.amounts_.get(True, [floor]), NEW_SCALED: self.amounts_.get(False, [floor])}
        for mark, candidates in offers.items():
            if len(candidates):
                values.extend(candidates)
                weights.extend([p[mark] / len(candidates)] * len(candidates))
        order = np.argsort(values)
        mass = np.cumsum(np.asarray(weights)[order])
        median = min(int(np.searchsorted(mass, .5 * mass[-1])), len(order) - 1)
        return float(np.clip(values[order[median]], floor, cap))

    def supported_quote(self, profile, quote):
        whole, scaled, total, residual = split_amounts(self.paid_[profile])
        prices = np.r_[whole, np.array(scaled) * PAYOUT_FACTOR, residual, total * PAYOUT_FACTOR] / 100.
        return bool(np.any(np.isclose(prices, quote, rtol=0., atol=.005)))
