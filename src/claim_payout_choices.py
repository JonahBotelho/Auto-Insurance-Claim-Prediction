"""Joint detection and severity for profiles with a positive payout history."""

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from src.dashboard_model import profile_keys
from src.feature_schema import PAYOUT_FACTOR
from src.payout_pricing import MEDIAN_PRICING_CAP, payout_signature

KINDS = ("new_cent", "new_scaled", "repeat_cent", "repeat_scaled")
PENALTY = 5.
REFERENCE_POINTS = (np.arange(64) + .5) / 64


@dataclass
class Alternatives:
    """Competing outcomes in contiguous groups with one group per input row."""

    positions: np.ndarray
    starts: np.ndarray
    row_index: np.ndarray
    evidence: np.ndarray
    amount: np.ndarray
    kind: np.ndarray
    selected: np.ndarray

    def probabilities(self, coefficients):
        utility = self.evidence @ coefficients
        center = np.maximum.reduceat(utility, self.starts)
        numerator = np.exp(utility - center[self.row_index])
        denominator = np.add.reduceat(numerator, self.starts)
        return numerator / denominator[self.row_index], utility, center + np.log(denominator)


class PayoutChoices:
    """Estimate a payout distribution by normalizing odds across eligible outcomes."""

    evidence_size = 7

    def fit(self, features, amounts):
        actual = np.asarray(amounts)
        self.positive_ = defaultdict(Counter)
        self.zeros_ = Counter()
        self.profiles_by_payout_ = defaultdict(set)
        for key, amount in zip(profile_keys(features), actual):
            if amount > 0:
                outcome = payout_signature(amount)
                self.positive_[key][outcome] += 1
                self.profiles_by_payout_[outcome[0]].add(key)
            else:
                self.zeros_[key] += 1
        self.fit_candidate_support()
        alternatives = self.alternatives(features, actual)
        self.coefficients_ = None
        if not len(alternatives.starts):
            return self
        self.coefficients_ = self._fit_choices(alternatives)
        self.fallback_payouts_ = {}
        labels = alternatives.kind[alternatives.selected]
        known_actual = actual[alternatives.positions]
        for kind in ("new_cent", "new_scaled"):
            novel = known_actual[labels == kind]
            self.fallback_payouts_[kind] = np.quantile(novel if len(novel) else actual[actual > 0], REFERENCE_POINTS)
        self.floor_ = float(np.quantile(actual[actual > 0], .05))
        return self

    def fit_candidate_support(self):
        """Fit additional payout evidence when a candidate model supplies it."""

    @staticmethod
    def _fit_choices(alternatives):
        def objective(coefficients):
            probability, utility, log_normalizer = alternatives.probabilities(coefficients)
            errors = probability.copy()
            errors[alternatives.selected] -= 1
            loss = (log_normalizer - utility[alternatives.selected]).sum()
            loss += .5 * PENALTY * np.dot(coefficients, coefficients)
            gradient = alternatives.evidence.T @ errors + PENALTY * coefficients
            return loss, gradient

        result = minimize(objective, np.zeros(alternatives.evidence.shape[1]), method="L-BFGS-B", jac=True)
        if not result.success:
            raise RuntimeError(result.message)
        return result.x

    def _payout_candidates(self, key, counts):
        for base in sorted({base for base, _ in counts}):
            for cent, kind in ((True, "repeat_cent"), (False, "repeat_scaled")):
                support = counts.get((base, cent), 0)
                paired = counts.get((base, not cent), 0)
                other = len(self.profiles_by_payout_[base]) - 1
                # Repeated variants in the fitting data are usually capped. Let the
                # likelihood estimate saturation with a separate indicator for two or more.
                evidence = (np.log1p(base / 100) - 7, support == 0, support >= 2,
                            paired == 0, paired >= 2, np.log1p(other), base % 100 == 0)
                yield kind, base / 100 if cent else PAYOUT_FACTOR * base / 100, evidence, (base, cent)

    @staticmethod
    def _group_features(exposure, rating, zeros, counts):
        cent = sum(count for (_, flag), count in counts.items() if flag)
        fractional = sum(counts.values()) - cent
        distinct = len({base for base, _ in counts})
        grid_error = abs(12 * exposure - round(12 * exposure))
        return (1., exposure - .5, (exposure - .5) ** 2, rating / 8, rating >= 7, grid_error <= .007,
                np.log1p(zeros), np.log1p(cent), np.log1p(fractional), np.log1p(distinct))

    def alternatives(self, rows, own_amounts=None):
        amounts = np.asarray(own_amounts) if own_amounts is not None else np.full(len(rows), -1.)
        keys = profile_keys(rows)
        positions, choices, row_index, starts = [], [], [], []
        selected, amounts_by_choice, kinds = [], [], []
        for position, (key, exposure, rating, own) in enumerate(zip(keys, rows.feature1, rows.feature3, amounts)):
            counts = self.positive_.get(key, {}).copy()
            if own > 0:
                counts[payout_signature(own)] -= 1
            counts = {outcome: count for outcome, count in counts.items() if count}
            if not counts:
                continue
            zeros = self.zeros_[key] - int(own == 0)
            context = self._group_features(exposure, rating, zeros, counts)
            starts.append(len(choices))
            positions.append(position)
            empty_evidence = (0.,) * self.evidence_size
            candidates = [("zero", 0., empty_evidence, None), ("new_cent", np.nan, empty_evidence, None),
                          ("new_scaled", np.nan, empty_evidence, None), *self._payout_candidates(key, counts)]
            observed = payout_signature(own) if own > 0 else None
            candidate_present = observed is not None and any(outcome == observed for _, _, _, outcome in candidates)
            for kind, amount, evidence, outcome in candidates:
                row_index.append(len(positions) - 1)
                amounts_by_choice.append(amount)
                kinds.append(kind)
                choices.append(self._choice_features(kind, context, evidence))
                if own >= 0 and self._was_observed(own, observed, candidate_present, kind, outcome):
                    selected.append(len(choices) - 1)
        if own_amounts is not None and len(selected) != len(starts):
            raise ValueError("Each fitting row requires one observed outcome.")
        return Alternatives(np.asarray(positions, dtype=int), np.asarray(starts, dtype=int),
                            np.asarray(row_index, dtype=int), np.asarray(choices), np.asarray(amounts_by_choice),
                            np.asarray(kinds), np.asarray(selected, dtype=int))

    @staticmethod
    def _was_observed(own, observed, candidate_present, kind, outcome):
        if own == 0:
            return kind == "zero"
        if candidate_present:
            return outcome == observed
        return kind == ("new_cent" if observed[1] else "new_scaled")

    @staticmethod
    def _choice_features(kind, context, evidence):
        values = np.zeros((len(KINDS), len(context) + len(evidence)))
        if kind != "zero":
            values[KINDS.index(kind)] = (*context, *evidence)
        return values.ravel()

    def predict(self, rows, baseline_probability, baseline_price):
        probability = np.array(baseline_probability, copy=True)
        prices = np.array(baseline_price, dtype=float, copy=True)
        if self.coefficients_ is None:
            return probability, prices
        alternatives = self.alternatives(rows)
        if len(alternatives.starts):
            weights, _, _ = alternatives.probabilities(self.coefficients_)
            probability[alternatives.positions] = 1 - weights[alternatives.starts]
            priced = self._median_amounts(alternatives, weights)
            positions = alternatives.positions
            prices[positions] = np.where(prices[positions] > 0, priced, 0.)
        return probability, prices

    def _median_amounts(self, alternatives, weights):
        prices = []
        end = np.r_[alternatives.starts[1:], len(weights)]
        for start, stop in zip(alternatives.starts, end):
            amounts = list(alternatives.amount[start:stop])
            mass = list(weights[start:stop])
            for index, kind in enumerate(alternatives.kind[start:stop]):
                if kind in self.fallback_payouts_:
                    novel = self.fallback_payouts_[kind]
                    amounts[index] = novel[0]
                    mass[index] /= len(novel)
                    amounts.extend(novel[1:])
                    mass.extend(np.repeat(mass[index], len(novel) - 1))
            order = np.argsort(amounts)
            cumulative = np.cumsum(np.asarray(mass)[order])
            prices.append(amounts[order[np.searchsorted(cumulative, .5)]])
        return np.clip(prices, min(self.floor_, MEDIAN_PRICING_CAP), MEDIAN_PRICING_CAP)
