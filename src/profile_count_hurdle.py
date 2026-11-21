"""Estimate whole-profile claim presence and correlated extra claims."""

from lightgbm import LGBMRegressor
import numpy as np
from scipy.stats import betabinom, binom

from src.profile_count_survival import COUNT, PARAMETERS, ProfileCountSurvival


class ProfileCountHurdle(ProfileCountSurvival):
    """Factor observed claim count into occurrence, extra count and dependence."""

    def fit_distribution(self, design, outcome):
        size = design.unseen_rows.to_numpy(dtype=int)
        mass = outcome.sum(axis=1)
        positive = outcome[:, 1:].sum(axis=1)
        self.presence_ = LGBMRegressor(objective="cross_entropy", **PARAMETERS)
        self.presence_.fit(design, positive / mass, sample_weight=mass / size)
        extra = COUNT[1:] - 1
        multi = (size >= 2) & (positive > 1e-8)
        conditional_mean = outcome[multi, 1:] @ extra / positive[multi] / (size[multi] - 1)
        self.mean_ = LGBMRegressor(objective="cross_entropy", **PARAMETERS)
        self.mean_.fit(design.loc[multi], conditional_mean, sample_weight=positive[multi])
        dependent = (size >= 3) & (positive > 1e-8)
        joint = outcome[dependent, 1:] @ (extra * (extra - 1))
        pair_probability = joint / positive[dependent] / ((size[dependent] - 1) * (size[dependent] - 2))
        self.pair_ = LGBMRegressor(objective="cross_entropy", **PARAMETERS)
        self.pair_.fit(design.loc[dependent], pair_probability, sample_weight=positive[dependent])

    def distributions(self, design, sizes):
        claim = self.presence_.predict(design)
        mean = np.clip(self.mean_.predict(design), .0001, .9999)
        pair = self.pair_.predict(design)
        # Match a feasible nonnegative within-profile correlation and shrink
        # it toward binomial draws where there are few extra observations.
        corr = np.clip((pair - mean ** 2) / (mean * (1 - mean)), .005, .8)
        corr = np.where(sizes >= 3, corr, .005)
        concentration = 1 / corr - 1
        extra = COUNT[1:] - 1
        independent = binom.pmf(extra, sizes[:, None] - 1, mean[:, None])
        dependent = betabinom.pmf(extra, sizes[:, None] - 1, (mean * concentration)[:, None],
                                  ((1 - mean) * concentration)[:, None])
        positive = (independent + dependent) / 2
        return np.column_stack([1 - claim, claim[:, None] * positive])
