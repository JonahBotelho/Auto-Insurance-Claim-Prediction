"""Retrieve independently labeled analogues in learned payout-history leaf space."""

from dataclasses import dataclass

from lightgbm import LGBMRanker, LGBMRegressor
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.special import betainc
from sklearn.model_selection import KFold

from src.profile_count_survival import COUNT, masked_scoring_examples
from src.dashboard_model import profile_keys
from src.feature_schema import RANDOM_SEED
from src.masked_payout_history import HISTORY_COLUMNS, aggregate_history, describe_history, history_design

TREE_COUNT = 96
NEIGHBORS = 100
SCORING_PEERS = HISTORY_COLUMNS.index("unseen_rows")


def encode_leaves(leaves, stride):
    nrows, ntrees = leaves.shape
    columns = (leaves + stride * np.arange(ntrees)).ravel()
    positions = np.arange(0, nrows * ntrees + 1, ntrees)
    return csr_matrix((np.ones(len(columns), dtype=np.float32), columns, positions), shape=(nrows, ntrees * stride))


def fitting_cases(features, targets, *, zero_only=False):
    keys, zeros, paid = aggregate_history(features, targets)
    positions, labels, histories, weights = [], [], [], []
    for index in np.flatnonzero(~pd.Index(keys).duplicated()):
        key = keys[index]
        if (zero_only and not zeros[key]) or (not zero_only and key not in paid):
            continue
        for state, distribution in masked_scoring_examples(zeros[key], paid.get(key, {}), zero_only=zero_only):
            weight = distribution.sum()
            positions.append(index)
            histories.append(state)
            labels.append(np.dot(distribution, COUNT) / (weight * state[SCORING_PEERS]))
            weights.append(weight)
    panel = history_design(features.iloc[positions], histories)
    return panel, np.asarray(labels), np.asarray(weights), keys[positions], zeros, paid


def regime(design):
    whole = design.past_whole.to_numpy() > 0.
    scaled = design.past_scaled.to_numpy() > 0.
    proof = design.reconciled_total.to_numpy() > 0.
    return whole.astype(int) + 2 * scaled + 4 * proof


@dataclass
class LeafDonors:
    fitted: object
    probability_model: object
    trees: csr_matrix
    keys: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    regime: np.ndarray
    stride: int

    def evidence(self, design, keys):
        leaves = self.fitted.predict(design, pred_leaf=True)
        encoded = encode_leaves(leaves, self.stride)
        query_regime = regime(design)
        risk = self.probability_model.predict(design)
        for start in range(0, len(design), 80):
            block = slice(start, start + 80)
            similarity = (encoded[block] @ self.trees.T).toarray() / leaves.shape[1]
            for index, distances in enumerate(similarity, start):
                masks = (self.keys != keys[index]) & (self.regime == query_regime[index]) & (distances >= .20)
                found = np.flatnonzero(masks)
                if len(found) > NEIGHBORS:
                    ranked = np.lexsort((found, -distances[found]))
                    found = found[ranked[:NEIGHBORS]]
                mass = distances[found] ** 4 * self.weights[found]
                yield index, self.keys[found], mass, self.labels[found], risk[index]


class ClaimHistoryLeafRetrieval:
    """Learn similarity from one set of profiles and estimate local risk on another."""

    def __init__(self, *, ranking=False, zero_only=False):
        self.ranking, self.zero_only = ranking, zero_only

    def learn_geometry(self, design, labels, weights, fold):
        configuration = dict(n_estimators=TREE_COUNT, num_leaves=14, min_child_samples=45,
                             learning_rate=.045, reg_lambda=22., random_state=RANDOM_SEED + fold,
                             n_jobs=2, verbosity=-1, deterministic=True, force_col_wise=True)
        probability = LGBMRegressor(objective="cross_entropy", **configuration).fit(design, labels, sample_weight=weights)
        if not self.ranking:
            return probability, probability
        # Compare positive and negative masked outcomes within the same payout
        # regime and scoring group size. Use the exact mask weights on each outcome.
        supported = np.column_stack([(1. - labels) * weights, labels * weights])
        indices, observed = np.nonzero(supported > 1e-8)
        groups = 2 * regime(design)[indices] + (design.unseen_rows.to_numpy()[indices] > 1)
        if self.zero_only:
            exposure_band = np.minimum(9, (10 * design.feature1.to_numpy()[indices]).astype(int))
            vehicle_band = np.searchsorted([3, 6], design.feature3.to_numpy()[indices])
            groups = 3 * (10 * groups + exposure_band) + vehicle_band
        order = np.argsort(groups, kind="stable")
        _, count = np.unique(groups, return_counts=True)
        ranker = LGBMRanker(objective="lambdarank", lambdarank_truncation_level=40, **configuration)
        ranker.fit(design.iloc[indices[order]], observed[order], group=count,
                   sample_weight=supported[indices[order], observed[order]])
        return ranker, probability

    def fit(self, features, target, *, metadata=None):
        cases, labels, weights, keys, self.zeros_, self.paid_ = fitting_cases(features, target, zero_only=self.zero_only)
        self.metadata_ = metadata
        cases = self.augment(cases, keys)
        profiles = np.unique(keys)
        self.donors_ = []
        folds = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
        for fold, (learn, reserve) in enumerate(folds.split(profiles)):
            train, donor = np.isin(keys, profiles[learn]), np.isin(keys, profiles[reserve])
            forest, probability = self.learn_geometry(cases.loc[train], labels[train], weights[train], fold)
            leaves = forest.predict(cases.loc[donor], pred_leaf=True)
            stride = forest.num_leaves
            self.donors_.append(LeafDonors(forest, probability, encode_leaves(leaves, stride), keys[donor], labels[donor],
                                          weights[donor], regime(cases.loc[donor]), stride))
        return self

    def augment(self, cases, keys):
        if self.metadata_ is None:
            return cases
        return cases.join(self.metadata_.reindex(keys).reset_index(drop=True))

    def predict(self, query):
        keys, first, inverse, sizes = np.unique(profile_keys(query), return_index=True, return_inverse=True, return_counts=True)
        paid = np.fromiter((key in self.paid_ for key in keys), dtype=bool)
        zero = np.array([self.zeros_[key] for key in keys])
        eligible = (~paid & (zero > 0)) if self.zero_only else paid
        if not eligible.any():
            return np.full((len(query), 4), np.nan), eligible[inverse]
        histories = []
        for key, seen_zero, size in zip(keys[eligible], zero[eligible], sizes[eligible], strict=True):
            observed = self.paid_.get(key, {})
            histories.append(describe_history(seen_zero, observed, seen_zero + sum(observed.values()) + size))
        design = history_design(query.iloc[first[eligible]], histories)
        design = self.augment(design, keys[eligible])
        evidence = list(zip(*(bank.evidence(design, keys[eligible]) for bank in self.donors_), strict=True))
        output = np.full((len(keys), 4), np.nan)
        output[eligible] = np.array([self.posterior(case) for case in evidence])
        return output[inverse], eligible[inverse]

    @staticmethod
    def posterior(case):
        _, donors, weight, labels, risks = zip(*case, strict=True)
        keys, weights, labels = map(np.concatenate, (donors, weight, labels))
        _, inverse = np.unique(keys, return_inverse=True)
        clustered = np.bincount(inverse, weights=weights)
        total = weights.sum()
        effective = total ** 2 / max(1e-12, np.dot(clustered, clustered))
        chance = np.dot(weights, labels) / max(total, 1e-12)
        prior = np.clip(np.mean(risks), .005, .995)
        alpha, beta = 12 * prior + effective * chance, 12 * (1 - prior) + effective * (1 - chance)
        below = betainc(alpha, beta, .275)
        return alpha / (alpha + beta), effective, 1. - below, prior
