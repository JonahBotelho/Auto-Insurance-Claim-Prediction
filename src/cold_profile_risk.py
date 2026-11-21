"""Score policies from previously unseen profiles using only rating features."""

import numpy as np

from src.claim_rating_features import CATEGORIES, actuarial_features
from src.dashboard_model import profile_keys
from src.feature_schema import RANDOM_SEED


def cold_profile_inputs(features):
    raw = actuarial_features(features, np.zeros(len(features))).drop(columns="zero_history")
    raw.index = profile_keys(features)
    raw["profile_size"] = raw.index.map(raw.index.value_counts()).astype(float)
    return raw


def build_cold_classifier(random_seed=RANDOM_SEED):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=270, depth=4, learning_rate=.045, l2_leaf_reg=45, random_seed=random_seed,
        thread_count=2, verbose=False, allow_writing_files=False, cat_features=list(CATEGORIES),
    )


class ColdProfileRisk:
    def __init__(self, random_seed=RANDOM_SEED):
        self.random_seed = random_seed

    def fit(self, features, amounts):
        self.estimator_ = build_cold_classifier(self.random_seed).fit(cold_profile_inputs(features), np.asarray(amounts) > 0)
        return self

    def predict(self, features):
        return self.estimator_.predict_proba(cold_profile_inputs(features))[:, 1]
