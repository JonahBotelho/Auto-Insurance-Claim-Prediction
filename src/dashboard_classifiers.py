"""Classifier factories and encodings matching the saved dashboard configurations."""

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier

from src.feature_schema import CATEGORICAL_FEATURES, HIST_GRADIENT_BOOSTING_PARAMS, RANDOM_SEED


def build_dashboard_classifier(settings, seed=RANDOM_SEED):
    common = dict(random_state=seed)
    family = settings["family"]
    if family == "hist":
        return HistGradientBoostingClassifier(
            max_iter=settings["iterations"], learning_rate=settings["rate"],
            max_leaf_nodes=settings["leaves"], min_samples_leaf=settings["leaf"],
            l2_regularization=settings["l2"], early_stopping=settings.get("early_stopping", False),
            categorical_features=HIST_GRADIENT_BOOSTING_PARAMS["categorical_features"], **common)
    if family == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=settings["iterations"], learning_rate=settings["rate"],
                              num_leaves=2 ** settings["depth"], min_child_samples=settings.get("leaf", 50),
                              reg_lambda=settings["l2"], verbosity=-1, n_jobs=4, **common)
    if family in ["cat", "cat_native"]:
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=settings["iterations"], learning_rate=settings["rate"],
                                  depth=settings["depth"], l2_leaf_reg=settings["l2"],
                                  random_seed=seed, verbose=False, thread_count=4,
                                  allow_writing_files=False,
                                  **({"cat_features": CATEGORICAL_FEATURES} if family == "cat_native" else {}))
    if family == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=settings["iterations"], learning_rate=settings["rate"],
                             max_depth=settings["depth"], min_child_weight=10,
                             reg_lambda=settings["l2"], tree_method="hist", n_jobs=4, **common)
    if family not in ["forest", "extra"]:
        raise ValueError(f"Unknown fitted classifier: {family}")
    estimator = RandomForestClassifier if family == "forest" else ExtraTreesClassifier
    return estimator(n_estimators=250, min_samples_leaf=settings["leaf"],
                     max_features=.8, n_jobs=4, **common)


def prepare_dashboard_features(features, family):
    if family == "hist":
        return features
    result = features.copy()
    for column in CATEGORICAL_FEATURES:
        result[column] = result[column].astype(str) if family == "cat_native" else result[column].astype(float)
    return result
