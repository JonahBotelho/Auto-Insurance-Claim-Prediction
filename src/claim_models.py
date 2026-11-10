"""
Builders for the course's regression models applied to claim amounts.

Each builder returns an unfitted scikit-learn estimator with fit/predict, so
the comparison script and predict.py can treat every model the same way.

- Ridge: Module 6 shrinkage (L2 penalty; keeps every feature, unlike lasso)
- Polynomial: Module 3 basis expansion, with lasso to control the extra terms
- Decision tree: Module 7, used here as a regression tree
- k-NN: Module 8, used here as a regressor (average of the k nearest claims)

log_target=True fits on log(1 + ClaimAmount); see preprocessing.py for why.
"""

from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor

from preprocessing import build_preprocessor, fit_on_log_target


def _with_optional_log_target(model, log_target: bool):
    return fit_on_log_target(model) if log_target else model


def build_ridge_claim_model(alpha: float = 100.0, log_target: bool = True):
    pipeline = Pipeline([("preprocess", build_preprocessor()), ("ridge", Ridge(alpha=alpha))])
    return _with_optional_log_target(pipeline, log_target)


def build_polynomial_claim_model(degree: int = 2, alpha: float = 0.01):
    """Polynomial terms of every feature, fitted with lasso on the log target.

    Degree 2 on the encoded features gives about 1,000 terms, so plain least
    squares would overfit; the lasso penalty removes the useless ones.
    """
    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("polynomial", PolynomialFeatures(degree=degree, include_bias=False)),
        ("lasso", Lasso(alpha=alpha, max_iter=2000)),
    ])
    return fit_on_log_target(pipeline)


def build_tree_claim_model(max_depth: int = 6, criterion: str = "squared_error",
                           log_target: bool = False):
    """Regression tree. Trees split on raw values, so no scaling is needed.

    criterion="absolute_error" makes each leaf predict the median claim of
    its customers, which matches the MAE metric but trains more slowly.
    min_samples_leaf stops leaves from fitting a handful of customers.
    """
    tree = DecisionTreeRegressor(
        max_depth=max_depth, criterion=criterion, min_samples_leaf=50, random_state=0
    )
    return _with_optional_log_target(tree, log_target)


def build_knn_claim_model(neighbor_count: int = 25, log_target: bool = True):
    """k-NN regressor. Features are standardized so no single feature
    dominates the distance (Module 8)."""
    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("knn", KNeighborsRegressor(n_neighbors=neighbor_count)),
    ])
    return _with_optional_log_target(pipeline, log_target)
