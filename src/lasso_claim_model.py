"""
Lasso regression model for claim amounts (Module 6: shrinkage methods).

Uses the shared preprocessing in preprocessing.py (one-hot categories,
standardized numeric features). By default the target is fitted as
log(1 + ClaimAmount): on raw amounts a few $100k+ claims dominate the
squared-error fit (CV MAE about 195), while on the log scale CV MAE is
about 103.4.
"""

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

from preprocessing import (  # noqa: F401  (re-exported for older imports)
    CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_preprocessor, fit_on_log_target,
    log_scale_to_claim_amount,
)

# Chosen by 5-fold cross-validation in analysis/lasso_cross_validation.py.
DEFAULT_ALPHA = 0.1


def build_lasso_claim_model(alpha: float = DEFAULT_ALPHA, log_target: bool = True):
    """Return an unfitted lasso pipeline with scikit-learn's fit/predict methods."""
    pipeline = Pipeline([
        ("preprocess", build_preprocessor()),
        ("lasso", Lasso(alpha=alpha, max_iter=10000)),
    ])
    return fit_on_log_target(pipeline) if log_target else pipeline
