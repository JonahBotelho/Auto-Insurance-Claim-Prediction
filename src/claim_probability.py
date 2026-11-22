"""Numerically stable transformations for fitted claim probabilities."""

import numpy as np
from scipy.special import logit


def log_odds(probability):
    return logit(np.clip(probability, .001, .999))
