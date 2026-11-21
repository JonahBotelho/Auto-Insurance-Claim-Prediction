"""Allow a high payout above the cap when matching fitting payouts support it."""

from copy import copy

import numpy as np

from src.dashboard_model import profile_keys
from src.feature_schema import PAYOUT_FACTOR
from src.payout_pricing import payout_signature


TAIL_POLICIES = ("observed_tail", "observed_tail_blend", "profile_tail_cap", "single_copy_tail", "paired_source_tail")


def supported_tails(pricer, query, reference, probabilities, fallback, share):
    cap = pricer.tail_cap_
    selected = (probabilities > .5) & np.isclose(reference, cap, rtol=0., atol=1e-8)
    original = reference.copy()
    if not selected.any():
        return {name: original.copy() for name in TAIL_POLICIES}
    unclipped = copy(pricer)
    unclipped.tail_cap_ = np.inf
    _, _, deeper, _ = unclipped.predict(query.loc[selected], probabilities[selected], fallback[selected],
                                      scaled_share=share[selected])
    histories = [pricer.model_.positive_.get(key, {}) for key in profile_keys(query.loc[selected])]
    matches = [payout_signature(price) for price in deeper]
    copies = np.array([history.get(key, 0) for key, history in zip(matches, histories)])
    paired = np.array([(key[0], not key[1]) in history for key, history in zip(matches, histories)])
    maximum = np.array([max((amount / 100. if cent else PAYOUT_FACTOR * amount / 100. for amount, cent in history),
                            default=cap) for history in histories])
    observed = original.copy()
    observed[selected] = np.where(copies > 0, np.maximum(cap, deeper), cap)
    exposed = original.copy()
    exposed[selected] = np.where(probabilities[selected] >= .65, np.maximum(cap, np.minimum(maximum, deeper)), cap)
    unique, matched = original.copy(), original.copy()
    unique[selected] = np.where(copies == 1, np.maximum(cap, deeper), cap)
    matched[selected] = np.where((copies == 1) & paired, np.maximum(cap, deeper), cap)
    return {
        "observed_tail": observed, "observed_tail_blend": (observed + original) / 2, "profile_tail_cap": exposed,
        "single_copy_tail": unique, "paired_source_tail": matched,
    }


