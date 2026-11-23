"""Find two-term explanations for a Bonus-Malus increase's reference window."""

import numpy as np
import pandas as pd


def extend_jump_windows(jumps, profiles, forward):
    """Let a same-Bonus-Malus term lead into an observed anniversary jump."""
    first_hops = pd.DataFrame(
        ((prior, middle) for prior, middles in forward.items() for middle in middles),
        columns=["key_t", "key_m"],
    )
    if first_hops.empty:
        return jumps
    columns = ["feature1", "feature2"]
    indexed = profiles.set_index("key")[columns]
    direct = jumps.loc[jumps.tier.eq(0) & jumps.exact_bm_rule.eq(1)].rename(columns={"key_t": "key_m"})
    chains = first_hops.merge(direct, on="key_m")
    if chains.empty:
        return jumps
    before = indexed.reindex(chains.key_t)
    middle = indexed.reindex(chains.key_m)
    duration = 12 * before.feature1.to_numpy()
    first_delay = middle.feature2.to_numpy() - before.feature2.to_numpy()
    next_delta = first_delay + chains.delta_licence.to_numpy()
    overlap = np.maximum(
        0., np.minimum(duration, next_delta - 2.) - np.maximum(0., next_delta - 14.)
    )
    adjacent = (first_delay > 0) & (np.abs(first_delay - duration) <= 1.25)
    distinct = chains.key_t.ne(chains.key_n).to_numpy()
    chains = chains.assign(tier=3., overlap_months=overlap, delay=next_delta - duration,
                           delta_licence=next_delta)
    valid = adjacent & distinct & (overlap >= 1.5) & (next_delta <= 24.)
    return pd.concat([jumps, chains.loc[valid, jumps.columns]], ignore_index=True)
