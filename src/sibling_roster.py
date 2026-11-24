"""Estimate missing profile roster rows from complementary-exposure sibling terms."""

import numpy as np
import pandas as pd

from src.bonus_malus_timeline import STATIC_POLICY_COLUMNS, _numeric_frame, _summarize_profiles
from src.feature_schema import PROFILE_COLUMNS

MAX_SIBLING_BLOCK_SIZE = 150
EXPOSURE_COMPLEMENT_TOLERANCE = 0.025


def _combined_profiles(reference_profiles, query_features):
    """Merge fitting profile counts with the scored query batch without target access."""
    query_num = _numeric_frame(query_features)
    query_keys = pd.util.hash_pandas_object(query_num, index=False).to_numpy()
    query_profiles = _summarize_profiles(query_num, query_keys).drop(columns=["pid"])
    combined = pd.concat([reference_profiles, query_profiles], ignore_index=True)
    merged = combined.groupby(list(PROFILE_COLUMNS), as_index=False).agg(
        n_portfolio=("n_portfolio", "sum"),
        key=("key", "first"),
    )
    merged["pid"] = np.arange(len(merged))
    return merged, query_keys


def sibling_extra_roster_rows(reference_profiles, query_features, *, max_roster=8):
    """Return non-negative extra roster rows implied by complementary-exposure same-policy siblings.

    Mid-year policy modifications split a calendar year into two terms whose exposures sum to ~1
    while sharing the same latent guarantee count. When a linked sibling term has more observed
    rows than the target profile, the difference bounds how many target rows were thinned away.
    """
    n_rows = len(query_features)
    if n_rows == 0 or reference_profiles is None or reference_profiles.empty:
        return np.zeros(n_rows, dtype=int)

    profiles, query_keys = _combined_profiles(reference_profiles, query_features)
    block_id = pd.util.hash_pandas_object(profiles[list(STATIC_POLICY_COLUMNS)], index=False)
    sizes = block_id.value_counts()
    subset = profiles[block_id.isin(sizes[(sizes >= 2) & (sizes <= MAX_SIBLING_BLOCK_SIZE)].index)].assign(
        _bid=block_id,
    )
    if subset.empty:
        return np.zeros(n_rows, dtype=int)

    pairs = subset.merge(subset, on="_bid", suffixes=("_t", "_s"))
    pairs = pairs[pairs["pid_t"] != pairs["pid_s"]]
    if pairs.empty:
        return np.zeros(n_rows, dtype=int)

    complementary = (
        (pairs["feature2_s"] - pairs["feature2_t"]).abs().le(18)
        & (pairs["feature8_s"] - pairs["feature8_t"]).abs().le(1)
        & (pairs["feature3_s"] - pairs["feature3_t"]).abs().le(1)
        & (pairs["feature1_t"] + pairs["feature1_s"] - 1.0).abs().le(EXPOSURE_COMPLEMENT_TOLERANCE)
    )
    siblings = pairs.loc[complementary]
    if siblings.empty:
        return np.zeros(n_rows, dtype=int)

    indexed = profiles.set_index("key")
    indexed["sibling_portfolio"] = siblings.groupby("key_t")["n_portfolio_s"].mean()
    info = indexed.reindex(query_keys)
    observed = info["n_portfolio"].to_numpy(dtype=float)
    sibling_size = info["sibling_portfolio"].fillna(info["n_portfolio"]).to_numpy(dtype=float)
    extra = np.maximum(0, np.floor(sibling_size - observed + 1e-9).astype(int))
    return np.minimum(extra, np.maximum(0, int(max_roster) - observed.astype(int)))
