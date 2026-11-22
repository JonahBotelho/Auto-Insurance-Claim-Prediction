"""Estimate how likely observed cents are to be a sum of several payouts."""

import numpy as np
import pandas as pd

from src.dashboard_model import profile_keys
from src.payout_source_evidence import PayoutSourceEvidence

MODULI = (100, 1000)
COLUMNS = ("residue_supported", *(
    f"mod_{modulus}_{name}" for modulus in MODULI
    for name in ("local_pairs", "local_triples", "total_pairs", "total_triples")
))


class PayoutResidueEvidence:
    def fit(self, features, amounts):
        source = PayoutSourceEvidence().fit(features, amounts)
        self.profiles_ = source.profiles_
        money = np.array(list(source.whole_support_), dtype=int)
        count = np.array(list(source.whole_support_.values()), dtype=float)
        self.pooled_ = {modulus: np.bincount(money % modulus, weights=count, minlength=modulus) for modulus in MODULI}
        return self

    def transform(self, features):
        keys, inverse = np.unique(profile_keys(features), return_inverse=True)
        results = np.asarray([self.describe(key) for key in keys], dtype=float)
        return pd.DataFrame(results[inverse], columns=COLUMNS, index=features.index)

    def describe(self, key):
        cent, scaled = PayoutSourceEvidence._by_source(self.profiles_.get(key, {}))
        total = max(scaled, default=0)
        residual = total - sum(cent)
        if residual <= 0:
            return (0.,) * len(COLUMNS)
        return (1., *(value for modulus in MODULI for value in self.ratios(tuple(cent), residual, total, modulus)))

    def ratios(self, own, residual, total, modulus):
        counts = self.pooled_[modulus].copy()
        np.add.at(counts, np.asarray(own, dtype=int) % modulus, -1.)
        single = (counts + 30. / modulus) / (counts.sum() + 30.)
        spectrum = np.fft.rfft(single)
        # Circular convolution pools the cents patterns of summed payouts
        # across independent profiles.
        pair = np.maximum(1e-10, np.fft.irfft(spectrum ** 2, n=modulus))
        triple = np.maximum(1e-10, np.fft.irfft(spectrum ** 3, n=modulus))
        residues = [residual % modulus, total % modulus]
        return np.column_stack([np.log(pair[residues] / single[residues]),
                                np.log(triple[residues] / single[residues])]).ravel()
