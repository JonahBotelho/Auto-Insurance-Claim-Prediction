"""Fit a second history calibration for quotes while retaining the primary classifier."""

from copy import copy

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.bonus_malus_claim_ensemble import EMTimelineClaimEnsemble


class CrossfitQuoteEMTimelineClaimEnsemble(EMTimelineClaimEnsemble):
    """Use the larger quote when both history calibrations flag a claim."""

    def _fit_calibrators(self, rows):
        super()._fit_calibrators(rows)
        self._first_calibration_panel = rows

    def fit(self, features, amounts):
        self.pricing_head_ = None
        super().fit(features, amounts)
        panel = self._first_calibration_panel
        del self._first_calibration_panel
        # The second head shares full-data components and payout histories.
        # Only the held-out history calibration inputs need a second set of folds.
        head = copy(self)
        with threadpool_limits(limits=2):
            extra = self._calibration_folds(features, np.asarray(amounts), repeat=1)
            calibration = pd.concat([panel, *extra], ignore_index=True)
            EMTimelineClaimEnsemble._fit_calibrators(head, calibration)
        self.pricing_head_ = head
        return self

    def _score_batch(self, features):
        risk, prices, flags, cutoff = super()._score_batch(features)
        if self.pricing_head_ is None or not flags.any():
            return risk, prices, flags, cutoff
        _, repeated_quotes, repeated_flags, _ = EMTimelineClaimEnsemble._score_batch(self.pricing_head_, features)
        prices = np.where(flags & repeated_flags, np.maximum(prices, repeated_quotes), prices)
        return risk, prices, flags, cutoff
