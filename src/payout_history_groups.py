"""Identify structural evidence groups in fitted payout histories."""

import numpy as np


GROUPS = ("no_positive", "whole_only", "scaled_only", "reconciled", "incomplete", "exhausted")


def evidence_flag(column):
    # CSVs mix booleans and zero for profiles with empty histories.
    return column.astype(str).isin(("True", "1", "1.0")).to_numpy()


def evidence_groups(rows):
    return np.select([
        rows.prior_positives.eq(0), evidence_flag(rows.copies_complete), evidence_flag(rows.sum_attested),
        evidence_flag(rows.whole_only), evidence_flag(rows.scaled_only),
    ], [0, 5, 3, 1, 2], default=4)


