"""Peer-relative delay labeling for Module 2.

Since the source data has no expected/target completion date, "delayed" is
defined relative to how long peer projects (same Work Type, or Work Category
when a Work Type group is too small to be statistically meaningful) actually
took to go from Sanction Date to Completion Date:

  duration <= P75 of peer group  -> ON_TRACK
  P75 < duration <= P90          -> AT_RISK
  duration > P90                 -> LIKELY_DELAYED

This is a documented, data-driven policy choice (approved for this project),
not an invented ground truth -- it is computed entirely from real historical
durations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mplads.config import (
    MODULE2_AT_RISK_PERCENTILE,
    MODULE2_LIKELY_DELAYED_PERCENTILE,
    MODULE2_MIN_PEER_GROUP_SIZE,
    MODULE2_PEER_GROUP_COL,
    MODULE2_PEER_GROUP_FALLBACK_COL,
    MODULE2_STATUS_LABELS,
)

DURATION_COL = "days_sanction_to_completion"
ON_TRACK, AT_RISK, LIKELY_DELAYED = MODULE2_STATUS_LABELS


def _group_thresholds(df: pd.DataFrame, group_col: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    filled = df[group_col].fillna("UNKNOWN")
    grouped = df[DURATION_COL].groupby(filled)
    at_risk_threshold = grouped.transform(lambda s: s.quantile(MODULE2_AT_RISK_PERCENTILE))
    delayed_threshold = grouped.transform(lambda s: s.quantile(MODULE2_LIKELY_DELAYED_PERCENTILE))
    group_size = grouped.transform("size")
    return at_risk_threshold, delayed_threshold, group_size


def assign_status_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 'status' (ON_TRACK/AT_RISK/LIKELY_DELAYED) and 'peer_group_used'
    columns to a copy of df. Requires DURATION_COL to already be present
    (produced by data_validation.validate_join)."""
    if DURATION_COL not in df.columns:
        raise KeyError(f"'{DURATION_COL}' column is required to assign delay labels.")

    out = df.copy()

    fine_at_risk, fine_delayed, fine_size = _group_thresholds(out, MODULE2_PEER_GROUP_COL)
    coarse_at_risk, coarse_delayed, _ = _group_thresholds(out, MODULE2_PEER_GROUP_FALLBACK_COL)

    use_fine = fine_size >= MODULE2_MIN_PEER_GROUP_SIZE
    at_risk_threshold = fine_at_risk.where(use_fine, coarse_at_risk)
    delayed_threshold = fine_delayed.where(use_fine, coarse_delayed)

    duration = out[DURATION_COL]
    status = pd.Series(ON_TRACK, index=out.index, dtype=object)
    status = status.mask(duration > at_risk_threshold, AT_RISK)
    status = status.mask(duration > delayed_threshold, LIKELY_DELAYED)

    out["status"] = status
    out["peer_group_used"] = np.where(use_fine, MODULE2_PEER_GROUP_COL, MODULE2_PEER_GROUP_FALLBACK_COL)
    return out
