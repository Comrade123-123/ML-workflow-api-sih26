"""The fail-fast target-validation gate for Module 1.

A cost-overrun regression model can only be trained if the historical data
contains genuine examples of overrun (actual final cost > sanctioned cost).
This module computes that overrun signal from real columns only
(Recommended Amount (INR) vs Amount Disbursed (INR)) and refuses to let
training proceed if there isn't enough of it — it never substitutes a
different target (e.g. underspend/utilization) to make the check pass.
"""
from __future__ import annotations

import pandas as pd

from mplads.config import (
    COL_AMOUNT_DISBURSED,
    COL_RECOMMENDED_AMOUNT,
    MIN_OVERRUN_RATIO,
    MIN_OVERRUN_SAMPLES,
)
from mplads.module1_cost_overrun.exceptions import TargetValidationError
from mplads.module1_cost_overrun.schemas import TargetValidationReport


def compute_overrun_pct(
    df: pd.DataFrame,
    sanctioned_col: str = COL_RECOMMENDED_AMOUNT,
    actual_col: str = COL_AMOUNT_DISBURSED,
) -> pd.Series:
    """overrun_pct = (actual - sanctioned) / sanctioned * 100.

    Rows with a non-positive sanctioned amount are excluded (undefined
    percentage), not treated as overrun or underrun.
    """
    valid = df[(df[sanctioned_col].notna()) & (df[sanctioned_col] > 0) & (df[actual_col].notna())]
    return (valid[actual_col] - valid[sanctioned_col]) / valid[sanctioned_col] * 100.0


def validate_overrun_target(
    df: pd.DataFrame,
    sanctioned_col: str = COL_RECOMMENDED_AMOUNT,
    actual_col: str = COL_AMOUNT_DISBURSED,
    min_positive_examples: int = MIN_OVERRUN_SAMPLES,
    min_positive_ratio: float = MIN_OVERRUN_RATIO,
) -> TargetValidationReport:
    """Verify the historical data supports training a cost-overrun regressor.

    Raises TargetValidationError if the data does not contain enough
    positive-overrun examples. Returns a TargetValidationReport (also usable
    for a passing check, e.g. surfaced via /status) otherwise.
    """
    overrun_pct = compute_overrun_pct(df, sanctioned_col, actual_col)
    total = len(overrun_pct)

    if total == 0:
        raise TargetValidationError(
            f"No rows with both a valid '{sanctioned_col}' and '{actual_col}' were found. "
            "Cannot compute an overrun target from this data."
        )

    positive = int((overrun_pct > 0).sum())
    ratio = positive / total

    is_valid = positive >= min_positive_examples and ratio >= min_positive_ratio

    if not is_valid:
        message = (
            f"Target validation FAILED: {positive} of {total} historical records "
            f"({ratio:.4%}) show actual cost exceeding sanctioned cost "
            f"('{actual_col}' > '{sanctioned_col}'). "
            f"A minimum of {min_positive_examples} positive-overrun examples "
            f"(and at least {min_positive_ratio:.2%} of the data) is required to train "
            "a genuine cost-overrun regression model. The current dataset does not "
            "contain enough real overrun history to train Module 1 - training will "
            "not proceed, and no substitute target (e.g. underspend/utilization) will "
            "be used in its place."
        )
        report = TargetValidationReport(
            is_valid=False,
            total_examples=total,
            positive_overrun_examples=positive,
            positive_overrun_ratio=ratio,
            min_required_samples=min_positive_examples,
            min_required_ratio=min_positive_ratio,
            message=message,
        )
        raise TargetValidationError(message) from None

    message = (
        f"Target validation PASSED: {positive} of {total} records "
        f"({ratio:.4%}) show genuine cost overrun, meeting the required "
        f"minimum of {min_positive_examples} samples / {min_positive_ratio:.2%} ratio."
    )
    return TargetValidationReport(
        is_valid=True,
        total_examples=total,
        positive_overrun_examples=positive,
        positive_overrun_ratio=ratio,
        min_required_samples=min_positive_examples,
        min_required_ratio=min_positive_ratio,
        message=message,
    )


def try_validate_overrun_target(
    df: pd.DataFrame,
    sanctioned_col: str = COL_RECOMMENDED_AMOUNT,
    actual_col: str = COL_AMOUNT_DISBURSED,
    min_positive_examples: int = MIN_OVERRUN_SAMPLES,
    min_positive_ratio: float = MIN_OVERRUN_RATIO,
) -> TargetValidationReport:
    """Non-raising variant of validate_overrun_target, for status reporting.

    Runs the same computation but returns a failing report instead of
    raising, so /status can show diagnostics without a 500 error.
    """
    try:
        return validate_overrun_target(
            df, sanctioned_col, actual_col, min_positive_examples, min_positive_ratio
        )
    except TargetValidationError as exc:
        overrun_pct = compute_overrun_pct(df, sanctioned_col, actual_col)
        total = len(overrun_pct)
        positive = int((overrun_pct > 0).sum()) if total else 0
        ratio = (positive / total) if total else 0.0
        return TargetValidationReport(
            is_valid=False,
            total_examples=total,
            positive_overrun_examples=positive,
            positive_overrun_ratio=ratio,
            min_required_samples=min_positive_examples,
            min_required_ratio=min_positive_ratio,
            message=str(exc),
        )
