"""Structural validation for Module 1's raw source data.

This layer only checks *shape*: required columns exist, dtypes are sane,
and the join key used to line up sanctioned vs. actual amounts resolves.
It says nothing about whether the resulting target is fit to train on —
that is target_validation.py's job.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from mplads.config import (
    COL_AMOUNT_DISBURSED,
    COL_RECOMMENDED_AMOUNT,
    COL_WORK_CATEGORY,
    COL_WORK_ID,
    WORKS_COMPLETED_CSV,
    WORKS_RECOMMENDED_CSV,
)
from mplads.module1_cost_overrun.exceptions import DataValidationError

REQUIRED_RECOMMENDED_COLUMNS = {
    COL_WORK_ID,
    COL_WORK_CATEGORY,
    "Work Type",
    "State",
    "IDA District",
    "Recommended Date",
    COL_RECOMMENDED_AMOUNT,
    "Sanction Date",
}

REQUIRED_COMPLETED_COLUMNS = {
    COL_WORK_ID,
    COL_WORK_CATEGORY,
    "Work Type",
    "State",
    "IDA District",
    "Completion Date",
    COL_AMOUNT_DISBURSED,
}


def _require_columns(df: pd.DataFrame, required: set[str], frame_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise DataValidationError(
            f"{frame_name} is missing required column(s): {sorted(missing)}. "
            f"Found columns: {sorted(df.columns)}"
        )


def validate_recommended_frame(df: pd.DataFrame) -> None:
    """Validate the raw Works_Recommended dataframe structurally."""
    if df.empty:
        raise DataValidationError("Works_Recommended data is empty.")
    _require_columns(df, REQUIRED_RECOMMENDED_COLUMNS, "Works_Recommended")

    if not pd.api.types.is_numeric_dtype(df[COL_RECOMMENDED_AMOUNT]):
        raise DataValidationError(
            f"'{COL_RECOMMENDED_AMOUNT}' must be numeric, got dtype "
            f"{df[COL_RECOMMENDED_AMOUNT].dtype}."
        )

    if df[COL_WORK_ID].isna().any():
        raise DataValidationError("Works_Recommended contains null 'Work ID' values.")


def validate_completed_frame(df: pd.DataFrame) -> None:
    """Validate the raw Works_Completed dataframe structurally."""
    if df.empty:
        raise DataValidationError("Works_Completed data is empty.")
    _require_columns(df, REQUIRED_COMPLETED_COLUMNS, "Works_Completed")

    if not pd.api.types.is_numeric_dtype(df[COL_AMOUNT_DISBURSED]):
        raise DataValidationError(
            f"'{COL_AMOUNT_DISBURSED}' must be numeric, got dtype "
            f"{df[COL_AMOUNT_DISBURSED].dtype}."
        )

    if df[COL_WORK_ID].isna().any():
        raise DataValidationError("Works_Completed contains null 'Work ID' values.")


def validate_join(recommended: pd.DataFrame, completed: pd.DataFrame) -> pd.DataFrame:
    """Join completed works to their sanctioned amount via Work ID.

    Raises DataValidationError if the join key does not resolve at all
    (e.g. wrong column, mismatched formats) — as opposed to merely having
    some non-matching rows, which is expected and handled by dropping them.
    """
    recommended_dedup = recommended.drop_duplicates(subset=COL_WORK_ID, keep="first")
    merged = completed.merge(
        recommended_dedup[[COL_WORK_ID, COL_RECOMMENDED_AMOUNT]],
        on=COL_WORK_ID,
        how="inner",
    )
    if merged.empty:
        raise DataValidationError(
            "Joining Works_Completed to Works_Recommended on 'Work ID' produced "
            "zero matching rows. Cannot pair sanctioned cost with actual cost."
        )
    return merged


def load_joined_training_frame(
    recommended_csv: Union[str, Path] = WORKS_RECOMMENDED_CSV,
    completed_csv: Union[str, Path] = WORKS_COMPLETED_CSV,
) -> pd.DataFrame:
    """Read, structurally validate, and join the two source CSVs used for
    Module 1: sanctioned amount (Works_Recommended) paired with actual final
    cost (Works_Completed) via Work ID. Raises DataValidationError on any
    structural problem before a single row is fitted or evaluated.
    """
    recommended = pd.read_csv(recommended_csv, low_memory=False)
    completed = pd.read_csv(completed_csv, low_memory=False)

    validate_recommended_frame(recommended)
    validate_completed_frame(completed)

    return validate_join(recommended, completed)
