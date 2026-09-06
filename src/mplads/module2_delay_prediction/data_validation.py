"""Structural validation for Module 2's raw source data.

Checks shape only: required columns exist, dtypes are sane, and the Work ID
join between Works_Recommended and Works_Completed resolves. Whether the
resulting peer-relative labels are fit to train on is target_validation.py's
job.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

from mplads.config import (
    COL_AMOUNT_DISBURSED,
    COL_COMPLETION_DATE,
    COL_IDA_DISTRICT,
    COL_RECOMMENDED_AMOUNT,
    COL_RECOMMENDED_DATE,
    COL_SANCTION_DATE,
    COL_STATE,
    COL_WORK_CATEGORY,
    COL_WORK_ID,
    COL_WORK_TYPE,
    WORKS_COMPLETED_CSV,
    WORKS_RECOMMENDED_CSV,
)
from mplads.module2_delay_prediction.exceptions import DataValidationError

REQUIRED_RECOMMENDED_COLUMNS = {
    COL_WORK_ID,
    COL_WORK_CATEGORY,
    COL_WORK_TYPE,
    COL_STATE,
    COL_IDA_DISTRICT,
    COL_RECOMMENDED_DATE,
    COL_RECOMMENDED_AMOUNT,
    COL_SANCTION_DATE,
}

REQUIRED_COMPLETED_COLUMNS = {
    COL_WORK_ID,
    COL_WORK_CATEGORY,
    COL_WORK_TYPE,
    COL_STATE,
    COL_IDA_DISTRICT,
    COL_COMPLETION_DATE,
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
    if df.empty:
        raise DataValidationError("Works_Recommended data is empty.")
    _require_columns(df, REQUIRED_RECOMMENDED_COLUMNS, "Works_Recommended")
    if df[COL_WORK_ID].isna().any():
        raise DataValidationError("Works_Recommended contains null 'Work ID' values.")


def validate_completed_frame(df: pd.DataFrame) -> None:
    if df.empty:
        raise DataValidationError("Works_Completed data is empty.")
    _require_columns(df, REQUIRED_COMPLETED_COLUMNS, "Works_Completed")
    if df[COL_WORK_ID].isna().any():
        raise DataValidationError("Works_Completed contains null 'Work ID' values.")


def validate_join(recommended: pd.DataFrame, completed: pd.DataFrame) -> pd.DataFrame:
    """Join completed works to their sanction-time features via Work ID."""
    recommended_dedup = recommended.drop_duplicates(subset=COL_WORK_ID, keep="first")
    merged = completed.merge(
        recommended_dedup[
            [COL_WORK_ID, COL_WORK_CATEGORY, COL_WORK_TYPE, COL_STATE, COL_IDA_DISTRICT,
             COL_RECOMMENDED_AMOUNT, COL_RECOMMENDED_DATE, COL_SANCTION_DATE]
        ],
        on=COL_WORK_ID,
        how="inner",
        suffixes=("_completed", ""),
    )
    if merged.empty:
        raise DataValidationError(
            "Joining Works_Completed to Works_Recommended on 'Work ID' produced "
            "zero matching rows."
        )

    sanction_date = pd.to_datetime(merged[COL_SANCTION_DATE], errors="coerce")
    completion_date = pd.to_datetime(merged[COL_COMPLETION_DATE], errors="coerce")
    merged = merged.assign(
        _sanction_date_parsed=sanction_date,
        _completion_date_parsed=completion_date,
    )
    merged = merged.dropna(subset=["_sanction_date_parsed", "_completion_date_parsed"])
    if merged.empty:
        raise DataValidationError(
            "No rows have both a parseable 'Sanction Date' and 'Completion Date'."
        )

    merged["days_sanction_to_completion"] = (
        merged["_completion_date_parsed"] - merged["_sanction_date_parsed"]
    ).dt.days
    return merged.drop(columns=["_sanction_date_parsed", "_completion_date_parsed"])


def load_joined_training_frame(
    recommended_csv: Union[str, Path] = WORKS_RECOMMENDED_CSV,
    completed_csv: Union[str, Path] = WORKS_COMPLETED_CSV,
) -> pd.DataFrame:
    """Read, structurally validate, and join the two source CSVs used for
    Module 2 labeling and training."""
    recommended = pd.read_csv(recommended_csv, low_memory=False)
    completed = pd.read_csv(completed_csv, low_memory=False)

    validate_recommended_frame(recommended)
    validate_completed_frame(completed)

    return validate_join(recommended, completed)
