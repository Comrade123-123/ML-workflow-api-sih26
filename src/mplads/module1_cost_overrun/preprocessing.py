"""Feature engineering and the sklearn preprocessing pipeline for Module 1.

Column mapping (actual CSV columns only — nothing invented):
  Works_Recommended_Cleaned.csv:
    - 'Recommended Amount (INR)'  -> sanctioned_cost (numeric feature + label input)
    - 'Work Category'             -> categorical feature
    - 'Work Type'                 -> categorical feature (nullable)
    - 'State', 'IDA District'     -> categorical features
    - 'Recommended Date', 'Sanction Date' -> derived duration feature
  Works_Completed_Cleaned.csv:
    - 'Amount Disbursed (INR)'    -> actual final cost (regression target, when validated)
    - 'Completion Date'           -> derived duration feature (with Sanction Date)
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mplads.config import (
    COL_COMPLETION_DATE,
    COL_IDA_DISTRICT,
    COL_RECOMMENDED_AMOUNT,
    COL_RECOMMENDED_DATE,
    COL_SANCTION_DATE,
    COL_STATE,
    COL_WORK_CATEGORY,
    COL_WORK_TYPE,
)

CATEGORICAL_FEATURES = [COL_WORK_CATEGORY, COL_WORK_TYPE, COL_STATE, COL_IDA_DISTRICT]
NUMERIC_FEATURES = [COL_RECOMMENDED_AMOUNT, "days_recommended_to_sanction", "days_sanction_to_completion"]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def add_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive duration/elapsed-day features from the raw date columns.

    Returns a copy; does not mutate the input frame. Missing dates (e.g. a
    not-yet-completed work has no Completion Date) yield NaN durations,
    which the preprocessing pipeline's imputer handles downstream.
    """
    out = df.copy()

    def _date_column(col: str) -> pd.Series:
        if col in out.columns:
            return pd.to_datetime(out[col], errors="coerce")
        return pd.Series(pd.NaT, index=out.index)

    recommended_date = _date_column(COL_RECOMMENDED_DATE)
    sanction_date = _date_column(COL_SANCTION_DATE)
    completion_date = _date_column(COL_COMPLETION_DATE)

    out["days_recommended_to_sanction"] = (sanction_date - recommended_date).dt.days
    out["days_sanction_to_completion"] = (completion_date - sanction_date).dt.days

    return out


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the modeling feature columns, adding any missing ones as
    all-NaN so the pipeline's imputers can fill them consistently."""
    out = df.copy()
    for col in ALL_FEATURES:
        if col not in out.columns:
            out[col] = pd.NA
    return out[ALL_FEATURES]


def build_preprocessing_pipeline() -> ColumnTransformer:
    """The reusable sklearn preprocessing pipeline: impute + one-hot encode
    categoricals, impute + scale numerics. Fit-able and testable independently
    of any trained regressor."""
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )
