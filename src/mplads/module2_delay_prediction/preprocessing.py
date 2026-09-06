"""Feature engineering and the sklearn preprocessing pipeline for Module 2.

Deliberately excludes any feature derived from the actual Sanction->
Completion duration (or Completion Date at all): that duration is exactly
what the peer-relative label is computed from, so including it (or a proxy
of it) as a feature would leak the label. Only pre-completion-knowable
fields are used, so the same feature set applies identically to a
historical completed project (for training) and a currently in-progress
project (for inference).
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mplads.config import (
    COL_IDA_DISTRICT,
    COL_RECOMMENDED_AMOUNT,
    COL_RECOMMENDED_DATE,
    COL_SANCTION_DATE,
    COL_STATE,
    COL_WORK_CATEGORY,
    COL_WORK_TYPE,
)

CATEGORICAL_FEATURES = [COL_WORK_CATEGORY, COL_WORK_TYPE, COL_STATE, COL_IDA_DISTRICT]
NUMERIC_FEATURES = [COL_RECOMMENDED_AMOUNT, "days_recommended_to_sanction"]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def add_processing_lag_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Derive 'days_recommended_to_sanction' -- the only duration feature
    used, since it is fixed and known at (or shortly after) sanction time,
    unlike Sanction->Completion which defines the label."""
    out = df.copy()

    def _date_column(col: str) -> pd.Series:
        if col in out.columns:
            return pd.to_datetime(out[col], errors="coerce")
        return pd.Series(pd.NaT, index=out.index)

    recommended_date = _date_column(COL_RECOMMENDED_DATE)
    sanction_date = _date_column(COL_SANCTION_DATE)
    out["days_recommended_to_sanction"] = (sanction_date - recommended_date).dt.days
    return out


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ALL_FEATURES:
        if col not in out.columns:
            out[col] = pd.NA
    return out[ALL_FEATURES]


def build_preprocessing_pipeline() -> ColumnTransformer:
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
