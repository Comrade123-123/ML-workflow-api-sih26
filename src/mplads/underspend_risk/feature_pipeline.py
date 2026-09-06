"""Deterministic feature engineering for underspend_risk, matching
scripts/mplads_option2_pipeline/train_underspend_model.py exactly -- same
feature names, same derivations, same column order. No feature here would
only become known after the prediction point (Sanction Date); see
OPTION2_MPLADS_TARGET_FEASIBILITY_REPORT.md section 8 for the full
SAFE/LEAKAGE audit this mirrors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mplads.underspend_risk.historical_rates import HistoricalRateLookup

FEATURE_COLUMNS_NUMERIC = [
    "Recommended Amount (INR)",
    "recommended_amount_log",
    "recommendation_to_sanction_days",
    "state_historical_underspend_rate",
    "work_type_historical_underspend_rate",
]
FEATURE_COLUMNS_CATEGORICAL = ["Work Category", "State", "Work Type"]
ALL_FEATURES = FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL


def add_derived_features(df: pd.DataFrame, rate_lookup: HistoricalRateLookup) -> pd.DataFrame:
    """Adds recommended_amount_log, recommendation_to_sanction_days, and the
    two historical-rate lookup features. Deterministic: same input + same
    rate_lookup always produces the same output, and unknown State/Work Type
    values fall back to the lookup's global_prior rather than raising or
    silently zero-filling."""
    out = df.copy()

    recommended_date = pd.to_datetime(out["Recommended Date"], errors="coerce")
    sanction_date = pd.to_datetime(out["Sanction Date"], errors="coerce")
    out["recommendation_to_sanction_days"] = (sanction_date - recommended_date).dt.days

    out["recommended_amount_log"] = np.log1p(out["Recommended Amount (INR)"].astype(float))

    out["state_historical_underspend_rate"] = out["State"].apply(rate_lookup.state_rate)
    out["work_type_historical_underspend_rate"] = out["Work Type"].apply(rate_lookup.work_type_rate)

    return out


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Returns exactly ALL_FEATURES, in that exact order, so training and
    inference always present identically-shaped input to the pipeline."""
    out = df.copy()
    for col in ALL_FEATURES:
        if col not in out.columns:
            out[col] = pd.NA
    return out[ALL_FEATURES]


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Identical structure to train_underspend_model.py's build_preprocessor()."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")), ("encode", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer([("num", numeric, FEATURE_COLUMNS_NUMERIC), ("cat", categorical, FEATURE_COLUMNS_CATEGORICAL)])
