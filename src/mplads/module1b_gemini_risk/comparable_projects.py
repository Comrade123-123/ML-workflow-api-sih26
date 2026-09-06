"""Historical comparable-project retrieval for Module 1B.

Reuses the existing, already-built and already-validated MPLADS dataset
(data/mplads_option2/datasets/work_level_completed_dataset.csv -- the exact
dataset scripts/train_underspend_risk.py trains on) as the sole source of
historical evidence. No new dataset is built, downloaded, or invented here.

Matching is deterministic and explainable: same Work Type (falling back to
Work Category if too few Work Type matches exist), ranked by closeness of
Recommended Amount. No fuzzy/ML matching -- this is retrieval, not a model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from mplads.module1b_gemini_risk.schemas import ComparableProject

COMPARABLE_DATASET_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "mplads_option2" / "datasets" / "work_level_completed_dataset.csv"
)

MIN_MATCHES_FOR_WORK_TYPE = 3
DEFAULT_TOP_N = 5


def _load_dataset() -> Optional[pd.DataFrame]:
    if not COMPARABLE_DATASET_PATH.exists():
        return None
    return pd.read_csv(COMPARABLE_DATASET_PATH)


def _row_to_comparable(row: pd.Series) -> ComparableProject:
    def _get(col: str):
        if col not in row.index:
            return None
        val = row[col]
        return None if pd.isna(val) else val

    return ComparableProject(
        work_id=str(row["Work ID"]),
        work_category=str(row["Work Category"]),
        work_type=str(row["Work Type"]),
        work_description=_get("Work Description"),
        state=str(row["State"]),
        recommended_amount=float(row["Recommended Amount (INR)"]),
        sanction_date=_get("Sanction Date"),
        completion_date=_get("Completion Date"),
        amount_disbursed=_get("Amount Disbursed (INR)"),
        underspend_pct=_get("underspend_pct"),
    )


def find_comparable_projects(
    work_type: str,
    work_category: str,
    state: str,
    recommended_amount: float,
    top_n: int = DEFAULT_TOP_N,
) -> tuple[list[ComparableProject], str]:
    """Returns (comparables, match_criteria_description). Never raises for a
    missing dataset -- callers get an empty list with an honest reason
    string, since the absence of comparables is not itself a service
    failure (Module 1B can still return the underspend signal + Gemini's
    assessment without historical examples)."""
    df = _load_dataset()
    if df is None:
        return [], "Historical comparable-project dataset not found on disk -- no comparables available."

    same_state = df["State"] == state

    work_type_matches = df[(df["Work Type"] == work_type) & same_state]
    if len(work_type_matches) < MIN_MATCHES_FOR_WORK_TYPE:
        work_type_matches = df[df["Work Type"] == work_type]

    if len(work_type_matches) >= MIN_MATCHES_FOR_WORK_TYPE:
        pool = work_type_matches
        criteria = f"Same Work Type ('{work_type}')"
        if same_state.any() and len(df[(df["Work Type"] == work_type) & same_state]) >= MIN_MATCHES_FOR_WORK_TYPE:
            criteria += f" and State ('{state}')"
    else:
        category_matches = df[(df["Work Category"] == work_category) & same_state]
        if len(category_matches) < MIN_MATCHES_FOR_WORK_TYPE:
            category_matches = df[df["Work Category"] == work_category]
        pool = category_matches
        criteria = f"Fell back to same Work Category ('{work_category}') -- too few same-Work-Type matches"

    if pool.empty:
        return [], f"No completed works found matching Work Type/Category for '{work_type}'."

    pool = pool.copy()
    pool["_amount_distance"] = (pool["Recommended Amount (INR)"] - recommended_amount).abs()
    pool = pool.sort_values("_amount_distance").head(top_n)

    criteria += f", ranked by closeness of Recommended Amount (target: {recommended_amount:,.0f})."

    comparables = [_row_to_comparable(row) for _, row in pool.iterrows()]
    return comparables, criteria
