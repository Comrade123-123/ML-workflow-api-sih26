"""Orchestration for Module 7: entity resolution, data rollup, delay-rate via
Module 2, velocity benchmarking, snapshot-based delta, and the final pulse
score.

Reuses Module 2's trained model (for delay rate) and its training-data join
(for national duration benchmarks) rather than recomputing that logic --
Module 7 depends on Module 2 the same way the project's teammates' Module 6
risk engine is meant to be an optional additional input, not a replacement
for these fundamentals.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from mplads.config import (
    COL_AMOUNT_DISBURSED,
    COL_FUND_DISBURSED,
    COL_RECOMMENDED_AMOUNT,
    COL_WORK_CATEGORY,
    COL_WORK_ID,
    COL_WORK_TYPE,
    EXPENDITURE_CSV,
    MODULE7_HISTORY_FILE,
    MODULE7_MIN_PEER_GROUP_SIZE,
    WORKS_COMPLETED_CSV,
    WORKS_RECOMMENDED_CSV,
)
from mplads.module2_delay_prediction.data_validation import load_joined_training_frame
from mplads.module2_delay_prediction.labeling import DURATION_COL
from mplads.module2_delay_prediction.labeling import assign_status_labels as module2_assign_status_labels
from mplads.module2_delay_prediction.registry import load_active_model as load_module2_model
from mplads.module7_pulse_score import calculator
from mplads.module7_pulse_score.exceptions import EntityNotFoundError
from mplads.module7_pulse_score.schemas import PulseScoreResponse

DELAYED_STATUSES = {"AT_RISK", "LIKELY_DELAYED"}

COL_CONSTITUENCY = "Constituency"
COL_MP_NAME = "Hon'ble Members of Parliament"

MODULE2_FEATURE_COLUMNS = [
    COL_WORK_CATEGORY,
    COL_WORK_TYPE,
    "State",
    "IDA District",
    COL_RECOMMENDED_AMOUNT,
    "Recommended Date",
    "Sanction Date",
]


# --- entity resolution ----------------------------------------------------


def _build_entity_index(recommended_df: pd.DataFrame) -> dict[str, str]:
    """Maps normalized (upper/stripped) Constituency or MP name -> canonical
    Constituency string. Constituency<->MP is a 1:1 bijection in this data,
    so either can be used as the path identifier."""
    index: dict[str, str] = {}
    pairs = recommended_df[[COL_CONSTITUENCY, COL_MP_NAME]].drop_duplicates()
    for constituency, mp_name in pairs.itertuples(index=False):
        canonical = str(constituency).strip()
        index[canonical.upper()] = canonical
        index[str(mp_name).strip().upper()] = canonical
    return index


def resolve_entity(raw_entity_id: str, recommended_df: pd.DataFrame) -> str:
    index = _build_entity_index(recommended_df)
    key = raw_entity_id.strip().upper()
    if key not in index:
        raise EntityNotFoundError(
            f"No MPLADS constituency or MP found matching '{raw_entity_id}'. "
            "Note: Rajya Sabha MPs are not attributable to a constituency in "
            "this dataset and cannot be scored."
        )
    return index[key]


# --- snapshot store (delta_vs_last_period) --------------------------------


def _load_history() -> dict:
    if not MODULE7_HISTORY_FILE.exists():
        return {}
    try:
        return json.loads(MODULE7_HISTORY_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_history(history: dict) -> None:
    MODULE7_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODULE7_HISTORY_FILE.write_text(json.dumps(history, indent=2))


def _record_and_compute_delta(entity_id: str, new_score: float) -> float:
    history = _load_history()
    previous = history.get(entity_id)
    delta = 0.0 if previous is None else round(new_score - previous["last_score"], 1)
    history[entity_id] = {
        "last_score": new_score,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_history(history)
    return delta


# --- national duration benchmark (for velocity) ---------------------------


def _national_benchmark_lookup(national_joined: pd.DataFrame):
    type_counts = national_joined.groupby(COL_WORK_TYPE)[DURATION_COL].count()
    type_median = national_joined.groupby(COL_WORK_TYPE)[DURATION_COL].median()
    category_median = national_joined.groupby(COL_WORK_CATEGORY)[DURATION_COL].median()
    overall_median = national_joined[DURATION_COL].median()

    def benchmark_for_row(work_type: object, work_category: object) -> float:
        if pd.notna(work_type) and type_counts.get(work_type, 0) >= MODULE7_MIN_PEER_GROUP_SIZE:
            return float(type_median[work_type])
        if pd.notna(work_category) and work_category in category_median.index:
            return float(category_median[work_category])
        return float(overall_median)

    return benchmark_for_row


# --- main entrypoint --------------------------------------------------------


def get_pulse_score(raw_entity_id: str) -> PulseScoreResponse:
    recommended = pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)
    completed = pd.read_csv(WORKS_COMPLETED_CSV, low_memory=False)
    expenditure = pd.read_csv(EXPENDITURE_CSV, low_memory=False)

    canonical_id = resolve_entity(raw_entity_id, recommended)

    entity_recommended = recommended[recommended[COL_CONSTITUENCY] == canonical_id]
    entity_completed = completed[completed[COL_CONSTITUENCY] == canonical_id]

    total_works = len(entity_recommended)
    completed_works = len(entity_completed)

    # --- fund utilization: completed works' disbursed amount, plus
    # Expenditure-table disbursement for works NOT already counted via
    # Works_Completed (avoids double-counting the ~25% of Work IDs present
    # in both).
    completed_work_ids = set(entity_completed[COL_WORK_ID])
    entity_expenditure = expenditure[
        (expenditure[COL_CONSTITUENCY] == canonical_id)
        & (~expenditure[COL_WORK_ID].isin(completed_work_ids))
    ]
    total_sanctioned = float(entity_recommended[COL_RECOMMENDED_AMOUNT].fillna(0).sum())
    total_disbursed = float(entity_completed[COL_AMOUNT_DISBURSED].fillna(0).sum()) + float(
        entity_expenditure[COL_FUND_DISBURSED].fillna(0).sum()
    )
    utilization_pct = calculator.fund_utilization_pct(total_sanctioned, total_disbursed)

    # --- delay rate: prefer Module 2's trained model, scored uniformly
    # across every recommended work (completed and in-progress alike).
    module2_model = load_module2_model()
    if module2_model is not None:
        predictions = module2_model.predict(entity_recommended[MODULE2_FEATURE_COLUMNS])
        delayed_mask = predictions.isin(DELAYED_STATUSES)
        delayed_works_count = int(delayed_mask.sum())
        entity_delay_rate = calculator.delay_rate(total_works, delayed_works_count)
        delay_source = "model"
    elif completed_works > 0:
        # Fallback: Module 2's model is unavailable. We can only honestly
        # assess delay for works that have actually completed (their true
        # peer-relative status is knowable); in-progress works with unknown
        # final duration are excluded from both numerator and denominator
        # rather than guessed at.
        national_joined = load_joined_training_frame()
        labeled_national = module2_assign_status_labels(national_joined)
        entity_labeled = labeled_national[labeled_national[COL_CONSTITUENCY] == canonical_id]
        delayed_works_count = int(entity_labeled["status"].isin(DELAYED_STATUSES).sum())
        entity_delay_rate = calculator.delay_rate(completed_works, delayed_works_count)
        delay_source = "fallback_completed_only"
    else:
        delayed_works_count = 0
        entity_delay_rate = 0.0
        delay_source = "no_data"

    # --- velocity: constituency's own completed-work durations vs a
    # national, work-type-mix-weighted benchmark.
    actual_avg_duration: Optional[float] = None
    benchmark_duration: Optional[float] = None
    if completed_works > 0:
        national_joined = load_joined_training_frame()
        entity_joined = national_joined[national_joined[COL_CONSTITUENCY] == canonical_id]
        if not entity_joined.empty:
            benchmark_fn = _national_benchmark_lookup(national_joined)
            benchmarks = entity_joined.apply(
                lambda row: benchmark_fn(row[COL_WORK_TYPE], row[COL_WORK_CATEGORY]), axis=1
            )
            actual_avg_duration = float(entity_joined[DURATION_COL].mean())
            benchmark_duration = float(benchmarks.mean())

    velocity = calculator.velocity_score(actual_avg_duration, benchmark_duration)

    completion = calculator.completion_rate(total_works, completed_works)
    pulse_score = calculator.compute_pulse_score(completion, utilization_pct, entity_delay_rate, velocity)
    band = calculator.health_band(pulse_score)
    delta = _record_and_compute_delta(canonical_id, pulse_score)

    return PulseScoreResponse(
        entity_id=canonical_id,
        pulse_score=pulse_score,
        delta_vs_last_period=delta,
        fund_utilization_rate_pct=round(utilization_pct, 1),
        total_works=total_works,
        completed_works=completed_works,
        delayed_works_count=delayed_works_count,
        health_band=band,
    )
