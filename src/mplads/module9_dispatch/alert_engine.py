"""Rule-based alert evaluation for Module 9.

Reuses Module 2's public inference entrypoint (predict_delay_risk) and
Module 7's public service (resolve_entity/get_pulse_score) rather than
reimplementing either. Neither Module 2 nor Module 7 is modified.
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

from mplads.config import COL_WORK_ID, MODULE9_DELAY_PROBABILITY_THRESHOLD, WORKS_RECOMMENDED_CSV
from mplads.module2_delay_prediction.exceptions import ModelNotAvailableError
from mplads.module2_delay_prediction.inference import predict_delay_risk
from mplads.module2_delay_prediction.schemas import PredictRequest as Module2PredictRequest
from mplads.module7_pulse_score.service import get_pulse_score, resolve_entity
from mplads.module9_dispatch.audit_logger import append_log
from mplads.module9_dispatch.exceptions import ProjectNotFoundError
from mplads.module9_dispatch.schemas import AlertEvaluateResponse, EventType, RiskLevel

COL_CONSTITUENCY = "Constituency"
COL_WORK_CATEGORY = "Work Category"
COL_WORK_TYPE = "Work Type"
COL_STATE = "State"
COL_IDA_DISTRICT = "IDA District"
COL_RECOMMENDED_AMOUNT = "Recommended Amount (INR)"
COL_RECOMMENDED_DATE = "Recommended Date"
COL_SANCTION_DATE = "Sanction Date"

_PULSE_ALERT_BANDS = {"CRITICAL", "NEEDS_ATTENTION"}
_RISK_ALERT_LEVELS = {"HIGH", "CRITICAL"}


def lookup_project(project_id: str, recommended: Optional[pd.DataFrame] = None) -> pd.Series:
    """Resolves project_id to exactly one Works_Recommended row.

    Raises ProjectNotFoundError both when the Work ID is unknown and when it
    is one of the ~5,440 not-yet-sanctioned "NA-" placeholder rows that
    share a duplicate Work ID string across many distinct real
    recommendations -- there is no single, confirmed project to act on in
    that case.
    """
    if recommended is None:
        recommended = pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)

    matches = recommended[recommended[COL_WORK_ID] == project_id]
    if len(matches) == 0:
        raise ProjectNotFoundError(f"No project found with Work ID '{project_id}'.")
    if len(matches) > 1:
        raise ProjectNotFoundError(
            f"Work ID '{project_id}' matches {len(matches)} rows -- it is one of the "
            "not-yet-sanctioned placeholder IDs shared across many distinct recommendations, "
            "not a single, confirmed project. Alerts and dispatch require a uniquely-identified "
            "sanctioned Work ID."
        )
    return matches.iloc[0]


def _build_module2_request(project_row: pd.Series) -> Module2PredictRequest:
    return Module2PredictRequest(
        project_id=str(project_row[COL_WORK_ID]),
        sanctioned_cost=float(project_row[COL_RECOMMENDED_AMOUNT]),
        work_category=str(project_row[COL_WORK_CATEGORY]),
        work_type=project_row[COL_WORK_TYPE] if pd.notna(project_row[COL_WORK_TYPE]) else None,
        state=str(project_row[COL_STATE]),
        ida_district=project_row[COL_IDA_DISTRICT] if pd.notna(project_row[COL_IDA_DISTRICT]) else None,
        recommended_date=project_row[COL_RECOMMENDED_DATE] if pd.notna(project_row[COL_RECOMMENDED_DATE]) else None,
        sanction_date=project_row[COL_SANCTION_DATE] if pd.notna(project_row[COL_SANCTION_DATE]) else None,
    )


def evaluate_alert(
    project_id: str, external_risk_signals: Optional[Dict[str, RiskLevel]] = None
) -> AlertEvaluateResponse:
    recommended = pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)
    project_row = lookup_project(project_id, recommended)
    constituency = resolve_entity(str(project_row[COL_CONSTITUENCY]), recommended)

    reasons: list[str] = []

    delay_probability: Optional[float] = None
    delay_status: Optional[str] = None
    try:
        delay_response = predict_delay_risk(_build_module2_request(project_row))
        delay_probability = delay_response.delay_probability
        delay_status = delay_response.status
        if delay_status == "LIKELY_DELAYED":
            reasons.append(f"Module 2 status is LIKELY_DELAYED (delay_probability={delay_probability}).")
        elif delay_probability >= MODULE9_DELAY_PROBABILITY_THRESHOLD:
            reasons.append(
                f"Module 2 delay_probability={delay_probability} >= threshold "
                f"{MODULE9_DELAY_PROBABILITY_THRESHOLD}."
            )
    except ModelNotAvailableError:
        reasons.append("Module 2 delay model unavailable -- delay-based rule skipped, not fabricated.")

    pulse = get_pulse_score(constituency)
    if pulse.health_band in _PULSE_ALERT_BANDS:
        reasons.append(f"Constituency '{constituency}' pulse health_band is {pulse.health_band}.")

    if external_risk_signals:
        for module_name, level in external_risk_signals.items():
            if level in _RISK_ALERT_LEVELS:
                reasons.append(f"External signal from {module_name}: {level}.")

    alert_triggered = len(reasons) > 0
    audit_log_id = None
    if alert_triggered:
        entry = append_log(
            project_id=str(project_row[COL_WORK_ID]),
            event_type=EventType.ALERT_TRIGGERED,
            actor="system_monitor",
            details={
                "reasons": reasons,
                "delay_probability": delay_probability,
                "delay_status": delay_status,
                "constituency": constituency,
                "pulse_score": pulse.pulse_score,
                "health_band": pulse.health_band,
                "external_risk_signals": external_risk_signals,
            },
        )
        audit_log_id = entry.log_id

    return AlertEvaluateResponse(
        project_id=str(project_row[COL_WORK_ID]),
        alert_triggered=alert_triggered,
        reasons=reasons,
        constituency=constituency,
        delay_probability=delay_probability,
        delay_status=delay_status,
        pulse_score=pulse.pulse_score,
        health_band=pulse.health_band,
        audit_log_id=audit_log_id,
    )
