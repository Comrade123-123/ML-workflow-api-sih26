"""Field inspector registry and dispatch workflow for Module 9.

SEED_INSPECTORS is illustrative/demo data only -- there is no real inspector
personnel dataset in this project. Identities are deliberately generic
(numbered "Field Inspector N") rather than realistic-sounding full names, to
avoid resembling fabricated records of real people; coverage areas use real
State values from the MPLADS data so matching against real projects is
meaningful. Mutable status/assignment state is persisted separately and
seeded from this catalog on first access.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, List

import pandas as pd

from mplads.config import (
    COL_STATE,
    MODULE9_INSPECTORS_STATE_FILE,
    WORKS_RECOMMENDED_CSV,
)
from mplads.module9_dispatch.alert_engine import lookup_project
from mplads.module9_dispatch.audit_logger import append_log
from mplads.module9_dispatch.exceptions import InspectorNotAvailableError, InspectorNotFoundError
from mplads.module9_dispatch.schemas import (
    CurrentAssignment,
    DispatchRequest,
    DispatchResponse,
    EventType,
    InspectorInfo,
    InspectorStatus,
    InspectorSuggestionResponse,
)

SEED_INSPECTORS: List[dict] = [
    {"inspector_id": "INSP-001", "name": "Field Inspector 1", "assigned_state_or_district": "Uttar Pradesh"},
    {"inspector_id": "INSP-002", "name": "Field Inspector 2", "assigned_state_or_district": "Uttar Pradesh"},
    {"inspector_id": "INSP-003", "name": "Field Inspector 3", "assigned_state_or_district": "Bihar"},
    {"inspector_id": "INSP-004", "name": "Field Inspector 4", "assigned_state_or_district": "Maharashtra"},
    {"inspector_id": "INSP-005", "name": "Field Inspector 5", "assigned_state_or_district": "Kerala"},
    {"inspector_id": "INSP-006", "name": "Field Inspector 6", "assigned_state_or_district": "Karnataka"},
    {"inspector_id": "INSP-007", "name": "Field Inspector 7", "assigned_state_or_district": "Punjab"},
    {"inspector_id": "INSP-008", "name": "Field Inspector 8", "assigned_state_or_district": "Rajasthan"},
    {"inspector_id": "INSP-009", "name": "Field Inspector 9", "assigned_state_or_district": "West Bengal"},
    {"inspector_id": "INSP-010", "name": "Field Inspector 10", "assigned_state_or_district": "Tamil Nadu"},
]


def _seed_state() -> Dict[str, dict]:
    return {
        row["inspector_id"]: {**row, "status": InspectorStatus.AVAILABLE.value, "current_assignment": None}
        for row in SEED_INSPECTORS
    }


def _load_state() -> Dict[str, dict]:
    if not MODULE9_INSPECTORS_STATE_FILE.exists():
        state = _seed_state()
        _save_state(state)
        return state
    try:
        return json.loads(MODULE9_INSPECTORS_STATE_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        state = _seed_state()
        _save_state(state)
        return state


def _save_state(state: Dict[str, dict]) -> None:
    MODULE9_INSPECTORS_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODULE9_INSPECTORS_STATE_FILE.write_text(json.dumps(state, indent=2))


def list_inspectors() -> List[InspectorInfo]:
    state = _load_state()
    return [InspectorInfo.model_validate(record) for record in state.values()]


def suggest_inspectors(project_id: str) -> InspectorSuggestionResponse:
    recommended = pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)
    project_row = lookup_project(project_id, recommended)
    state_name = str(project_row[COL_STATE])
    constituency = str(project_row["Constituency"])

    inspectors = list_inspectors()
    eligible = [
        inspector
        for inspector in inspectors
        if inspector.status == InspectorStatus.AVAILABLE and inspector.assigned_state_or_district == state_name
    ]
    return InspectorSuggestionResponse(
        project_id=str(project_row["Work ID"]),
        constituency=constituency,
        state=state_name,
        eligible_inspectors=eligible,
    )


def dispatch_inspector(request: DispatchRequest) -> DispatchResponse:
    recommended = pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)
    project_row = lookup_project(request.project_id, recommended)
    canonical_project_id = str(project_row["Work ID"])

    state = _load_state()
    record = state.get(request.inspector_id)
    if record is None:
        raise InspectorNotFoundError(f"No inspector found with inspector_id '{request.inspector_id}'.")
    if record["status"] != InspectorStatus.AVAILABLE.value:
        raise InspectorNotAvailableError(
            f"Inspector '{request.inspector_id}' is not AVAILABLE (current status: {record['status']})."
        )

    assigned_at = datetime.now(timezone.utc).isoformat()
    record["status"] = InspectorStatus.ASSIGNED.value
    record["current_assignment"] = CurrentAssignment(
        project_id=canonical_project_id,
        inspection_date=request.inspection_date,
        notes=request.notes,
        assigned_at=assigned_at,
    ).model_dump(mode="json")
    state[request.inspector_id] = record
    _save_state(state)

    audit_entry = append_log(
        project_id=canonical_project_id,
        event_type=EventType.INSPECTOR_DISPATCHED,
        actor=request.actor,
        details={
            "inspector_id": request.inspector_id,
            "inspector_name": record["name"],
            "inspection_date": request.inspection_date,
            "notes": request.notes,
        },
    )

    return DispatchResponse(inspector=InspectorInfo.model_validate(record), audit_log=audit_entry)
