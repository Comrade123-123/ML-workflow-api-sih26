import pandas as pd
import pytest

from mplads.config import WORKS_RECOMMENDED_CSV
from mplads.module9_dispatch import audit_logger, inspector_service
from mplads.module9_dispatch.exceptions import (
    InspectorNotAvailableError,
    InspectorNotFoundError,
    ProjectNotFoundError,
)
from mplads.module9_dispatch.schemas import DispatchRequest, InspectorStatus


@pytest.fixture(scope="module")
def recommended_df():
    return pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)


@pytest.fixture(scope="module")
def real_project_for_state(recommended_df):
    """A real project whose State matches one of the seeded inspectors
    (Karnataka -> INSP-006), so suggestion/dispatch tests have a guaranteed
    eligible inspector."""
    karnataka = recommended_df[
        (~recommended_df["Work ID"].astype(str).str.startswith("NA-")) & (recommended_df["State"] == "Karnataka")
    ]
    return karnataka.iloc[0]["Work ID"]


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(inspector_service, "MODULE9_INSPECTORS_STATE_FILE", tmp_path / "inspectors_state.json")
    monkeypatch.setattr(audit_logger, "MODULE9_AUDIT_TRAIL_FILE", tmp_path / "audit_trail.json")


def test_list_inspectors_seeds_from_catalog_on_first_access():
    inspectors = inspector_service.list_inspectors()
    assert len(inspectors) == len(inspector_service.SEED_INSPECTORS)
    assert all(i.status == InspectorStatus.AVAILABLE for i in inspectors)


def test_suggest_inspectors_matches_by_state(real_project_for_state):
    suggestion = inspector_service.suggest_inspectors(real_project_for_state)
    assert suggestion.state == "Karnataka"
    assert len(suggestion.eligible_inspectors) >= 1
    assert all(i.assigned_state_or_district == "Karnataka" for i in suggestion.eligible_inspectors)


def test_suggest_inspectors_unknown_project_raises():
    with pytest.raises(ProjectNotFoundError):
        inspector_service.suggest_inspectors("WS/MP-DOES-NOT-EXIST/2024-2025/999999")


def test_dispatch_inspector_updates_status_and_writes_audit_log(real_project_for_state):
    response = inspector_service.dispatch_inspector(
        DispatchRequest(
            project_id=real_project_for_state,
            inspector_id="INSP-006",
            inspection_date="2026-09-10",
            notes="Routine verification",
        )
    )
    assert response.inspector.status == InspectorStatus.ASSIGNED
    assert response.inspector.current_assignment.project_id == real_project_for_state
    assert response.audit_log.event_type.value == "INSPECTOR_DISPATCHED"
    assert response.audit_log.actor == "authority_officer"

    inspectors = inspector_service.list_inspectors()
    updated = next(i for i in inspectors if i.inspector_id == "INSP-006")
    assert updated.status == InspectorStatus.ASSIGNED


def test_dispatch_inspector_already_assigned_raises(real_project_for_state):
    inspector_service.dispatch_inspector(
        DispatchRequest(project_id=real_project_for_state, inspector_id="INSP-006", inspection_date="2026-09-10")
    )
    with pytest.raises(InspectorNotAvailableError):
        inspector_service.dispatch_inspector(
            DispatchRequest(project_id=real_project_for_state, inspector_id="INSP-006", inspection_date="2026-09-11")
        )


def test_dispatch_inspector_unknown_inspector_raises(real_project_for_state):
    with pytest.raises(InspectorNotFoundError):
        inspector_service.dispatch_inspector(
            DispatchRequest(project_id=real_project_for_state, inspector_id="INSP-999", inspection_date="2026-09-10")
        )


def test_dispatch_inspector_unknown_project_raises():
    with pytest.raises(ProjectNotFoundError):
        inspector_service.dispatch_inspector(
            DispatchRequest(
                project_id="WS/MP-DOES-NOT-EXIST/2024-2025/999999",
                inspector_id="INSP-001",
                inspection_date="2026-09-10",
            )
        )
