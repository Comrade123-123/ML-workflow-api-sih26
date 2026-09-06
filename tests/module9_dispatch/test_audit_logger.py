import pytest

from mplads.module9_dispatch import audit_logger
from mplads.module9_dispatch.schemas import EventType


@pytest.fixture(autouse=True)
def isolated_audit_trail(tmp_path, monkeypatch):
    monkeypatch.setattr(audit_logger, "MODULE9_AUDIT_TRAIL_FILE", tmp_path / "audit_trail.json")


def test_append_log_creates_entry_with_generated_id_and_timestamp():
    entry = audit_logger.append_log(
        project_id="WS/MP1/2024-2025/1",
        event_type=EventType.ALERT_TRIGGERED,
        actor="system_monitor",
        details={"reasons": ["test"]},
    )
    assert entry.project_id == "WS/MP1/2024-2025/1"
    assert entry.event_type == EventType.ALERT_TRIGGERED
    assert entry.actor == "system_monitor"
    assert entry.log_id
    assert entry.timestamp


def test_append_log_is_additive_not_overwriting():
    audit_logger.append_log("proj-1", EventType.ALERT_TRIGGERED, "system_monitor", {})
    audit_logger.append_log("proj-2", EventType.INSPECTOR_DISPATCHED, "authority_officer", {})
    all_entries = audit_logger.read_audit_trail()
    assert len(all_entries) == 2


def test_read_audit_trail_filters_by_project_id():
    audit_logger.append_log("proj-1", EventType.ALERT_TRIGGERED, "system_monitor", {})
    audit_logger.append_log("proj-2", EventType.ALERT_TRIGGERED, "system_monitor", {})
    filtered = audit_logger.read_audit_trail(project_id="proj-1")
    assert len(filtered) == 1
    assert filtered[0].project_id == "proj-1"


def test_read_audit_trail_filters_by_actor():
    audit_logger.append_log("proj-1", EventType.ALERT_TRIGGERED, "system_monitor", {})
    audit_logger.append_log("proj-2", EventType.INSPECTOR_DISPATCHED, "authority_officer", {})
    filtered = audit_logger.read_audit_trail(actor="authority_officer")
    assert len(filtered) == 1
    assert filtered[0].actor == "authority_officer"


def test_read_audit_trail_on_missing_file_returns_empty_list():
    assert audit_logger.read_audit_trail() == []
