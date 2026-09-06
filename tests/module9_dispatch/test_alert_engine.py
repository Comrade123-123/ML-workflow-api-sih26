import pandas as pd
import pytest

from mplads.config import WORKS_RECOMMENDED_CSV
from mplads.module7_pulse_score import service as module7_service
from mplads.module9_dispatch import alert_engine, audit_logger
from mplads.module9_dispatch.exceptions import ProjectNotFoundError


@pytest.fixture(scope="module")
def recommended_df():
    return pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)


@pytest.fixture(scope="module")
def real_project_id(recommended_df):
    """A genuine, uniquely-identified (non-placeholder) Work ID."""
    non_placeholder = recommended_df[~recommended_df["Work ID"].astype(str).str.startswith("NA-")]
    return non_placeholder.iloc[0]["Work ID"]


@pytest.fixture(scope="module")
def placeholder_project_id(recommended_df):
    """A Work ID string shared by many distinct not-yet-sanctioned rows."""
    placeholders = recommended_df[recommended_df["Work ID"].astype(str).str.startswith("NA-")]
    return placeholders.iloc[0]["Work ID"]


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(audit_logger, "MODULE9_AUDIT_TRAIL_FILE", tmp_path / "audit_trail.json")
    monkeypatch.setattr(module7_service, "MODULE7_HISTORY_FILE", tmp_path / "pulse_history.json")


def test_lookup_project_ambiguous_placeholder_raises(recommended_df, placeholder_project_id):
    with pytest.raises(ProjectNotFoundError):
        alert_engine.lookup_project(placeholder_project_id, recommended_df)


def test_lookup_project_unknown_id_raises(recommended_df):
    with pytest.raises(ProjectNotFoundError):
        alert_engine.lookup_project("WS/MP-DOES-NOT-EXIST/2024-2025/999999", recommended_df)


def test_lookup_project_real_id_resolves_uniquely(recommended_df, real_project_id):
    row = alert_engine.lookup_project(real_project_id, recommended_df)
    assert row["Work ID"] == real_project_id


def test_evaluate_alert_real_project_returns_well_formed_response(real_project_id):
    response = alert_engine.evaluate_alert(real_project_id)
    assert response.project_id == real_project_id
    assert response.constituency
    assert 0.0 <= response.pulse_score <= 100.0
    assert response.health_band in {"EXCELLENT", "GOOD", "NEEDS_ATTENTION", "CRITICAL"}
    if response.alert_triggered:
        assert response.audit_log_id is not None
        assert len(response.reasons) > 0
    else:
        assert response.audit_log_id is None
        assert response.reasons == []


def test_evaluate_alert_writes_audit_log_only_when_triggered(real_project_id):
    response = alert_engine.evaluate_alert(real_project_id)
    entries = audit_logger.read_audit_trail(project_id=real_project_id)
    if response.alert_triggered:
        assert len(entries) == 1
        assert entries[0].event_type.value == "ALERT_TRIGGERED"
        assert entries[0].actor == "system_monitor"
    else:
        assert len(entries) == 0


def test_evaluate_alert_external_high_risk_signal_forces_trigger(real_project_id):
    response = alert_engine.evaluate_alert(real_project_id, external_risk_signals={"module6_risk_scoring_engine": "CRITICAL"})
    assert response.alert_triggered is True
    assert any("module6_risk_scoring_engine" in r for r in response.reasons)


def test_evaluate_alert_low_external_signal_does_not_force_trigger(real_project_id):
    response = alert_engine.evaluate_alert(real_project_id, external_risk_signals={"module6_risk_scoring_engine": "LOW"})
    assert not any("module6_risk_scoring_engine" in r for r in response.reasons)


def test_evaluate_alert_unknown_project_raises():
    with pytest.raises(ProjectNotFoundError):
        alert_engine.evaluate_alert("WS/MP-DOES-NOT-EXIST/2024-2025/999999")
