import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mplads.api.main import app
from mplads.config import WORKS_RECOMMENDED_CSV
from mplads.module7_pulse_score import service as module7_service
from mplads.module9_dispatch import audit_logger, inspector_service

client = TestClient(app)


@pytest.fixture(scope="module")
def recommended_df():
    return pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)


@pytest.fixture(scope="module")
def real_project_for_state(recommended_df):
    karnataka = recommended_df[
        (~recommended_df["Work ID"].astype(str).str.startswith("NA-")) & (recommended_df["State"] == "Karnataka")
    ]
    return karnataka.iloc[0]["Work ID"]


@pytest.fixture(scope="module")
def placeholder_project_id(recommended_df):
    placeholders = recommended_df[recommended_df["Work ID"].astype(str).str.startswith("NA-")]
    return placeholders.iloc[0]["Work ID"]


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(audit_logger, "MODULE9_AUDIT_TRAIL_FILE", tmp_path / "audit_trail.json")
    monkeypatch.setattr(inspector_service, "MODULE9_INSPECTORS_STATE_FILE", tmp_path / "inspectors_state.json")
    monkeypatch.setattr(module7_service, "MODULE7_HISTORY_FILE", tmp_path / "pulse_history.json")


def test_evaluate_alert_endpoint(real_project_for_state):
    response = client.post("/api/v1/alerts/evaluate", json={"project_id": real_project_for_state})
    assert response.status_code == 200
    body = response.json()
    assert body["project_id"] == real_project_for_state
    assert "reasons" in body


def test_evaluate_alert_endpoint_ambiguous_placeholder_returns_404(placeholder_project_id):
    response = client.post("/api/v1/alerts/evaluate", json={"project_id": placeholder_project_id})
    assert response.status_code == 404
    assert response.json()["detail"]["error"] == "project_not_found"


def test_evaluate_alert_endpoint_unknown_project_returns_404():
    response = client.post("/api/v1/alerts/evaluate", json={"project_id": "WS/MP-NOPE/2024-2025/1"})
    assert response.status_code == 404


def test_audit_trail_endpoint_reflects_triggered_alert(real_project_for_state):
    eval_response = client.post(
        "/api/v1/alerts/evaluate",
        json={"project_id": real_project_for_state, "external_risk_signals": {"module6_risk_scoring_engine": "CRITICAL"}},
    )
    assert eval_response.json()["alert_triggered"] is True

    response = client.get("/api/v1/alerts/audit-trail", params={"project_id": real_project_for_state})
    assert response.status_code == 200
    entries = response.json()
    assert len(entries) == 1
    assert entries[0]["event_type"] == "ALERT_TRIGGERED"


def test_list_inspectors_endpoint():
    response = client.get("/api/v1/inspectors")
    assert response.status_code == 200
    body = response.json()
    assert len(body) == len(inspector_service.SEED_INSPECTORS)


def test_suggest_inspectors_endpoint(real_project_for_state):
    """Work IDs contain literal '/' characters (e.g. WS/MP620/2024-2025/133166),
    so the route uses the :path converter -- passing the raw id works directly."""
    response = client.get(f"/api/v1/inspectors/suggest/{real_project_for_state}")
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "Karnataka"
    assert len(body["eligible_inspectors"]) >= 1


def test_dispatch_inspector_endpoint(real_project_for_state):
    response = client.post(
        "/api/v1/inspectors/dispatch",
        json={
            "project_id": real_project_for_state,
            "inspector_id": "INSP-006",
            "inspection_date": "2026-09-10",
            "notes": "Routine check",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["inspector"]["status"] == "ASSIGNED"
    assert body["audit_log"]["event_type"] == "INSPECTOR_DISPATCHED"

    # dispatching the same (now-assigned) inspector again should 409
    response2 = client.post(
        "/api/v1/inspectors/dispatch",
        json={"project_id": real_project_for_state, "inspector_id": "INSP-006", "inspection_date": "2026-09-11"},
    )
    assert response2.status_code == 409
    assert response2.json()["detail"]["error"] == "inspector_not_available"


def test_dispatch_inspector_endpoint_unknown_inspector_returns_404(real_project_for_state):
    response = client.post(
        "/api/v1/inspectors/dispatch",
        json={"project_id": real_project_for_state, "inspector_id": "INSP-999", "inspection_date": "2026-09-10"},
    )
    assert response.status_code == 404
    assert response.json()["detail"]["error"] == "inspector_not_found"
