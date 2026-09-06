import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mplads.api.main import app
from mplads.config import WORKS_RECOMMENDED_CSV
from mplads.module7_pulse_score import service as module7_service
from mplads.module8_report_generation import scheduler

client = TestClient(app)


@pytest.fixture
def real_constituency():
    df = pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)
    return df.iloc[0]["Constituency"]


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "MODULE8_SUBSCRIPTION_RUNS_FILE", tmp_path / "subscription_runs.json")
    monkeypatch.setattr(module7_service, "MODULE7_HISTORY_FILE", tmp_path / "pulse_history.json")


def test_generate_report_endpoint_json(real_constituency):
    response = client.post(
        "/api/v1/reports/generate",
        json={"report_type": "FINANCIAL_SUMMARY", "entity_id": real_constituency, "format": "json"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["report_type"] == "FINANCIAL_SUMMARY"
    assert body["entity_id"] == real_constituency
    assert body["rendered_text"] is None
    assert "summary" in body["data"]


def test_generate_report_endpoint_csv(real_constituency):
    response = client.post(
        "/api/v1/reports/generate",
        json={"report_type": "FUND_UTILIZATION", "entity_id": real_constituency, "format": "csv"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["rendered_text"] is not None
    assert "metric,value" in body["rendered_text"]


def test_generate_report_endpoint_unknown_entity_returns_404():
    response = client.post(
        "/api/v1/reports/generate",
        json={"report_type": "FINANCIAL_SUMMARY", "entity_id": "Nowhere Land XYZ", "format": "json"},
    )
    assert response.status_code == 404
    assert response.json()["detail"]["error"] == "entity_not_found"


def test_generate_report_endpoint_rejects_invalid_report_type():
    response = client.post("/api/v1/reports/generate", json={"report_type": "NOT_A_TYPE", "format": "json"})
    assert response.status_code == 422


def test_list_subscriptions_endpoint():
    response = client.get("/api/v1/reports/subscriptions")
    assert response.status_code == 200
    ids = {s["subscription_id"] for s in response.json()}
    assert ids == {"DAILY_ANOMALY_PULSE_LOG", "WEEKLY_RISK_SCORE_HEATMAP", "MONTHLY_FEDERAL_COST_AUDIT"}


def test_trigger_subscription_endpoint():
    response = client.post("/api/v1/reports/subscriptions/DAILY_ANOMALY_PULSE_LOG/trigger")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "SUCCESS"
    assert len(body["reports"]) == 1


def test_trigger_unknown_subscription_returns_404():
    response = client.post("/api/v1/reports/subscriptions/NOT_REAL/trigger")
    assert response.status_code == 404
    assert response.json()["detail"]["error"] == "unknown_subscription"
