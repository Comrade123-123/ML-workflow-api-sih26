import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mplads.api.main import app
from mplads.config import WORKS_RECOMMENDED_CSV
from mplads.module7_pulse_score import service

client = TestClient(app)


@pytest.fixture
def real_constituency():
    df = pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)
    return df.iloc[0]["Constituency"]


@pytest.fixture(autouse=True)
def isolated_history(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "MODULE7_HISTORY_FILE", tmp_path / "pulse_history.json")


def test_pulse_score_endpoint_returns_contract_shape(real_constituency):
    response = client.get(f"/api/v1/pulse-score/{real_constituency}")
    assert response.status_code == 200

    body = response.json()
    assert set(body.keys()) == {
        "entity_id",
        "pulse_score",
        "delta_vs_last_period",
        "fund_utilization_rate_pct",
        "total_works",
        "completed_works",
        "delayed_works_count",
        "health_band",
    }
    assert body["entity_id"] == real_constituency
    assert body["health_band"] in {"EXCELLENT", "GOOD", "NEEDS_ATTENTION", "CRITICAL"}


def test_pulse_score_endpoint_unknown_entity_returns_404():
    response = client.get("/api/v1/pulse-score/Definitely-Not-A-Real-Place")
    assert response.status_code == 404
    body = response.json()
    assert body["detail"]["error"] == "entity_not_found"
