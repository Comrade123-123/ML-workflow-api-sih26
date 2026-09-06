from fastapi.testclient import TestClient

from mplads.api.main import app
from mplads.underspend_risk.registry import is_model_available

client = TestClient(app)


def test_status_endpoint_reports_model_available_after_training():
    """A legitimate artifact has been trained and persisted (see
    scripts/train_underspend_risk.py) -- the status endpoint must reflect
    that honestly, with the exact same target/disclaimer text regardless of
    availability."""
    assert is_model_available() is True  # precondition: artifact trained and persisted in this repo

    response = client.get("/api/v1/ml/underspend-risk/status")
    assert response.status_code == 200

    body = response.json()
    assert body["model_available"] is True
    assert body["model_version"] is not None
    assert body["trained_at"] is not None
    assert "has_underspend" in body["target_definition"]
    assert "NOT" in body["disclaimer"]


def test_predict_endpoint_returns_real_prediction_not_fabricated():
    """With a real artifact persisted, the endpoint must return an actual
    model-computed score -- not a hardcoded/fabricated value -- via the
    real production inference path (no test-only model injection here)."""
    payload = {
        "work_id": "WS/MP620/2024-2025/133166",
        "recommended_amount": 497185.0,
        "work_category": "Normal/Others",
        "work_type": "Construction of buildings for community cultural activities",
        "state": "Karnataka",
        "recommended_date": "2024-07-08",
        "sanction_date": "2024-07-09",
    }
    response = client.post("/api/v1/ml/underspend-risk", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["work_id"] == payload["work_id"]
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_level"] in ("TYPICAL", "ELEVATED")
    assert body["model_version"] is not None
    assert "has_underspend" in body["target_definition"]
    assert "NOT" in body["disclaimer"]


def test_predict_endpoint_returns_503_when_model_unavailable(monkeypatch):
    """Simulates unavailability via dependency injection (monkeypatching the
    registry lookup the router depends on) rather than relying on ambient
    repo state -- the repo now legitimately has a trained artifact, so this
    is the correct way to keep exercising the "no fabricated prediction"
    guarantee without deleting the real artifact."""
    import mplads.underspend_risk.inference as inference_module

    monkeypatch.setattr(inference_module, "load_active_model", lambda: None)

    payload = {
        "work_id": "WS/MP1/2024-2025/000001",
        "recommended_amount": 500_000,
        "work_category": "Normal/Others",
        "work_type": "Roads",
        "state": "Bihar",
        "recommended_date": "2024-07-08",
        "sanction_date": "2024-07-20",
    }
    response = client.post("/api/v1/ml/underspend-risk", json=payload)

    assert response.status_code == 503
    body = response.json()
    assert body["detail"]["error"] == "model_unavailable"
    assert "reason" in body["detail"]
    # ensure no prediction fields leaked into an error response
    assert "risk_score" not in body


def test_predict_endpoint_rejects_malformed_request_missing_required_fields():
    response = client.post("/api/v1/ml/underspend-risk", json={"work_id": "WS/MP1/2024-2025/000001"})
    assert response.status_code == 422


def test_predict_endpoint_rejects_non_positive_recommended_amount():
    payload = {
        "work_id": "WS/MP1/2024-2025/000001",
        "recommended_amount": -100,
        "work_category": "Normal/Others",
        "work_type": "Roads",
        "state": "Bihar",
        "recommended_date": "2024-07-08",
        "sanction_date": "2024-07-20",
    }
    response = client.post("/api/v1/ml/underspend-risk", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_rejects_wrong_types():
    payload = {
        "work_id": "WS/MP1/2024-2025/000001",
        "recommended_amount": "not-a-number",
        "work_category": "Normal/Others",
        "work_type": "Roads",
        "state": "Bihar",
        "recommended_date": "2024-07-08",
        "sanction_date": "2024-07-20",
    }
    response = client.post("/api/v1/ml/underspend-risk", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_rejects_empty_body():
    response = client.post("/api/v1/ml/underspend-risk", json={})
    assert response.status_code == 422


def test_health_endpoint_still_works():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_existing_module1_endpoints_unaffected():
    """Scope-control regression check: adding this router must not disturb
    Module 1's existing routes."""
    response = client.get("/api/v1/module1/status")
    assert response.status_code == 200


def test_existing_module2_endpoints_unaffected():
    response = client.post("/api/v1/module2/predict", json={"project_id": "x"})
    assert response.status_code == 422  # still validates the same way as before, just reachable
