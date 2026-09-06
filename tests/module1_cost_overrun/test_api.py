from fastapi.testclient import TestClient

from mplads.api.main import app
from mplads.module1_cost_overrun.registry import is_model_available

client = TestClient(app)


def test_status_endpoint_reports_model_unavailable_with_diagnostic():
    assert is_model_available() is False  # precondition: nothing trained in this repo

    response = client.get("/api/v1/module1/status")
    assert response.status_code == 200

    body = response.json()
    assert body["model_available"] is False
    assert body["model_version"] is None

    diagnostic = body["target_validation"]
    assert diagnostic is not None
    assert diagnostic["is_valid"] is False
    assert diagnostic["positive_overrun_examples"] == 0
    assert diagnostic["total_examples"] > 0


def test_predict_endpoint_returns_503_not_a_fabricated_prediction():
    payload = {
        "project_id": "AIP-802",
        "sanctioned_cost": 2_000_000,
        "work_category": "Normal/Others",
        "state": "Bihar",
    }
    response = client.post("/api/v1/module1/predict", json=payload)

    assert response.status_code == 503
    body = response.json()
    assert body["detail"]["error"] == "model_unavailable"
    assert "reason" in body["detail"]
    # ensure no prediction fields leaked into an error response
    assert "predicted_final_cost" not in body


def test_predict_endpoint_rejects_malformed_request():
    response = client.post("/api/v1/module1/predict", json={"project_id": "AIP-802"})
    assert response.status_code == 422


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
