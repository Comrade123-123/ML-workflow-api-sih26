from fastapi.testclient import TestClient

from mplads.api.main import app
from mplads.module2_delay_prediction.registry import is_model_available

client = TestClient(app)


def test_status_endpoint_reports_target_validation_diagnostic():
    response = client.get("/api/v1/module2/status")
    assert response.status_code == 200

    body = response.json()
    assert body["model_available"] == is_model_available()

    diagnostic = body["target_validation"]
    assert diagnostic is not None
    assert diagnostic["total_examples"] > 0
    assert set(diagnostic["class_counts"].keys()) == {"ON_TRACK", "AT_RISK", "LIKELY_DELAYED"}


def test_predict_endpoint_matches_model_availability():
    payload = {
        "project_id": "AIP-802",
        "sanctioned_cost": 2_000_000,
        "work_category": "Normal/Others",
        "state": "Bihar",
    }
    response = client.post("/api/v1/module2/predict", json=payload)

    if not is_model_available():
        assert response.status_code == 503
        body = response.json()
        assert body["detail"]["error"] == "model_unavailable"
        assert "predicted_final_cost" not in body
        assert "status" not in body
    else:
        assert response.status_code == 200
        body = response.json()
        assert body["project_id"] == "AIP-802"
        assert 0.0 <= body["delay_probability"] <= 1.0
        assert body["status"] in {"ON_TRACK", "AT_RISK", "LIKELY_DELAYED"}


def test_predict_endpoint_rejects_malformed_request():
    response = client.post("/api/v1/module2/predict", json={"project_id": "AIP-802"})
    assert response.status_code == 422
