"""HTTP-level tests for Module 1B. The production code path
(post_assess -> assess_project -> get_ai_assessment -> _call_gemini_raw) is
exercised end-to-end, but `_call_gemini_raw` -- the one function that
touches the network -- is always monkeypatched. No real Gemini API call is
ever made in this test suite."""
import json

from fastapi.testclient import TestClient

from mplads.api.main import app
import mplads.module1b_gemini_risk.gemini_client as gemini_client

client = TestClient(app)

VALID_ASSESSMENT_JSON = {
    "overall_risk_level": "MEDIUM",
    "executive_summary": "Moderate underspend-risk signal.",
    "key_risk_factors": ["Elevated historical rate for this State/Work Type."],
    "underspend_signal_interpretation": "Historical pattern-based signal, not a cost-overrun probability.",
    "historical_comparison": ["Comparable recommended amounts observed."],
    "data_quality_observations": ["No anomalies observed."],
    "recommended_review_actions": ["Routine monitoring recommended."],
    "confidence": "MEDIUM",
    "disclaimer": "Not a cost-overrun prediction; not evidence of wrongdoing.",
}


def _valid_payload(**overrides):
    payload = {
        "project_id": "WS/MP620/2024-2025/133166",
        "work_category": "Normal/Others",
        "work_type": "Construction of buildings for community cultural activities",
        "state": "Karnataka",
        "recommended_amount": 497185.0,
        "recommendation_date": "2024-07-08",
        "sanction_date": "2024-07-09",
    }
    payload.update(overrides)
    return payload


def test_status_endpoint_reports_underspend_and_gemini_availability(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.get("/api/v1/module1b/status")
    assert response.status_code == 200
    body = response.json()
    assert body["module"] == "module1b"
    assert body["underspend_model_available"] is True  # real persisted artifact, per prior sessions
    assert body["gemini_api_key_configured"] is False
    assert "gemini_model" in body


def test_assess_endpoint_returns_full_structured_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-testing")
    monkeypatch.setattr(gemini_client, "_call_gemini_raw", lambda prompt, api_key, model_name: json.dumps(VALID_ASSESSMENT_JSON))

    response = client.post("/api/v1/module1b/assess", json=_valid_payload())
    assert response.status_code == 200
    body = response.json()
    assert body["project_id"] == "WS/MP620/2024-2025/133166"
    assert body["model"]["provider"] == "Google Gemini"
    assert 0.0 <= body["quantitative_signal"]["risk_score"] <= 1.0
    assert body["ai_assessment"]["overall_risk_level"] == "MEDIUM"
    assert "comparable_projects" in body["historical_evidence"]
    assert "disclaimer" in body


def test_assess_endpoint_rejects_malformed_request():
    response = client.post("/api/v1/module1b/assess", json={"project_id": "X"})
    assert response.status_code == 422


def test_assess_endpoint_returns_503_when_gemini_api_key_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    response = client.post("/api/v1/module1b/assess", json=_valid_payload())
    assert response.status_code == 503
    body = response.json()
    assert body["detail"]["error"] == "gemini_unavailable"
    assert "ai_assessment" not in body


def test_assess_endpoint_returns_503_on_malformed_gemini_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-testing")
    monkeypatch.setattr(gemini_client, "_call_gemini_raw", lambda prompt, api_key, model_name: "not json {{{")

    response = client.post("/api/v1/module1b/assess", json=_valid_payload())
    assert response.status_code == 503
    body = response.json()
    assert body["detail"]["error"] == "gemini_response_invalid"


def test_assess_endpoint_returns_503_on_gemini_timeout(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-testing")

    def timeout_call(prompt, api_key, model_name):
        raise TimeoutError("simulated timeout")

    monkeypatch.setattr(gemini_client, "_call_gemini_raw", timeout_call)
    response = client.post("/api/v1/module1b/assess", json=_valid_payload())
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "gemini_unavailable"


def test_assess_endpoint_returns_503_when_underspend_model_unavailable(monkeypatch):
    import mplads.underspend_risk.inference as underspend_inference

    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-testing")
    monkeypatch.setattr(underspend_inference, "load_active_model", lambda: None)

    response = client.post("/api/v1/module1b/assess", json=_valid_payload())
    assert response.status_code == 503
    assert response.json()["detail"]["error"] == "underspend_model_unavailable"


def test_gemini_api_key_never_appears_in_any_response_body(monkeypatch):
    """Security: neither a successful nor a failing response may ever
    contain the actual configured API key value."""
    secret_value = "AQ.SuperSecretKeyValueThatMustNeverLeak98765"
    monkeypatch.setenv("GEMINI_API_KEY", secret_value)
    monkeypatch.setattr(gemini_client, "_call_gemini_raw", lambda prompt, api_key, model_name: json.dumps(VALID_ASSESSMENT_JSON))

    ok_response = client.post("/api/v1/module1b/assess", json=_valid_payload())
    assert secret_value not in ok_response.text

    status_response = client.get("/api/v1/module1b/status")
    assert secret_value not in status_response.text

    monkeypatch.setattr(gemini_client, "_call_gemini_raw", lambda *a: (_ for _ in ()).throw(TimeoutError("x")))
    failing_response = client.post("/api/v1/module1b/assess", json=_valid_payload())
    assert secret_value not in failing_response.text


def test_health_endpoint_still_works():
    response = client.get("/health")
    assert response.status_code == 200


def test_existing_underspend_risk_endpoint_unaffected():
    response = client.get("/api/v1/ml/underspend-risk/status")
    assert response.status_code == 200


def test_existing_module1_endpoint_unaffected():
    response = client.get("/api/v1/module1/status")
    assert response.status_code == 200


def test_existing_module2_endpoint_unaffected():
    response = client.post("/api/v1/module2/predict", json={"project_id": "x"})
    assert response.status_code == 422
