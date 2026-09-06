"""Full Module 1B pipeline tests. Uses the REAL, already-persisted
underspend_risk model (never retrained or modified here) but always injects
a fake Gemini `call_fn` -- no real Gemini API call is ever made."""
import json

import pytest

from mplads.module1b_gemini_risk.assessment import assess_project
from mplads.module1b_gemini_risk.exceptions import UnderspendModelUnavailableError
from mplads.module1b_gemini_risk.schemas import AssessRequest

VALID_ASSESSMENT_JSON = {
    "overall_risk_level": "MEDIUM",
    "executive_summary": "Moderate underspend-risk signal for this work category and state.",
    "key_risk_factors": ["Elevated historical underspend rate for this State/Work Type combination."],
    "underspend_signal_interpretation": "Historical pattern-based signal, not a cost-overrun probability.",
    "historical_comparison": ["Similar recommended amounts observed among comparable completed works."],
    "data_quality_observations": ["No anomalies observed in the supplied evidence."],
    "recommended_review_actions": ["Routine monitoring recommended."],
    "confidence": "MEDIUM",
    "disclaimer": "Not a cost-overrun prediction; not evidence of wrongdoing.",
}


def _fake_gemini_call(prompt, api_key, model_name):
    return json.dumps(VALID_ASSESSMENT_JSON)


def _sample_request(**overrides):
    payload = dict(
        project_id="WS/MP620/2024-2025/133166",
        work_category="Normal/Others",
        work_type="Construction of buildings for community cultural activities",
        state="Karnataka",
        recommended_amount=497185.0,
        recommendation_date="2024-07-08",
        sanction_date="2024-07-09",
    )
    payload.update(overrides)
    return AssessRequest(**payload)


def test_full_pipeline_produces_a_complete_assessment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-testing")
    response = assess_project(_sample_request(), call_fn=_fake_gemini_call)

    assert response.project_id == "WS/MP620/2024-2025/133166"
    assert response.model.module == "module1b"
    assert response.model.provider == "Google Gemini"
    assert 0.0 <= response.quantitative_signal.risk_score <= 1.0
    assert response.quantitative_signal.risk_level in ("TYPICAL", "ELEVATED")
    assert response.ai_assessment.overall_risk_level == "MEDIUM"
    assert isinstance(response.historical_evidence.comparable_projects, list)
    assert "cost-overrun" in response.disclaimer.lower() or "cost overrun" in response.disclaimer.lower()


def test_evidence_package_separates_observed_model_and_historical_data(monkeypatch):
    """Inspects the evidence package actually built and handed to Gemini."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-testing")
    captured = {}

    def capturing_call(prompt, api_key, model_name):
        captured["prompt"] = prompt
        return json.dumps(VALID_ASSESSMENT_JSON)

    assess_project(_sample_request(), call_fn=capturing_call)

    evidence_text = captured["prompt"]
    assert '"project"' in evidence_text
    assert '"underspend_model"' in evidence_text
    assert '"historical_context"' in evidence_text
    assert "not a cost-overrun probability" in evidence_text.lower() or "not a cost-overrun probability" in evidence_text


def test_underspend_model_unavailable_raises_before_calling_gemini(monkeypatch):
    """If the existing underspend model is unavailable, Module 1B must fail
    honestly and must NOT call Gemini at all."""
    import mplads.underspend_risk.inference as underspend_inference

    monkeypatch.setattr(underspend_inference, "load_active_model", lambda: None)

    def should_not_be_called(prompt, api_key, model_name):
        raise AssertionError("Gemini must not be called when the underspend model is unavailable")

    with pytest.raises(UnderspendModelUnavailableError):
        assess_project(_sample_request(), call_fn=should_not_be_called)


def test_comparable_projects_are_included_for_a_known_category(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-testing")
    response = assess_project(_sample_request(), call_fn=_fake_gemini_call)
    assert len(response.historical_evidence.comparable_projects) > 0
