"""Gemini client tests -- NO real network call is ever made here. Every test
either injects a fake `call_fn` or manipulates the GEMINI_API_KEY
environment variable; `_call_gemini_raw` (the only function that touches
the network) is never invoked directly."""
import json

import pytest

from mplads.module1b_gemini_risk.exceptions import GeminiResponseError, GeminiUnavailableError
from mplads.module1b_gemini_risk.gemini_client import get_ai_assessment

VALID_ASSESSMENT_JSON = {
    "overall_risk_level": "MEDIUM",
    "executive_summary": "This project shows a moderate underspend-risk signal.",
    "key_risk_factors": ["State-level historical underspend rate is elevated for Karnataka."],
    "underspend_signal_interpretation": "The model estimates a 53% probability of underspend based on historical patterns.",
    "historical_comparison": ["Comparable works in the same category show similar recommended amounts."],
    "data_quality_observations": ["No unusual data quality issues observed."],
    "recommended_review_actions": ["Standard periodic review recommended."],
    "confidence": "MEDIUM",
    "disclaimer": "This is not a cost-overrun prediction and does not indicate wrongdoing.",
}

EVIDENCE = {"project": {"observed_data": {"project_id": "X"}}, "underspend_model": {}, "historical_context": {}}


def test_successful_response_parses_into_ai_assessment(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-testing")

    def fake_call(prompt, api_key, model_name):
        return json.dumps(VALID_ASSESSMENT_JSON)

    result = get_ai_assessment(EVIDENCE, call_fn=fake_call)
    assert result.overall_risk_level == "MEDIUM"
    assert result.confidence == "MEDIUM"


def test_response_wrapped_in_markdown_fences_is_still_parsed(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-testing")

    def fake_call(prompt, api_key, model_name):
        return "```json\n" + json.dumps(VALID_ASSESSMENT_JSON) + "\n```"

    result = get_ai_assessment(EVIDENCE, call_fn=fake_call)
    assert result.overall_risk_level == "MEDIUM"


def test_malformed_json_raises_gemini_response_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-testing")

    def fake_call(prompt, api_key, model_name):
        return "this is not json at all {{{"

    with pytest.raises(GeminiResponseError):
        get_ai_assessment(EVIDENCE, call_fn=fake_call)


def test_json_missing_required_fields_raises_gemini_response_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-testing")

    def fake_call(prompt, api_key, model_name):
        return json.dumps({"overall_risk_level": "MEDIUM"})  # missing everything else

    with pytest.raises(GeminiResponseError):
        get_ai_assessment(EVIDENCE, call_fn=fake_call)


def test_json_with_invalid_enum_value_raises_gemini_response_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-testing")
    bad = dict(VALID_ASSESSMENT_JSON)
    bad["overall_risk_level"] = "CATASTROPHIC"  # not one of LOW/MEDIUM/HIGH

    def fake_call(prompt, api_key, model_name):
        return json.dumps(bad)

    with pytest.raises(GeminiResponseError):
        get_ai_assessment(EVIDENCE, call_fn=fake_call)


def test_missing_api_key_raises_gemini_unavailable_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def fake_call(prompt, api_key, model_name):
        raise AssertionError("should never be called when the API key is missing")

    with pytest.raises(GeminiUnavailableError):
        get_ai_assessment(EVIDENCE, call_fn=fake_call)


def test_call_failure_raises_gemini_unavailable_error(monkeypatch):
    """Simulates a timeout / API error from the SDK."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-testing")

    def failing_call(prompt, api_key, model_name):
        raise TimeoutError("simulated network timeout")

    with pytest.raises(GeminiUnavailableError):
        get_ai_assessment(EVIDENCE, call_fn=failing_call)


def test_empty_response_raises_gemini_unavailable_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-testing")

    def fake_call(prompt, api_key, model_name):
        raise GeminiUnavailableError("Gemini returned an empty response.")

    with pytest.raises(GeminiUnavailableError):
        get_ai_assessment(EVIDENCE, call_fn=fake_call)


def test_api_key_never_appears_in_exception_message(monkeypatch):
    """Security: the actual key value must never leak into an error message."""
    secret_value = "AQ.SuperSecretKeyValueThatMustNeverLeak12345"
    monkeypatch.setenv("GEMINI_API_KEY", secret_value)

    def failing_call(prompt, api_key, model_name):
        raise TimeoutError("simulated network timeout")

    with pytest.raises(GeminiUnavailableError) as exc_info:
        get_ai_assessment(EVIDENCE, call_fn=failing_call)
    assert secret_value not in str(exc_info.value)
