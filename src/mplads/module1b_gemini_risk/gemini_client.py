"""Gemini API client for Module 1B.

Uses the current official Google GenAI SDK (`google-genai`, `from google
import genai`) -- verified against official Google documentation as the
current recommended package; the older `google-generativeai` package is
legacy. See https://ai.google.dev/gemini-api/docs/libraries.

The API key is read ONLY from the GEMINI_API_KEY environment variable --
never hardcoded, never accepted as a request parameter, never logged, and
never echoed back in any response.

The actual network call is isolated in `_call_gemini_raw` so it can be
monkeypatched/injected in tests -- no test in this repository makes a real
Gemini API call.
"""
from __future__ import annotations

import json
import os

from pydantic import ValidationError

from mplads.module1b_gemini_risk.exceptions import GeminiResponseError, GeminiUnavailableError
from mplads.module1b_gemini_risk.schemas import AIAssessment

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"

SYSTEM_INSTRUCTION = (
    "You are an explainable project-risk assessment layer for a government "
    "infrastructure monitoring system (MPLADS). You are given a structured "
    "evidence package containing: observed project data, an existing "
    "machine-learning underspend-risk signal, and historical comparable "
    "projects. "
    "Assess project cost and execution risk using ONLY the supplied "
    "evidence. Do not invent facts, historical projects, costs, or "
    "government rules. Treat the existing ML score strictly as an "
    "underspend-risk signal -- it is NOT a cost-overrun probability, and "
    "you must not convert it into one or claim any probability of cost "
    "overrun. Do not predict or state a final project cost. Do not claim "
    "fraud, corruption, or misconduct based on this data -- at most, note "
    "that something is unusual and would benefit from human review. "
    "Respond with ONLY a single JSON object matching exactly this shape "
    "(no markdown fences, no prose outside the JSON): "
    '{"overall_risk_level": "LOW|MEDIUM|HIGH", "executive_summary": string, '
    '"key_risk_factors": [string, ...], '
    '"underspend_signal_interpretation": string, '
    '"historical_comparison": [string, ...], '
    '"data_quality_observations": [string, ...], '
    '"recommended_review_actions": [string, ...], '
    '"confidence": "LOW|MEDIUM|HIGH", "disclaimer": string}'
)


def _get_api_key() -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GeminiUnavailableError("GEMINI_API_KEY environment variable is not set.")
    return api_key


def get_model_name() -> str:
    return os.environ.get("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)


def _call_gemini_raw(prompt: str, api_key: str, model_name: str) -> str:
    """The only function in this module that talks to the network. Kept
    tiny and isolated so tests can monkeypatch it directly instead of
    mocking the SDK's internals."""
    from google import genai  # imported lazily so the package is only required at call time

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=[SYSTEM_INSTRUCTION, prompt],
    )
    text = response.text
    if not text:
        raise GeminiUnavailableError("Gemini returned an empty response.")
    return text


def _strip_markdown_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        lines = lines[1:] if lines else lines
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines)
    return stripped.strip()


def get_ai_assessment(evidence_package: dict, call_fn=None) -> AIAssessment:
    """Builds the prompt from the evidence package, calls Gemini, and
    validates the response into an AIAssessment.

    `call_fn` can be injected for testing (signature: (prompt, api_key,
    model_name) -> str); production callers omit it and the real
    `_call_gemini_raw` (network call) is used.

    Raises GeminiUnavailableError for missing key / network / API failures,
    and GeminiResponseError for a reachable-but-unusable response (not JSON,
    or JSON that fails schema validation). Never returns a fabricated
    assessment.
    """
    api_key = _get_api_key()
    model_name = get_model_name()
    call = call_fn if call_fn is not None else _call_gemini_raw

    prompt = (
        "Evidence package (JSON):\n"
        + json.dumps(evidence_package, indent=2, default=str)
        + "\n\nProduce the structured JSON assessment now."
    )

    try:
        raw_text = call(prompt, api_key, model_name)
    except GeminiUnavailableError:
        raise
    except Exception as exc:  # noqa: BLE001 -- any SDK/network failure is treated as provider-unavailable
        raise GeminiUnavailableError(f"Gemini API call failed: {exc!r}") from exc

    cleaned = _strip_markdown_fences(raw_text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise GeminiResponseError(f"Gemini response was not valid JSON: {exc}") from exc

    try:
        return AIAssessment.model_validate(parsed)
    except ValidationError as exc:
        raise GeminiResponseError(f"Gemini response did not match the expected schema: {exc}") from exc
