"""Orchestrates Module 1B: existing underspend-risk model -> historical
comparable projects -> evidence package -> Gemini -> structured response.

Reuses src/mplads/underspend_risk/inference.py's predict_underspend_risk
directly -- does not retrain, duplicate, or reimplement the underspend
model in any way. If that model is unavailable, this module surfaces that
unavailability honestly rather than substituting anything.
"""
from __future__ import annotations

from mplads.module1b_gemini_risk.comparable_projects import find_comparable_projects
from mplads.module1b_gemini_risk.exceptions import UnderspendModelUnavailableError
from mplads.module1b_gemini_risk.gemini_client import get_ai_assessment, get_model_name
from mplads.module1b_gemini_risk.schemas import (
    AssessRequest,
    AssessResponse,
    ComparableProject,
    HistoricalEvidence,
    ModelInfo,
    UnderspendSignal,
)
from mplads.underspend_risk.exceptions import ModelNotAvailableError
from mplads.underspend_risk.inference import predict_underspend_risk
from mplads.underspend_risk.schemas import PredictRequest as UnderspendPredictRequest

MODULE1B_DISCLAIMER = (
    "This is an AI-Powered Project Cost & Risk Assessment. It does NOT "
    "predict a project's final cost and does NOT calculate a cost-overrun "
    "probability -- MPLADS expenditure is structurally capped at the "
    "recommended/sanctioned amount, so cost overrun is not a valid target "
    "for this data (see OPTION2_MPLADS_TARGET_FEASIBILITY_REPORT.md and "
    "scripts/esakshi_costoverrun_research/). The quantitative signal below "
    "is the existing, independently-validated underspend-risk model's "
    "output. The AI assessment is Gemini's structured interpretation of "
    "that signal plus historical comparable projects -- it is explanatory "
    "reasoning over supplied evidence, not an independent prediction, and "
    "must not be treated as proof of wrongdoing or a certainty about "
    "future project outcomes."
)


def _build_underspend_request(request: AssessRequest) -> UnderspendPredictRequest:
    return UnderspendPredictRequest(
        work_id=request.project_id,
        recommended_amount=request.recommended_amount,
        work_category=request.work_category,
        work_type=request.work_type,
        state=request.state,
        recommended_date=request.recommendation_date,
        sanction_date=request.sanction_date,
    )


def _build_evidence_package(request: AssessRequest, signal: UnderspendSignal, evidence: HistoricalEvidence) -> dict:
    """Structured evidence object handed to Gemini. Clearly separates
    observed data, model-derived indicators, and historical comparisons --
    Gemini's own output is the only AI-generated part, added later."""
    return {
        "project": {
            "observed_data": {
                "project_id": request.project_id,
                "work_category": request.work_category,
                "work_type": request.work_type,
                "work_description": request.work_description,
                "state": request.state,
                "constituency": request.constituency,
                "recommended_amount": request.recommended_amount,
                "sanctioned_amount": request.sanctioned_amount,
                "recommendation_date": request.recommendation_date,
                "sanction_date": request.sanction_date,
            }
        },
        "underspend_model": {
            "source": "existing validated src/mplads/underspend_risk model -- not modified or retrained by Module 1B",
            "risk_score": signal.risk_score,
            "risk_level": signal.risk_level,
            "model_version": signal.model_version,
            "target_definition": signal.target_definition,
            "interpretation_note": (
                "This is a probability of historical underspend (disbursing less than the "
                "recommended amount), NOT a cost-overrun probability. Cost overrun does not "
                "occur in this data (expenditure is capped at the recommended/sanctioned amount)."
            ),
        },
        "historical_context": {
            "match_criteria": evidence.match_criteria,
            "comparable_projects": [c.model_dump() for c in evidence.comparable_projects],
        },
    }


def assess_project(request: AssessRequest, call_fn=None) -> AssessResponse:
    """Full Module 1B pipeline for a single project. `call_fn` is only for
    injecting a fake Gemini call in tests -- production callers omit it."""
    underspend_request = _build_underspend_request(request)
    try:
        underspend_response = predict_underspend_risk(underspend_request)
    except ModelNotAvailableError as exc:
        raise UnderspendModelUnavailableError(str(exc)) from exc

    signal = UnderspendSignal(
        risk_score=underspend_response.risk_score,
        risk_level=underspend_response.risk_level,
        model_version=underspend_response.model_version,
        target_definition=underspend_response.target_definition,
    )

    comparables, match_criteria = find_comparable_projects(
        work_type=request.work_type,
        work_category=request.work_category,
        state=request.state,
        recommended_amount=request.recommended_amount,
    )
    historical_evidence = HistoricalEvidence(comparable_projects=comparables, match_criteria=match_criteria)

    evidence_package = _build_evidence_package(request, signal, historical_evidence)
    ai_assessment = get_ai_assessment(evidence_package, call_fn=call_fn)

    return AssessResponse(
        project_id=request.project_id,
        model=ModelInfo(gemini_model=get_model_name()),
        quantitative_signal=signal,
        historical_evidence=historical_evidence,
        ai_assessment=ai_assessment,
        disclaimer=MODULE1B_DISCLAIMER,
    )
