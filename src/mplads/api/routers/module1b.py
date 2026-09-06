"""API routes for Module 1B (AI-Powered Project Cost & Risk Assessment).

Isolated from Module 1A (Cost Overrun Detection), the existing
underspend_risk router, and Modules 2/7/8/9 -- a distinct prefix, distinct
router, no shared endpoint logic. This module CALLS the existing
underspend_risk inference internally (see assessment.py) but never
modifies it.
"""
from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException

from mplads.module1b_gemini_risk.assessment import assess_project
from mplads.module1b_gemini_risk.exceptions import (
    GeminiResponseError,
    GeminiUnavailableError,
    UnderspendModelUnavailableError,
)
from mplads.module1b_gemini_risk.gemini_client import get_model_name
from mplads.module1b_gemini_risk.schemas import AssessRequest, AssessResponse, StatusResponse
from mplads.underspend_risk.registry import is_model_available as is_underspend_model_available

router = APIRouter(prefix="/api/v1/module1b", tags=["module1b-gemini-risk"])


@router.get("/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    return StatusResponse(
        underspend_model_available=is_underspend_model_available(),
        gemini_api_key_configured=bool(os.environ.get("GEMINI_API_KEY")),
        gemini_model=get_model_name(),
    )


@router.post(
    "/assess",
    response_model=AssessResponse,
    responses={503: {"description": "Underspend model or Gemini API unavailable."}},
)
def post_assess(request: AssessRequest) -> AssessResponse:
    try:
        return assess_project(request)
    except UnderspendModelUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": "underspend_model_unavailable", "reason": str(exc)},
        ) from exc
    except GeminiUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": "gemini_unavailable", "reason": str(exc)},
        ) from exc
    except GeminiResponseError as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": "gemini_response_invalid", "reason": str(exc)},
        ) from exc
