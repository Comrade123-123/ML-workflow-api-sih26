"""API routes for the MPLADS Expenditure Utilization Risk capability.

Isolated from Module 1/2/7/8/9 -- a distinct prefix, distinct router, no
shared endpoint logic. See config.UNDERSPEND_TARGET_DEFINITION and
UNDERSPEND_RISK_DISCLAIMER for the exact terminology this must always
surface: this is a limited statistical signal, not cost-overrun, fraud,
corruption, or misconduct detection.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from mplads.config import UNDERSPEND_RISK_DISCLAIMER, UNDERSPEND_TARGET_DEFINITION
from mplads.underspend_risk.exceptions import ModelNotAvailableError
from mplads.underspend_risk.inference import predict_underspend_risk
from mplads.underspend_risk.registry import is_model_available, load_active_model
from mplads.underspend_risk.schemas import PredictRequest, PredictResponse, StatusResponse

router = APIRouter(prefix="/api/v1/ml", tags=["underspend-risk"])


@router.get("/underspend-risk/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    """Reports whether a trained model is currently available."""
    if is_model_available():
        model = load_active_model()
        return StatusResponse(
            model_available=True,
            model_version=model.model_version if model else None,
            trained_at=model.trained_at if model else None,
            target_definition=UNDERSPEND_TARGET_DEFINITION,
            disclaimer=UNDERSPEND_RISK_DISCLAIMER,
        )

    return StatusResponse(
        model_available=False,
        model_version=None,
        trained_at=None,
        target_definition=UNDERSPEND_TARGET_DEFINITION,
        disclaimer=UNDERSPEND_RISK_DISCLAIMER,
    )


@router.post(
    "/underspend-risk",
    response_model=PredictResponse,
    responses={503: {"description": "No trained underspend-risk model is available."}},
)
def post_predict(request: PredictRequest) -> PredictResponse:
    try:
        return predict_underspend_risk(request)
    except ModelNotAvailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": "model_unavailable", "reason": str(exc)},
        ) from exc
