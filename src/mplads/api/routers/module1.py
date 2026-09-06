"""API routes for Module 1 — Cost Overrun Detection."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from mplads.config import MIN_OVERRUN_RATIO, MIN_OVERRUN_SAMPLES
from mplads.module1_cost_overrun.data_validation import load_joined_training_frame
from mplads.module1_cost_overrun.exceptions import DataValidationError, ModelNotAvailableError
from mplads.module1_cost_overrun.inference import predict_cost_overrun
from mplads.module1_cost_overrun.registry import is_model_available, load_active_model
from mplads.module1_cost_overrun.schemas import PredictRequest, PredictResponse, StatusResponse
from mplads.module1_cost_overrun.target_validation import try_validate_overrun_target

router = APIRouter(prefix="/api/v1/module1", tags=["module1-cost-overrun"])


@router.get("/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    """Reports whether a trained model exists, and if not, exactly why —
    via the same target-validation diagnostic the training script uses."""
    target_validation = None
    try:
        frame = load_joined_training_frame()
        target_validation = try_validate_overrun_target(
            frame, min_positive_examples=MIN_OVERRUN_SAMPLES, min_positive_ratio=MIN_OVERRUN_RATIO
        )
    except DataValidationError:
        target_validation = None

    if is_model_available():
        model = load_active_model()
        return StatusResponse(
            model_available=True,
            model_version=model.model_version if model else None,
            trained_at=model.trained_at if model else None,
            target_validation=target_validation,
        )

    return StatusResponse(
        model_available=False,
        model_version=None,
        trained_at=None,
        target_validation=target_validation,
    )


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={503: {"description": "No trained cost-overrun model is available."}},
)
def post_predict(request: PredictRequest) -> PredictResponse:
    try:
        return predict_cost_overrun(request)
    except ModelNotAvailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": "model_unavailable", "reason": str(exc)},
        ) from exc
