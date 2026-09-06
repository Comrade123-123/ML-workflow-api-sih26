"""API routes for Module 2 -- Delayed Project Prediction."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from mplads.config import MODULE2_MIN_CLASS_SAMPLES
from mplads.module2_delay_prediction.data_validation import load_joined_training_frame
from mplads.module2_delay_prediction.exceptions import DataValidationError, ModelNotAvailableError
from mplads.module2_delay_prediction.inference import predict_delay_risk
from mplads.module2_delay_prediction.labeling import assign_status_labels
from mplads.module2_delay_prediction.registry import is_model_available, load_active_model
from mplads.module2_delay_prediction.schemas import PredictRequest, PredictResponse, StatusResponse
from mplads.module2_delay_prediction.target_validation import try_validate_class_balance

router = APIRouter(prefix="/api/v1/module2", tags=["module2-delay-prediction"])


@router.get("/status", response_model=StatusResponse)
def get_status() -> StatusResponse:
    """Reports whether a trained model exists, and the class-balance
    diagnostic from the peer-relative labeling gate."""
    target_validation = None
    try:
        frame = load_joined_training_frame()
        labeled = assign_status_labels(frame)
        target_validation = try_validate_class_balance(labeled, min_samples_per_class=MODULE2_MIN_CLASS_SAMPLES)
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
    responses={503: {"description": "No trained delay-risk model is available."}},
)
def post_predict(request: PredictRequest) -> PredictResponse:
    try:
        return predict_delay_risk(request)
    except ModelNotAvailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": "model_unavailable", "reason": str(exc)},
        ) from exc
