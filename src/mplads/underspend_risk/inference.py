"""Turns a PredictRequest into the API's PredictResponse contract.

Raises ModelNotAvailableError if no trained model is loaded -- the router
turns that into an HTTP 503. Never fabricates a prediction as a fallback.
"""
from __future__ import annotations

import pandas as pd

from mplads.config import (
    UNDERSPEND_RISK_DECISION_THRESHOLD,
    UNDERSPEND_RISK_DISCLAIMER,
    UNDERSPEND_TARGET_DEFINITION,
)
from mplads.underspend_risk.exceptions import ModelNotAvailableError
from mplads.underspend_risk.model import UnderspendRiskModel
from mplads.underspend_risk.registry import load_active_model
from mplads.underspend_risk.schemas import PredictRequest, PredictResponse


def _request_to_frame(request: PredictRequest) -> pd.DataFrame:
    row = {
        "Work Category": request.work_category,
        "Work Type": request.work_type,
        "State": request.state,
        "Recommended Amount (INR)": request.recommended_amount,
        "Recommended Date": request.recommended_date,
        "Sanction Date": request.sanction_date,
    }
    return pd.DataFrame([row])


def predict_underspend_risk(request: PredictRequest, model: UnderspendRiskModel | None = None) -> PredictResponse:
    """Predict expenditure-utilization risk score for a single work.

    `model` can be injected for testing; production callers omit it and the
    currently persisted artifact (if any) is loaded from the registry.
    """
    active_model = model if model is not None else load_active_model()
    if active_model is None:
        raise ModelNotAvailableError(
            "No trained underspend-risk model is currently available. Run "
            "scripts/train_underspend_risk.py to train and persist one -- "
            "see GET /api/v1/ml/underspend-risk/status."
        )

    features = _request_to_frame(request)
    risk_score = float(active_model.predict_proba(features).iloc[0])
    risk_level = "ELEVATED" if risk_score >= UNDERSPEND_RISK_DECISION_THRESHOLD else "TYPICAL"

    return PredictResponse(
        work_id=request.work_id,
        risk_score=round(risk_score, 4),
        risk_level=risk_level,
        model_version=active_model.model_version or "unknown",
        target_definition=UNDERSPEND_TARGET_DEFINITION,
        disclaimer=UNDERSPEND_RISK_DISCLAIMER,
    )
