"""Turns a PredictRequest into the API's PredictResponse contract.

Raises ModelNotAvailableError if no trained model is loaded -- the router
turns that into an HTTP 503. Never fabricates a prediction as a fallback.
"""
from __future__ import annotations

import pandas as pd

from mplads.config import MODULE2_STATUS_LABELS
from mplads.module2_delay_prediction.exceptions import ModelNotAvailableError
from mplads.module2_delay_prediction.model import DelayRiskModel
from mplads.module2_delay_prediction.registry import load_active_model
from mplads.module2_delay_prediction.schemas import PredictRequest, PredictResponse

ON_TRACK, AT_RISK, LIKELY_DELAYED = MODULE2_STATUS_LABELS


def _request_to_frame(request: PredictRequest) -> pd.DataFrame:
    row = {
        "Work Category": request.work_category,
        "Work Type": request.work_type,
        "State": request.state,
        "IDA District": request.ida_district,
        "Recommended Amount (INR)": request.sanctioned_cost,
        "Recommended Date": request.recommended_date,
        "Sanction Date": request.sanction_date,
    }
    return pd.DataFrame([row])


def predict_delay_risk(request: PredictRequest, model: DelayRiskModel | None = None) -> PredictResponse:
    """Predict delay risk status + probability for a single project.

    `model` can be injected for testing; production callers omit it and the
    currently persisted artifact (if any) is loaded from the registry.
    """
    active_model = model if model is not None else load_active_model()
    if active_model is None:
        raise ModelNotAvailableError(
            "No trained delay-risk model is currently available. Run "
            "scripts/train_module2.py to train and persist one -- see "
            "GET /api/v1/module2/status for the target-validation diagnostic."
        )

    features = _request_to_frame(request)
    probabilities = active_model.predict_proba(features).iloc[0]
    status = probabilities.idxmax()
    delay_probability = float(1.0 - probabilities.get(ON_TRACK, 0.0))

    return PredictResponse(
        project_id=request.project_id,
        delay_probability=round(delay_probability, 4),
        status=status,
    )
