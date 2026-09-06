"""Turns a PredictRequest into the API's PredictResponse contract.

Raises ModelNotAvailableError if no legitimately trained model is loaded —
the router is responsible for turning that into an HTTP 503. This module
never fabricates a prediction as a fallback.
"""
from __future__ import annotations

import pandas as pd

from mplads.config import RISK_THRESHOLD_HIGH, RISK_THRESHOLD_MEDIUM
from mplads.module1_cost_overrun.exceptions import ModelNotAvailableError
from mplads.module1_cost_overrun.model import CostOverrunModel
from mplads.module1_cost_overrun.registry import load_active_model
from mplads.module1_cost_overrun.schemas import PredictRequest, PredictResponse


def _risk_band(overrun_pct: float) -> str:
    if overrun_pct > RISK_THRESHOLD_HIGH:
        return "HIGH"
    if overrun_pct > RISK_THRESHOLD_MEDIUM:
        return "MEDIUM"
    return "LOW"


def _request_to_frame(request: PredictRequest) -> pd.DataFrame:
    row = {
        "Work Category": request.work_category,
        "Work Type": request.work_type,
        "State": request.state,
        "IDA District": request.ida_district,
        "Recommended Amount (INR)": request.sanctioned_cost,
        "Recommended Date": request.recommended_date,
        "Sanction Date": request.sanction_date,
        "Completion Date": None,
    }
    return pd.DataFrame([row])


def predict_cost_overrun(request: PredictRequest, model: CostOverrunModel | None = None) -> PredictResponse:
    """Predict final cost / overrun for a single project.

    `model` can be injected for testing; production callers omit it and the
    currently persisted artifact (if any) is loaded from the registry.
    """
    active_model = model if model is not None else load_active_model()
    if active_model is None:
        raise ModelNotAvailableError(
            "No trained cost-overrun model is currently available. The MPLADS "
            "dataset does not yet contain legitimate overrun-labeled examples "
            "required to train Module 1 — see GET /api/v1/module1/status for "
            "the target-validation diagnostic."
        )

    features = _request_to_frame(request)
    predicted_final_cost = float(active_model.predict(features).iloc[0])
    sanctioned_cost = request.sanctioned_cost
    overrun_pct = (predicted_final_cost - sanctioned_cost) / sanctioned_cost * 100.0

    return PredictResponse(
        project_id=request.project_id,
        predicted_final_cost=predicted_final_cost,
        sanctioned_cost=sanctioned_cost,
        overrun_pct=round(overrun_pct, 4),
        risk=_risk_band(overrun_pct),
    )
