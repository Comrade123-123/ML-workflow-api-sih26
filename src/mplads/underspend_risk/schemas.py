"""Pydantic request/response contracts for the MPLADS Expenditure
Utilization Risk ("underspend_risk") capability.

Field set matches exactly the SAFE (prediction-point-available) features
documented in OPTION2_MPLADS_TARGET_FEASIBILITY_REPORT.md section 8. There
is deliberately no field for Completion Date, Amount Disbursed, or any other
post-outcome value -- those define the target and cannot be accepted as
input even by mistake.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

RiskLevel = Literal["TYPICAL", "ELEVATED"]


class PredictRequest(BaseModel):
    """Features for a single work, as of its Sanction Date (the documented
    prediction point -- see OPTION2_MPLADS_TARGET_FEASIBILITY_REPORT.md
    section 8). A work must already be sanctioned (and therefore have a real
    Work ID -- see that report's section 4 on the "NA-" placeholder rows)
    before a prediction is meaningful.
    """

    work_id: str = Field(..., description="Work ID (assigned at sanction)")
    recommended_amount: float = Field(..., gt=0, description="'Recommended Amount (INR)' column")
    work_category: str = Field(..., description="'Work Category' column")
    work_type: str = Field(..., description="'Work Type' column -- also used for the historical peer-rate lookup")
    state: str = Field(..., description="'State' column -- also used for the historical peer-rate lookup")
    recommended_date: str = Field(..., description="ISO date, 'Recommended Date' column")
    sanction_date: str = Field(..., description="ISO date, 'Sanction Date' column -- the prediction point")


class PredictResponse(BaseModel):
    """Response contract per the Option 2/2A retained-capability
    specification. `risk_score` is the model's raw predicted probability of
    has_underspend; `risk_level` is derived from that same score using the
    ONE decision threshold actually evaluated in the validated research
    (UNDERSPEND_RISK_DECISION_THRESHOLD) -- no additional untested tiers."""

    work_id: str
    risk_score: float = Field(..., ge=0, le=1, description="Predicted probability of has_underspend")
    risk_level: RiskLevel
    model_version: str
    target_definition: str
    disclaimer: str


class ModelUnavailableResponse(BaseModel):
    error: Literal["model_unavailable"] = "model_unavailable"
    reason: str
    detail: Optional[dict] = None


class StatusResponse(BaseModel):
    model_available: bool
    model_version: Optional[str] = None
    trained_at: Optional[str] = None
    target_definition: str
    disclaimer: str
