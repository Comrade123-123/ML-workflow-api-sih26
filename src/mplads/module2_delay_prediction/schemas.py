"""Pydantic request/response contracts for Module 2 (Delayed Project Prediction)."""
from typing import Dict, Literal, Optional

from pydantic import BaseModel, Field

StatusLabel = Literal["ON_TRACK", "AT_RISK", "LIKELY_DELAYED"]


class PredictRequest(BaseModel):
    """Features for a single in-progress project.

    Only pre-completion-knowable fields -- there is no final duration input,
    since that is exactly what the model predicts a risk band for.
    """

    project_id: str = Field(..., description="Work ID / project identifier")
    sanctioned_cost: float = Field(..., gt=0, description="Recommended Amount (INR)")
    work_category: str = Field(..., description="'Work Category' column")
    work_type: Optional[str] = Field(None, description="'Work Type' column")
    state: str = Field(..., description="'State' column")
    ida_district: Optional[str] = Field(None, description="'IDA District' column")
    recommended_date: Optional[str] = Field(None, description="ISO date, 'Recommended Date' column")
    sanction_date: Optional[str] = Field(None, description="ISO date, 'Sanction Date' column")


class PredictResponse(BaseModel):
    """Matches the project-specification output contract exactly."""

    project_id: str
    delay_probability: float
    status: StatusLabel


class ModelUnavailableResponse(BaseModel):
    error: Literal["model_unavailable"] = "model_unavailable"
    reason: str
    detail: Optional[dict] = None


class TargetValidationReport(BaseModel):
    """Diagnostic summary from the peer-relative labeling + class-balance gate."""

    is_valid: bool
    total_examples: int
    class_counts: Dict[str, int]
    min_required_samples_per_class: int
    message: str


class StatusResponse(BaseModel):
    model_available: bool
    model_version: Optional[str] = None
    trained_at: Optional[str] = None
    target_validation: Optional[TargetValidationReport] = None
