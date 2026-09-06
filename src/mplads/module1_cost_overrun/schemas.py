"""Pydantic request/response contracts for Module 1 (Cost Overrun Detection)."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Features for a single project, matching the actual source-CSV columns.

    Field names intentionally mirror the CSV columns' meaning (not their raw
    "Column Name (INR)" spelling) since this is a JSON API contract.
    """

    project_id: str = Field(..., description="Work ID / project identifier")
    sanctioned_cost: float = Field(..., gt=0, description="Recommended Amount (INR) at sanction")
    work_category: str = Field(..., description="Works_* 'Work Category' column")
    work_type: Optional[str] = Field(None, description="Works_* 'Work Type' column")
    state: str = Field(..., description="'State' column")
    ida_district: Optional[str] = Field(None, description="'IDA District' column")
    recommended_date: Optional[str] = Field(None, description="ISO date, 'Recommended Date' column")
    sanction_date: Optional[str] = Field(None, description="ISO date, 'Sanction Date' column")
    elapsed_days_since_sanction: Optional[int] = Field(
        None, ge=0, description="Days elapsed since sanction, for an in-progress project"
    )


class PredictResponse(BaseModel):
    """Matches the project-specification output contract exactly."""

    project_id: str
    predicted_final_cost: float
    sanctioned_cost: float
    overrun_pct: float
    risk: Literal["LOW", "MEDIUM", "HIGH"]


class ModelUnavailableResponse(BaseModel):
    """Returned (HTTP 503) instead of a fabricated prediction when no valid
    trained model exists."""

    error: Literal["model_unavailable"] = "model_unavailable"
    reason: str
    detail: Optional[dict] = None


class TargetValidationReport(BaseModel):
    """Diagnostic summary produced by the target-validation gate, surfaced
    via /status so callers understand *why* no model is available."""

    is_valid: bool
    total_examples: int
    positive_overrun_examples: int
    positive_overrun_ratio: float
    min_required_samples: int
    min_required_ratio: float
    message: str


class StatusResponse(BaseModel):
    model_available: bool
    model_version: Optional[str] = None
    trained_at: Optional[str] = None
    target_validation: Optional[TargetValidationReport] = None
