"""Pydantic request/response contracts for Module 1B (AI-Powered Project
Cost & Risk Assessment).

Field names are a superset of the existing underspend_risk.PredictRequest
(see src/mplads/underspend_risk/schemas.py) -- reusing that schema's actual
field semantics rather than inventing incompatible names, per the existing
model's documented SAFE/prediction-point contract. Extra fields here
(work_description, constituency, sanctioned_amount) are accepted for
context/evidence purposes only; only the fields the underspend model
actually consumes are forwarded to it.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

OverallRiskLevel = Literal["LOW", "MEDIUM", "HIGH"]
ConfidenceLevel = Literal["LOW", "MEDIUM", "HIGH"]


class AssessRequest(BaseModel):
    """Project information for a single work, as of its Sanction Date --
    the same prediction point as the existing underspend_risk model (a work
    must already be sanctioned before assessment is meaningful)."""

    project_id: str = Field(..., description="Work ID (assigned at sanction)")
    work_category: str = Field(..., description="'Work Category' column")
    work_type: str = Field(..., description="'Work Type' column")
    work_description: Optional[str] = Field(None, description="Free-text 'Work Description' -- context only, not a model feature")
    state: str = Field(..., description="'State' column")
    constituency: Optional[str] = Field(None, description="'Constituency' column -- context only")
    recommended_amount: float = Field(..., gt=0, description="'Recommended Amount (INR)' column")
    sanctioned_amount: Optional[float] = Field(
        None, gt=0,
        description=(
            "'Sanction Amount (INR)' if known. Research finding (see "
            "scripts/esakshi_costoverrun_research/): on the official MPLADS "
            "eSAKSHI portal this value has been found to equal the "
            "recommended amount in every case checked -- provided here for "
            "completeness/evidence transparency, not used as a separate "
            "model feature."
        ),
    )
    recommendation_date: str = Field(..., description="ISO date, 'Recommended Date' column")
    sanction_date: str = Field(..., description="ISO date, 'Sanction Date' column -- the prediction point")


class UnderspendSignal(BaseModel):
    """The existing underspend_risk model's output, passed through
    unmodified -- Module 1B does not retrain, adjust, or reinterpret this
    score as a cost-overrun probability."""

    risk_score: float = Field(..., ge=0, le=1)
    risk_level: Literal["TYPICAL", "ELEVATED"]
    model_version: str
    target_definition: str


class ComparableProject(BaseModel):
    """A historical completed work used as evidence. Every field here is an
    actual value from data/mplads_option2/datasets/work_level_completed_dataset.csv
    -- nothing here is generated or estimated."""

    work_id: str
    work_category: str
    work_type: str
    work_description: Optional[str] = None
    state: str
    recommended_amount: float
    sanction_date: Optional[str] = None
    completion_date: Optional[str] = None
    amount_disbursed: Optional[float] = None
    underspend_pct: Optional[float] = None


class HistoricalEvidence(BaseModel):
    comparable_projects: List[ComparableProject]
    match_criteria: str = Field(..., description="Plain description of how comparables were selected, for transparency")


class AIAssessment(BaseModel):
    """Gemini's structured output. Validated with Pydantic -- if Gemini's
    response does not conform to this shape, the router surfaces an honest
    503 rather than a partially-fabricated result."""

    overall_risk_level: OverallRiskLevel
    executive_summary: str
    key_risk_factors: List[str]
    underspend_signal_interpretation: str = Field(..., description="Gemini's plain-language interpretation of the underspend signal -- not a new number")
    historical_comparison: List[str]
    data_quality_observations: List[str]
    recommended_review_actions: List[str]
    confidence: ConfidenceLevel
    disclaimer: str


class ModelInfo(BaseModel):
    module: Literal["module1b"] = "module1b"
    name: Literal["AI-Powered Project Cost & Risk Assessment"] = "AI-Powered Project Cost & Risk Assessment"
    provider: Literal["Google Gemini"] = "Google Gemini"
    gemini_model: str


class AssessResponse(BaseModel):
    project_id: str
    model: ModelInfo
    quantitative_signal: UnderspendSignal
    historical_evidence: HistoricalEvidence
    ai_assessment: AIAssessment
    disclaimer: str = Field(
        ...,
        description="Top-level disclaimer, always present regardless of where in the response a reader looks",
    )


class StatusResponse(BaseModel):
    underspend_model_available: bool
    gemini_api_key_configured: bool
    gemini_model: str
    module: Literal["module1b"] = "module1b"
