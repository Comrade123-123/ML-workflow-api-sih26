"""Pydantic contracts for Module 8 (Report Generation)."""
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class ReportType(str, Enum):
    FINANCIAL_SUMMARY = "FINANCIAL_SUMMARY"
    PROJECT_STATUS = "PROJECT_STATUS"
    FUND_UTILIZATION = "FUND_UTILIZATION"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"
    SITE_VERIFICATION = "SITE_VERIFICATION"
    AUDIT_COMPLIANCE = "AUDIT_COMPLIANCE"


ReportFormat = Literal["json", "csv", "markdown"]
Cadence = Literal["DAILY", "WEEKLY", "MONTHLY"]
RunStatus = Literal["SUCCESS", "FAILED"]


class ReportRequest(BaseModel):
    report_type: ReportType
    entity_id: Optional[str] = None
    format: ReportFormat = "json"


class ReportResponse(BaseModel):
    """Consistent envelope for every report type. `data` always carries the
    structured payload ({"summary": {...}, "breakdown": [...]?,
    "placeholders": {...}?}); `rendered_text` is populated only when
    `format` is csv or markdown, so json requests stay lean."""

    report_type: ReportType
    entity_id: Optional[str] = None
    generated_at: str
    format: ReportFormat
    data: Dict[str, Any]
    rendered_text: Optional[str] = None


class SubscriptionInfo(BaseModel):
    subscription_id: str
    name: str
    cadence: Cadence
    report_types: List[ReportType]
    description: str
    last_run_at: Optional[str] = None
    last_run_status: Optional[RunStatus] = None


class TriggerResponse(BaseModel):
    subscription_id: str
    triggered_at: str
    status: RunStatus
    reports: List[ReportResponse]
