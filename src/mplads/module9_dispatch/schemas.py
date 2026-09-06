"""Pydantic contracts for Module 9 (Alert Generation & Dispatch)."""
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"]


class EventType(str, Enum):
    ALERT_TRIGGERED = "ALERT_TRIGGERED"
    INSPECTOR_DISPATCHED = "INSPECTOR_DISPATCHED"


Actor = Literal["system_monitor", "ai_agent", "authority_officer"]


class AuditLogEntry(BaseModel):
    log_id: str
    timestamp: str
    project_id: str
    event_type: EventType
    actor: Actor
    details: Dict[str, Any]


class AlertEvaluateRequest(BaseModel):
    project_id: str
    external_risk_signals: Optional[Dict[str, RiskLevel]] = Field(
        default=None,
        description="Optional precomputed risk levels from teammate Modules 3-6, keyed by module name.",
    )


class AlertEvaluateResponse(BaseModel):
    project_id: str
    alert_triggered: bool
    reasons: List[str]
    constituency: str
    delay_probability: Optional[float] = None
    delay_status: Optional[str] = None
    pulse_score: float
    health_band: str
    audit_log_id: Optional[str] = None


class InspectorStatus(str, Enum):
    AVAILABLE = "AVAILABLE"
    ASSIGNED = "ASSIGNED"


class CurrentAssignment(BaseModel):
    project_id: str
    inspection_date: str
    notes: Optional[str] = None
    assigned_at: str


class InspectorInfo(BaseModel):
    inspector_id: str
    name: str
    assigned_state_or_district: str
    status: InspectorStatus
    current_assignment: Optional[CurrentAssignment] = None


class InspectorSuggestionResponse(BaseModel):
    project_id: str
    constituency: str
    state: str
    eligible_inspectors: List[InspectorInfo]


class DispatchRequest(BaseModel):
    project_id: str
    inspector_id: str
    inspection_date: str
    notes: Optional[str] = None
    actor: Actor = "authority_officer"


class DispatchResponse(BaseModel):
    inspector: InspectorInfo
    audit_log: AuditLogEntry
