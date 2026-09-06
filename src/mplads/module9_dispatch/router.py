"""API routes for Module 9 -- Alert Generation & Field Inspector Dispatch."""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from mplads.module7_pulse_score.exceptions import EntityNotFoundError
from mplads.module9_dispatch import alert_engine, audit_logger, inspector_service
from mplads.module9_dispatch.exceptions import (
    InspectorNotAvailableError,
    InspectorNotFoundError,
    ProjectNotFoundError,
)
from mplads.module9_dispatch.schemas import (
    AlertEvaluateRequest,
    AlertEvaluateResponse,
    AuditLogEntry,
    DispatchRequest,
    DispatchResponse,
    InspectorInfo,
    InspectorSuggestionResponse,
)

router = APIRouter(prefix="/api/v1", tags=["module9-dispatch"])


@router.post(
    "/alerts/evaluate",
    response_model=AlertEvaluateResponse,
    responses={404: {"description": "Unknown or ambiguous project_id."}},
)
def post_evaluate_alert(request: AlertEvaluateRequest) -> AlertEvaluateResponse:
    try:
        return alert_engine.evaluate_alert(request.project_id, request.external_risk_signals)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"error": "project_not_found", "reason": str(exc)}) from exc
    except EntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"error": "entity_not_found", "reason": str(exc)}) from exc


@router.get("/alerts/audit-trail", response_model=List[AuditLogEntry])
def get_audit_trail(
    project_id: Optional[str] = Query(default=None),
    actor: Optional[str] = Query(default=None),
) -> List[AuditLogEntry]:
    return audit_logger.read_audit_trail(project_id=project_id, actor=actor)


@router.get("/inspectors", response_model=List[InspectorInfo])
def get_inspectors() -> List[InspectorInfo]:
    return inspector_service.list_inspectors()


@router.get(
    "/inspectors/suggest/{project_id:path}",
    response_model=InspectorSuggestionResponse,
    responses={404: {"description": "Unknown or ambiguous project_id."}},
)
def get_suggested_inspectors(project_id: str) -> InspectorSuggestionResponse:
    try:
        return inspector_service.suggest_inspectors(project_id)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"error": "project_not_found", "reason": str(exc)}) from exc


@router.post(
    "/inspectors/dispatch",
    response_model=DispatchResponse,
    responses={
        404: {"description": "Unknown/ambiguous project_id or unknown inspector_id."},
        409: {"description": "Inspector is not AVAILABLE."},
    },
)
def post_dispatch_inspector(request: DispatchRequest) -> DispatchResponse:
    try:
        return inspector_service.dispatch_inspector(request)
    except ProjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"error": "project_not_found", "reason": str(exc)}) from exc
    except InspectorNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"error": "inspector_not_found", "reason": str(exc)}) from exc
    except InspectorNotAvailableError as exc:
        raise HTTPException(status_code=409, detail={"error": "inspector_not_available", "reason": str(exc)}) from exc
