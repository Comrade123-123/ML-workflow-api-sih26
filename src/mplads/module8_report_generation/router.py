"""API routes for Module 8 -- Report Generation."""
from __future__ import annotations

from typing import List

from fastapi import APIRouter, HTTPException

from mplads.module7_pulse_score.exceptions import EntityNotFoundError
from mplads.module8_report_generation import generator, scheduler
from mplads.module8_report_generation.exceptions import UnknownReportTypeError, UnknownSubscriptionError
from mplads.module8_report_generation.schemas import ReportRequest, ReportResponse, SubscriptionInfo, TriggerResponse

router = APIRouter(prefix="/api/v1/reports", tags=["module8-report-generation"])


@router.post(
    "/generate",
    response_model=ReportResponse,
    responses={404: {"description": "Unknown constituency/MP entity_id."}},
)
def post_generate_report(request: ReportRequest) -> ReportResponse:
    try:
        return generator.generate_report(request.report_type, request.entity_id, request.format)
    except EntityNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"error": "entity_not_found", "reason": str(exc)}) from exc
    except UnknownReportTypeError as exc:
        raise HTTPException(status_code=422, detail={"error": "unknown_report_type", "reason": str(exc)}) from exc


@router.get("/subscriptions", response_model=List[SubscriptionInfo])
def get_subscriptions() -> List[SubscriptionInfo]:
    return scheduler.list_subscriptions()


@router.post(
    "/subscriptions/{subscription_id}/trigger",
    response_model=TriggerResponse,
    responses={404: {"description": "Unknown subscription_id."}},
)
def post_trigger_subscription(subscription_id: str) -> TriggerResponse:
    try:
        return scheduler.run_subscription(subscription_id)
    except UnknownSubscriptionError as exc:
        raise HTTPException(status_code=404, detail={"error": "unknown_subscription", "reason": str(exc)}) from exc
