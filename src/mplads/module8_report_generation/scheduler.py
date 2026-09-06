"""Scheduled report subscriptions for Module 8.

This is a static catalog + manual/on-demand "batch runner" with persisted
run-history -- not an autonomous cron scheduler. The endpoints this module
supports (list + manual trigger, no "enable auto-run" control) don't call
for one, and adding real background scheduling would mean pulling in an
extra dependency (e.g. APScheduler) beyond what's needed here.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List

from mplads.config import MODULE8_SUBSCRIPTION_RUNS_FILE
from mplads.module8_report_generation import generator
from mplads.module8_report_generation.exceptions import UnknownSubscriptionError
from mplads.module8_report_generation.schemas import ReportType, SubscriptionInfo, TriggerResponse

SUBSCRIPTION_CATALOG: Dict[str, Dict[str, Any]] = {
    "DAILY_ANOMALY_PULSE_LOG": {
        "name": "Daily Anomaly & Pulse Log",
        "cadence": "DAILY",
        "report_types": [ReportType.RISK_ASSESSMENT],
        "description": (
            "Daily national snapshot of at-risk/delayed works. The anomaly section is a Module 3 "
            "placeholder until that model is integrated."
        ),
    },
    "WEEKLY_RISK_SCORE_HEATMAP": {
        "name": "Weekly Risk Score Heatmap",
        "cadence": "WEEKLY",
        "report_types": [ReportType.RISK_ASSESSMENT],
        "description": "Per-constituency delay-rate and health-band grid, refreshed weekly.",
    },
    "MONTHLY_FEDERAL_COST_AUDIT": {
        "name": "Monthly Federal Cost Audit",
        "cadence": "MONTHLY",
        "report_types": [ReportType.FINANCIAL_SUMMARY, ReportType.AUDIT_COMPLIANCE],
        "description": "National financial summary combined with audit-compliance flags.",
    },
}


def _load_run_history() -> Dict[str, Any]:
    if not MODULE8_SUBSCRIPTION_RUNS_FILE.exists():
        return {}
    try:
        return json.loads(MODULE8_SUBSCRIPTION_RUNS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_run_history(history: Dict[str, Any]) -> None:
    MODULE8_SUBSCRIPTION_RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODULE8_SUBSCRIPTION_RUNS_FILE.write_text(json.dumps(history, indent=2))


def list_subscriptions() -> List[SubscriptionInfo]:
    history = _load_run_history()
    subscriptions = []
    for subscription_id, meta in SUBSCRIPTION_CATALOG.items():
        run_info = history.get(subscription_id, {})
        subscriptions.append(
            SubscriptionInfo(
                subscription_id=subscription_id,
                name=meta["name"],
                cadence=meta["cadence"],
                report_types=meta["report_types"],
                description=meta["description"],
                last_run_at=run_info.get("last_run_at"),
                last_run_status=run_info.get("last_run_status"),
            )
        )
    return subscriptions


def run_subscription(subscription_id: str) -> TriggerResponse:
    meta = SUBSCRIPTION_CATALOG.get(subscription_id)
    if meta is None:
        raise UnknownSubscriptionError(f"Unknown subscription_id: {subscription_id}")

    triggered_at = datetime.now(timezone.utc).isoformat()
    status = "SUCCESS"
    reports = []
    try:
        for report_type in meta["report_types"]:
            reports.append(generator.generate_report(report_type, entity_id=None, format="json"))
    except Exception:
        status = "FAILED"
        reports = []

    history = _load_run_history()
    history[subscription_id] = {"last_run_at": triggered_at, "last_run_status": status}
    _save_run_history(history)

    return TriggerResponse(subscription_id=subscription_id, triggered_at=triggered_at, status=status, reports=reports)
