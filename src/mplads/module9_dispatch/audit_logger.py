"""Append-only audit trail for Module 9.

Deliberately exposes no update/delete function -- log entries, once written,
are never modified. Designed as a generic utility (not hardcoded to Module
9's own ALERT_TRIGGERED/INSPECTOR_DISPATCHED event types) so other modules
(e.g. a future Module 3/4 anomaly-verification flow) can append to the same
audit_trail.json schema later without any change here.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mplads.config import MODULE9_AUDIT_TRAIL_FILE
from mplads.module9_dispatch.schemas import Actor, AuditLogEntry, EventType


def _load_entries() -> List[dict]:
    if not MODULE9_AUDIT_TRAIL_FILE.exists():
        return []
    try:
        return json.loads(MODULE9_AUDIT_TRAIL_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _save_entries(entries: List[dict]) -> None:
    MODULE9_AUDIT_TRAIL_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODULE9_AUDIT_TRAIL_FILE.write_text(json.dumps(entries, indent=2))


def append_log(project_id: str, event_type: EventType, actor: Actor, details: Dict[str, Any]) -> AuditLogEntry:
    entry = AuditLogEntry(
        log_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc).isoformat(),
        project_id=project_id,
        event_type=event_type,
        actor=actor,
        details=details,
    )
    entries = _load_entries()
    entries.append(entry.model_dump(mode="json"))
    _save_entries(entries)
    return entry


def read_audit_trail(project_id: Optional[str] = None, actor: Optional[str] = None) -> List[AuditLogEntry]:
    entries = [AuditLogEntry.model_validate(e) for e in _load_entries()]
    if project_id is not None:
        entries = [e for e in entries if e.project_id == project_id]
    if actor is not None:
        entries = [e for e in entries if e.actor == actor]
    return entries
