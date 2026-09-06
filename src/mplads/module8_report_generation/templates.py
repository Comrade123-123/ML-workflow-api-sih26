"""Report metadata and CSV/Markdown rendering for Module 8.

Every generator in generator.py produces its payload in one consistent
shape: {"summary": {...flat key/value...}, "breakdown": [...flat dicts...]?,
"placeholders": {...teammate-module hooks...}?}. That convention lets a
single generic renderer serve all 6 templates instead of six bespoke ones.
"""
from __future__ import annotations

import csv
import io
from typing import Any, Dict, Optional

from mplads.module8_report_generation.schemas import ReportType

TEMPLATE_METADATA: Dict[ReportType, Dict[str, str]] = {
    ReportType.FINANCIAL_SUMMARY: {
        "name": "Financial Summary",
        "description": "Sanctioned vs. disbursed totals, fund utilization ratios, and a per-constituency breakdown.",
    },
    ReportType.PROJECT_STATUS: {
        "name": "Project Status",
        "description": "Completed vs. in-progress vs. delayed works breakdown, powered by the Module 2 delay-risk model.",
    },
    ReportType.FUND_UTILIZATION: {
        "name": "Fund Utilization",
        "description": "Expenditure pacing and tranche utilization broken down by Work Type.",
    },
    ReportType.RISK_ASSESSMENT: {
        "name": "Risk Assessment",
        "description": "Aggregate view of at-risk/delayed projects, with placeholder hooks for Modules 3-6.",
    },
    ReportType.SITE_VERIFICATION: {
        "name": "Site Verification",
        "description": "Aggregation of completed works' recorded photographic evidence (Has Image).",
    },
    ReportType.AUDIT_COMPLIANCE: {
        "name": "Audit & Compliance",
        "description": "Recommended-but-unsanctioned works, missing-description flags, and Module 2 delay violations.",
    },
}


def render_csv(data: Dict[str, Any]) -> str:
    """Summary as a 2-column metric/value table, optionally followed by a
    blank line and the breakdown table (if present)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    summary = data.get("summary") or {}
    if summary:
        writer.writerow(["metric", "value"])
        for key, value in summary.items():
            writer.writerow([key, value])

    breakdown = data.get("breakdown")
    if breakdown:
        if summary:
            writer.writerow([])
        headers = list(breakdown[0].keys())
        dict_writer = csv.DictWriter(buffer, fieldnames=headers)
        dict_writer.writeheader()
        for row in breakdown:
            dict_writer.writerow(row)

    return buffer.getvalue()


def render_markdown(report_type: ReportType, entity_id: Optional[str], data: Dict[str, Any]) -> str:
    meta = TEMPLATE_METADATA[report_type]
    lines = [f"# {meta['name']}", "", meta["description"], "", f"**Scope:** {entity_id or 'ALL (national)'}", ""]

    summary = data.get("summary") or {}
    if summary:
        lines.append("## Summary")
        for key, value in summary.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")

    breakdown = data.get("breakdown")
    if breakdown:
        lines.append("## Breakdown")
        headers = list(breakdown[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in breakdown:
            lines.append("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |")
        lines.append("")

    placeholders = data.get("placeholders")
    if placeholders:
        lines.append("## Not Yet Available (Teammate Modules)")
        for key, hook in placeholders.items():
            lines.append(f"- **{key}** ({hook.get('module')}): {hook.get('description')}")
        lines.append("")

    return "\n".join(lines)
