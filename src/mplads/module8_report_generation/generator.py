"""Report computation for Module 8's 6 templates.

Reuses Module 7's public resolve_entity()/get_pulse_score() and Module 2's
public model registry rather than duplicating their internals. Where no
public function exposes the exact figures needed (e.g. raw currency totals
per constituency), a small self-contained rollup is computed here directly
from the source CSVs -- Modules 1, 2 and 7 are not modified.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd

from mplads.config import (
    COL_AMOUNT_DISBURSED,
    COL_FUND_DISBURSED,
    COL_RECOMMENDED_AMOUNT,
    COL_WORK_CATEGORY,
    COL_WORK_ID,
    COL_WORK_TYPE,
    EXPENDITURE_CSV,
    WORKS_COMPLETED_CSV,
    WORKS_RECOMMENDED_CSV,
)
from mplads.module2_delay_prediction.registry import load_active_model as load_module2_model
from mplads.module7_pulse_score.service import get_pulse_score, resolve_entity
from mplads.module8_report_generation import templates
from mplads.module8_report_generation.exceptions import UnknownReportTypeError
from mplads.module8_report_generation.schemas import ReportFormat, ReportResponse, ReportType

COL_CONSTITUENCY = "Constituency"
COL_SANCTION_DATE = "Sanction Date"
COL_WORK_DESCRIPTION = "Work Description"
COL_HAS_IMAGE = "Has Image"

DELAY_MODEL_FEATURE_COLUMNS = [
    COL_WORK_CATEGORY,
    COL_WORK_TYPE,
    "State",
    "IDA District",
    COL_RECOMMENDED_AMOUNT,
    "Recommended Date",
    COL_SANCTION_DATE,
]

GeneratorResult = Tuple[Dict[str, Any], Optional[str]]


def _placeholder(module: str, description: str) -> Dict[str, str]:
    return {"status": "not_yet_available", "module": module, "description": description}


def _score_delay_status(recommended_df: pd.DataFrame) -> Optional[pd.Series]:
    """Returns Module 2's predicted status per row, or None if no trained
    model is currently available (never fabricates a prediction)."""
    model = load_module2_model()
    if model is None:
        return None
    return model.predict(recommended_df[DELAY_MODEL_FEATURE_COLUMNS])


# --- 1. FINANCIAL_SUMMARY ---------------------------------------------------


def _financial_rollup_by_constituency(
    recommended: pd.DataFrame, completed: pd.DataFrame, expenditure: pd.DataFrame
) -> pd.DataFrame:
    """One row per constituency: total_sanctioned, total_disbursed,
    total_works, fund_utilization_pct. Uses groupby aggregates rather than
    re-filtering the full frames per constituency, so it stays fast at the
    national scale (~529 constituencies).

    The row index is the UNION of constituency names seen across all three
    source files, not just Works_Recommended. A handful of Expenditure rows
    (source-data gap, confirmed not a typo/near-duplicate of any real
    constituency) reference constituency names absent from Works_Recommended
    entirely; indexing on Recommended alone would silently drop that money
    from every national total via reindex(). Those extra rows carry
    total_works=0 and are excluded from the per-constituency breakdown (see
    generate_financial_summary) but still count towards the summary totals.
    """
    all_constituencies = sorted(
        set(recommended[COL_CONSTITUENCY]) | set(completed[COL_CONSTITUENCY]) | set(expenditure[COL_CONSTITUENCY])
    )
    sanctioned = recommended.groupby(COL_CONSTITUENCY)[COL_RECOMMENDED_AMOUNT].sum()
    total_works = recommended.groupby(COL_CONSTITUENCY).size()

    completed_ids = set(completed[COL_WORK_ID])
    non_completed_expenditure = expenditure[~expenditure[COL_WORK_ID].isin(completed_ids)]
    disbursed_completed = completed.groupby(COL_CONSTITUENCY)[COL_AMOUNT_DISBURSED].sum()
    disbursed_expenditure = non_completed_expenditure.groupby(COL_CONSTITUENCY)[COL_FUND_DISBURSED].sum()

    rollup = pd.DataFrame(index=all_constituencies)
    rollup["total_sanctioned"] = sanctioned.reindex(rollup.index).fillna(0.0)
    rollup["total_works"] = total_works.reindex(rollup.index).fillna(0).astype(int)
    rollup["total_disbursed"] = disbursed_completed.reindex(rollup.index).fillna(0.0) + disbursed_expenditure.reindex(
        rollup.index
    ).fillna(0.0)
    rollup["fund_utilization_pct"] = (rollup["total_disbursed"] / rollup["total_sanctioned"] * 100.0).where(
        rollup["total_sanctioned"] > 0, 0.0
    )
    return rollup


def generate_financial_summary(entity_id: Optional[str]) -> GeneratorResult:
    recommended = pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)
    completed = pd.read_csv(WORKS_COMPLETED_CSV, low_memory=False)
    expenditure = pd.read_csv(EXPENDITURE_CSV, low_memory=False)

    canonical_id = resolve_entity(entity_id, recommended) if entity_id else None
    rollup = _financial_rollup_by_constituency(recommended, completed, expenditure)

    if canonical_id:
        row = rollup.loc[canonical_id]
        summary = {
            "total_sanctioned": round(float(row["total_sanctioned"]), 2),
            "total_disbursed": round(float(row["total_disbursed"]), 2),
            "fund_utilization_pct": round(float(row["fund_utilization_pct"]), 2),
            "total_works": int(row["total_works"]),
        }
        return {"summary": summary}, canonical_id

    total_sanctioned = float(rollup["total_sanctioned"].sum())
    total_disbursed = float(rollup["total_disbursed"].sum())
    registered = rollup[rollup["total_works"] > 0]
    unmatched = rollup[rollup["total_works"] == 0]
    summary = {
        "total_sanctioned": round(total_sanctioned, 2),
        "total_disbursed": round(total_disbursed, 2),
        "fund_utilization_pct": round(total_disbursed / total_sanctioned * 100.0, 2) if total_sanctioned > 0 else 0.0,
        "total_works": int(rollup["total_works"].sum()),
        "constituency_count": len(registered),
    }
    if not unmatched.empty:
        summary["unmatched_expenditure_note"] = (
            f"{len(unmatched)} constituency name(s) appear in Expenditure/Completed records but have no "
            f"matching entry in Works_Recommended (source-data gap: {sorted(unmatched.index.tolist())}). "
            f"Their disbursed amount (Rs {unmatched['total_disbursed'].sum():,.2f}) is included in "
            "total_disbursed above but excluded from the per-constituency breakdown below."
        )
    breakdown = [
        {
            "entity_id": constituency,
            "total_sanctioned": round(float(r["total_sanctioned"]), 2),
            "total_disbursed": round(float(r["total_disbursed"]), 2),
            "fund_utilization_pct": round(float(r["fund_utilization_pct"]), 2),
            "total_works": int(r["total_works"]),
        }
        for constituency, r in registered.sort_values("fund_utilization_pct").iterrows()
    ]
    return {"summary": summary, "breakdown": breakdown}, None


# --- 2. PROJECT_STATUS ------------------------------------------------------


def generate_project_status(entity_id: Optional[str]) -> GeneratorResult:
    recommended = pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)
    completed = pd.read_csv(WORKS_COMPLETED_CSV, low_memory=False)

    canonical_id = resolve_entity(entity_id, recommended) if entity_id else None
    if canonical_id:
        recommended = recommended[recommended[COL_CONSTITUENCY] == canonical_id]
        completed = completed[completed[COL_CONSTITUENCY] == canonical_id]

    completed_ids = set(completed[COL_WORK_ID])
    is_completed = recommended[COL_WORK_ID].isin(completed_ids)
    predictions = _score_delay_status(recommended)

    total_works = len(recommended)
    completed_works = int(is_completed.sum())
    summary = {
        "total_works": total_works,
        "completed_works": completed_works,
        "in_progress_works": total_works - completed_works,
        "delay_classification_source": "model" if predictions is not None else "unavailable",
        "on_track_count": int((predictions == "ON_TRACK").sum()) if predictions is not None else None,
        "at_risk_count": int((predictions == "AT_RISK").sum()) if predictions is not None else None,
        "likely_delayed_count": int((predictions == "LIKELY_DELAYED").sum()) if predictions is not None else None,
    }

    if canonical_id:
        return {"summary": summary}, canonical_id

    working = recommended.assign(_is_completed=is_completed)
    if predictions is not None:
        working = working.assign(_status=predictions)
    breakdown = []
    for constituency, group in working.groupby(COL_CONSTITUENCY):
        total = len(group)
        completed_count = int(group["_is_completed"].sum())
        row = {
            "entity_id": constituency,
            "total_works": total,
            "completed_works": completed_count,
            "in_progress_works": total - completed_count,
        }
        if predictions is not None:
            status_counts = group["_status"].value_counts()
            row["at_risk_count"] = int(status_counts.get("AT_RISK", 0))
            row["likely_delayed_count"] = int(status_counts.get("LIKELY_DELAYED", 0))
        breakdown.append(row)
    breakdown.sort(key=lambda r: -r["total_works"])
    return {"summary": summary, "breakdown": breakdown}, None


# --- 3. FUND_UTILIZATION ----------------------------------------------------


def generate_fund_utilization(entity_id: Optional[str]) -> GeneratorResult:
    recommended = pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)
    completed = pd.read_csv(WORKS_COMPLETED_CSV, low_memory=False)
    expenditure = pd.read_csv(EXPENDITURE_CSV, low_memory=False)

    canonical_id = resolve_entity(entity_id, recommended) if entity_id else None
    if canonical_id:
        recommended = recommended[recommended[COL_CONSTITUENCY] == canonical_id]
        completed = completed[completed[COL_CONSTITUENCY] == canonical_id]
        expenditure = expenditure[expenditure[COL_CONSTITUENCY] == canonical_id]

    # groupby drops NaN keys by default -- Works_Recommended has ~8% missing
    # Work Type (the unsanctioned "NA-" placeholder rows). Label them
    # explicitly so their sanctioned amounts aren't silently dropped from
    # the totals below.
    recommended = recommended.assign(**{COL_WORK_TYPE: recommended[COL_WORK_TYPE].fillna("UNKNOWN")})

    sanctioned_by_type = recommended.groupby(COL_WORK_TYPE)[COL_RECOMMENDED_AMOUNT].sum()
    completed_ids = set(completed[COL_WORK_ID])
    non_completed_expenditure = expenditure[~expenditure[COL_WORK_ID].isin(completed_ids)]
    disbursed_completed_by_type = completed.groupby(COL_WORK_TYPE)[COL_AMOUNT_DISBURSED].sum()
    disbursed_expenditure_by_type = non_completed_expenditure.groupby(COL_WORK_TYPE)[COL_FUND_DISBURSED].sum()
    txn_count_by_type = expenditure.groupby(COL_WORK_TYPE).size()
    avg_txn_amount_by_type = expenditure.groupby(COL_WORK_TYPE)[COL_FUND_DISBURSED].mean()

    work_types = sorted(set(sanctioned_by_type.index) | set(txn_count_by_type.index))
    breakdown = []
    for work_type in work_types:
        sanctioned = float(sanctioned_by_type.get(work_type, 0.0))
        disbursed = float(disbursed_completed_by_type.get(work_type, 0.0)) + float(
            disbursed_expenditure_by_type.get(work_type, 0.0)
        )
        breakdown.append(
            {
                "work_type": work_type,
                "total_sanctioned": round(sanctioned, 2),
                "total_disbursed": round(disbursed, 2),
                "fund_utilization_pct": round(disbursed / sanctioned * 100.0, 2) if sanctioned > 0 else 0.0,
                "transaction_count": int(txn_count_by_type.get(work_type, 0)),
                "avg_transaction_amount": round(float(avg_txn_amount_by_type.get(work_type, 0.0)), 2),
            }
        )
    breakdown.sort(key=lambda r: -r["total_sanctioned"])

    total_sanctioned = float(sanctioned_by_type.sum())
    total_disbursed = sum(r["total_disbursed"] for r in breakdown)
    summary = {
        "total_sanctioned": round(total_sanctioned, 2),
        "total_disbursed": round(total_disbursed, 2),
        "fund_utilization_pct": round(total_disbursed / total_sanctioned * 100.0, 2) if total_sanctioned > 0 else 0.0,
        "work_type_count": len(breakdown),
    }
    return {"summary": summary, "breakdown": breakdown}, canonical_id


# --- 4. RISK_ASSESSMENT -----------------------------------------------------

_RISK_PLACEHOLDERS = {
    "module3_unusual_expenditure": _placeholder(
        "module3_unusual_expenditure", "Isolation-Forest-based anomaly scores are not yet integrated."
    ),
    "module4_duplicate_work_detection": _placeholder(
        "module4_duplicate_work_detection", "Text/geospatial/metadata duplicate-work flags are not yet integrated."
    ),
    "module5_document_site_verification": _placeholder(
        "module5_document_site_verification", "OCR/geotag consistency verification is not yet integrated."
    ),
    "module6_risk_scoring_engine": _placeholder(
        "module6_risk_scoring_engine", "The composite risk-scoring engine is not yet integrated."
    ),
}


def generate_risk_assessment(entity_id: Optional[str]) -> GeneratorResult:
    recommended = pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)
    canonical_id = resolve_entity(entity_id, recommended) if entity_id else None
    scope = recommended[recommended[COL_CONSTITUENCY] == canonical_id] if canonical_id else recommended

    predictions = _score_delay_status(scope)
    at_risk = int((predictions == "AT_RISK").sum()) if predictions is not None else None
    likely_delayed = int((predictions == "LIKELY_DELAYED").sum()) if predictions is not None else None

    summary: Dict[str, Any] = {
        "total_works_assessed": len(scope),
        "at_risk_count": at_risk,
        "likely_delayed_count": likely_delayed,
        "delay_classification_source": "model" if predictions is not None else "unavailable",
    }

    if canonical_id:
        pulse = get_pulse_score(canonical_id)
        summary["pulse_score"] = pulse.pulse_score
        summary["health_band"] = pulse.health_band
        return {"summary": summary, "placeholders": _RISK_PLACEHOLDERS}, canonical_id

    working = scope.assign(_status=predictions) if predictions is not None else scope
    breakdown = []
    for constituency, group in working.groupby(COL_CONSTITUENCY):
        row = {"entity_id": constituency, "total_works": len(group)}
        if predictions is not None:
            status_counts = group["_status"].value_counts()
            row["at_risk_count"] = int(status_counts.get("AT_RISK", 0))
            row["likely_delayed_count"] = int(status_counts.get("LIKELY_DELAYED", 0))
        else:
            row["at_risk_count"] = None
            row["likely_delayed_count"] = None
        breakdown.append(row)
    breakdown.sort(key=lambda r: -((r["at_risk_count"] or 0) + (r["likely_delayed_count"] or 0)))
    return {"summary": summary, "breakdown": breakdown, "placeholders": _RISK_PLACEHOLDERS}, None


# --- 5. SITE_VERIFICATION ---------------------------------------------------

_SITE_VERIFICATION_PLACEHOLDERS = {
    "module5_document_site_verification": _placeholder(
        "module5_document_site_verification",
        "OCR/geotag consistency checks are not yet integrated; this report currently reflects only whether a "
        "completion photo was recorded (Has Image).",
    ),
}


def generate_site_verification(entity_id: Optional[str]) -> GeneratorResult:
    recommended = pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)
    completed = pd.read_csv(WORKS_COMPLETED_CSV, low_memory=False)

    canonical_id = resolve_entity(entity_id, recommended) if entity_id else None
    if canonical_id:
        completed = completed[completed[COL_CONSTITUENCY] == canonical_id]

    total_completed = len(completed)
    with_image = int(completed[COL_HAS_IMAGE].sum()) if total_completed else 0
    summary = {
        "completed_works": total_completed,
        "works_with_photographic_evidence": with_image,
        "works_without_photographic_evidence": total_completed - with_image,
        "photographic_evidence_pct": round(with_image / total_completed * 100.0, 2) if total_completed else 0.0,
    }

    if canonical_id:
        return {"summary": summary, "placeholders": _SITE_VERIFICATION_PLACEHOLDERS}, canonical_id

    breakdown = []
    for constituency, group in completed.groupby(COL_CONSTITUENCY):
        total = len(group)
        with_image_count = int(group[COL_HAS_IMAGE].sum())
        breakdown.append(
            {
                "entity_id": constituency,
                "completed_works": total,
                "works_with_photographic_evidence": with_image_count,
                "photographic_evidence_pct": round(with_image_count / total * 100.0, 2) if total else 0.0,
            }
        )
    breakdown.sort(key=lambda r: r["photographic_evidence_pct"])
    return {"summary": summary, "breakdown": breakdown, "placeholders": _SITE_VERIFICATION_PLACEHOLDERS}, None


# --- 6. AUDIT_COMPLIANCE ----------------------------------------------------

_AUDIT_PLACEHOLDERS = {
    "module3_unusual_expenditure_anomalies": _placeholder(
        "module3_unusual_expenditure", "Isolation-Forest-based anomaly flags are not yet integrated."
    ),
}


def generate_audit_compliance(entity_id: Optional[str]) -> GeneratorResult:
    recommended = pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)
    canonical_id = resolve_entity(entity_id, recommended) if entity_id else None
    scope = recommended[recommended[COL_CONSTITUENCY] == canonical_id] if canonical_id else recommended

    unsanctioned = scope[COL_SANCTION_DATE].isna()
    missing_description = scope[COL_WORK_DESCRIPTION].isna()
    predictions = _score_delay_status(scope)
    delay_violations = int((predictions == "LIKELY_DELAYED").sum()) if predictions is not None else None

    summary = {
        "total_works_reviewed": len(scope),
        "recommended_but_unsanctioned_count": int(unsanctioned.sum()),
        "missing_work_description_count": int(missing_description.sum()),
        "delay_violation_count_likely_delayed": delay_violations,
        "delay_classification_source": "model" if predictions is not None else "unavailable",
    }
    return {"summary": summary, "placeholders": _AUDIT_PLACEHOLDERS}, canonical_id


# --- dispatch ----------------------------------------------------------------

_GENERATORS: Dict[ReportType, Callable[[Optional[str]], GeneratorResult]] = {
    ReportType.FINANCIAL_SUMMARY: generate_financial_summary,
    ReportType.PROJECT_STATUS: generate_project_status,
    ReportType.FUND_UTILIZATION: generate_fund_utilization,
    ReportType.RISK_ASSESSMENT: generate_risk_assessment,
    ReportType.SITE_VERIFICATION: generate_site_verification,
    ReportType.AUDIT_COMPLIANCE: generate_audit_compliance,
}


def generate_report(report_type: ReportType, entity_id: Optional[str] = None, format: ReportFormat = "json") -> ReportResponse:
    """Raises EntityNotFoundError (Module 7's) if entity_id doesn't resolve,
    or UnknownReportTypeError defensively (should be unreachable given
    ReportType is an exhaustively-mapped enum)."""
    generator_fn = _GENERATORS.get(report_type)
    if generator_fn is None:
        raise UnknownReportTypeError(f"Unknown report_type: {report_type}")

    data, canonical_entity_id = generator_fn(entity_id)

    rendered_text = None
    if format == "csv":
        rendered_text = templates.render_csv(data)
    elif format == "markdown":
        rendered_text = templates.render_markdown(report_type, canonical_entity_id, data)

    return ReportResponse(
        report_type=report_type,
        entity_id=canonical_entity_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        format=format,
        data=data,
        rendered_text=rendered_text,
    )
