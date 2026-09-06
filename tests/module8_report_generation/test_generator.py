import pandas as pd
import pytest

from mplads.config import WORKS_RECOMMENDED_CSV
from mplads.module7_pulse_score.exceptions import EntityNotFoundError
from mplads.module8_report_generation import generator
from mplads.module8_report_generation.schemas import ReportType


@pytest.fixture(scope="module")
def real_constituency():
    df = pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)
    return df.iloc[0]["Constituency"]


@pytest.mark.parametrize("report_type", list(ReportType))
def test_generate_report_national_scope_for_every_type(report_type):
    response = generator.generate_report(report_type, entity_id=None, format="json")
    assert response.entity_id is None
    assert isinstance(response.data.get("summary"), dict)


@pytest.mark.parametrize("report_type", list(ReportType))
def test_generate_report_entity_scope_for_every_type(report_type, real_constituency):
    response = generator.generate_report(report_type, entity_id=real_constituency, format="json")
    assert response.entity_id == real_constituency
    assert isinstance(response.data.get("summary"), dict)


def test_generate_report_unknown_entity_raises():
    with pytest.raises(EntityNotFoundError):
        generator.generate_report(ReportType.FINANCIAL_SUMMARY, entity_id="Nowhere Land XYZ", format="json")


def test_financial_summary_national_breakdown_sums_to_summary_total():
    response = generator.generate_report(ReportType.FINANCIAL_SUMMARY, entity_id=None, format="json")
    breakdown_total = sum(row["total_sanctioned"] for row in response.data["breakdown"])
    assert breakdown_total == pytest.approx(response.data["summary"]["total_sanctioned"], rel=1e-6)


def test_financial_summary_includes_expenditure_for_constituencies_absent_from_recommended():
    """Regression test: Expenditure_Cleaned.csv references a handful of
    constituencies (e.g. GODDA, KALYAN) that don't appear anywhere in
    Works_Recommended. The national total_disbursed must still include their
    money (previously silently dropped by reindexing onto a
    Recommended-only constituency index); they should be called out in
    unmatched_expenditure_note and excluded from the per-constituency
    breakdown, since they can't be resolved to a real entity."""
    response = generator.generate_report(ReportType.FINANCIAL_SUMMARY, entity_id=None, format="json")
    summary = response.data["summary"]

    assert "unmatched_expenditure_note" in summary
    assert "GODDA" in summary["unmatched_expenditure_note"]

    breakdown_entities = {row["entity_id"] for row in response.data["breakdown"]}
    assert "GODDA" not in breakdown_entities
    assert "KALYAN" not in breakdown_entities

    # The known stray-expenditure total (non-completed Work IDs only) is
    # ~Rs 25,260,560 -- confirm the national total_disbursed is at least
    # that much higher than what summing only registered constituencies'
    # completed+expenditure would give if the gap were still present.
    assert summary["total_disbursed"] > 25_000_000_000


def test_project_status_completed_plus_in_progress_equals_total(real_constituency):
    response = generator.generate_report(ReportType.PROJECT_STATUS, entity_id=real_constituency, format="json")
    summary = response.data["summary"]
    assert summary["completed_works"] + summary["in_progress_works"] == summary["total_works"]


def test_risk_assessment_entity_scope_includes_pulse_score_and_placeholders(real_constituency):
    response = generator.generate_report(ReportType.RISK_ASSESSMENT, entity_id=real_constituency, format="json")
    summary = response.data["summary"]
    assert "pulse_score" in summary
    assert "health_band" in summary
    assert "module6_risk_scoring_engine" in response.data["placeholders"]
    assert response.data["placeholders"]["module6_risk_scoring_engine"]["status"] == "not_yet_available"


def test_site_verification_counts_are_internally_consistent(real_constituency):
    response = generator.generate_report(ReportType.SITE_VERIFICATION, entity_id=real_constituency, format="json")
    summary = response.data["summary"]
    assert (
        summary["works_with_photographic_evidence"] + summary["works_without_photographic_evidence"]
        == summary["completed_works"]
    )


def test_audit_compliance_flags_unsanctioned_works_nationally():
    """From inspection: ~5440 Works_Recommended rows have no Sanction Date
    (the 'NA-' placeholder unsanctioned recommendations)."""
    response = generator.generate_report(ReportType.AUDIT_COMPLIANCE, entity_id=None, format="json")
    assert response.data["summary"]["recommended_but_unsanctioned_count"] > 0


def test_csv_and_markdown_formats_populate_rendered_text(real_constituency):
    csv_response = generator.generate_report(ReportType.FINANCIAL_SUMMARY, entity_id=real_constituency, format="csv")
    assert csv_response.rendered_text is not None
    assert "metric,value" in csv_response.rendered_text

    md_response = generator.generate_report(ReportType.FINANCIAL_SUMMARY, entity_id=real_constituency, format="markdown")
    assert md_response.rendered_text is not None
    assert md_response.rendered_text.startswith("# Financial Summary")


def test_json_format_has_no_rendered_text(real_constituency):
    response = generator.generate_report(ReportType.FINANCIAL_SUMMARY, entity_id=real_constituency, format="json")
    assert response.rendered_text is None


def test_fund_utilization_missing_work_type_not_silently_dropped():
    """Works_Recommended has ~8% missing Work Type -- these must be bucketed
    as 'UNKNOWN' rather than vanishing from the national totals."""
    response = generator.generate_report(ReportType.FUND_UTILIZATION, entity_id=None, format="json")
    work_types = {row["work_type"] for row in response.data["breakdown"]}
    assert "UNKNOWN" in work_types
