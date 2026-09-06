from mplads.module8_report_generation import templates
from mplads.module8_report_generation.schemas import ReportType


def test_render_csv_summary_only():
    csv_text = templates.render_csv({"summary": {"a": 1, "b": 2}})
    lines = csv_text.splitlines()
    assert lines[0] == "metric,value"
    assert "a,1" in lines
    assert "b,2" in lines


def test_render_csv_with_breakdown():
    data = {"summary": {"a": 1}, "breakdown": [{"x": 1, "y": 2}, {"x": 3, "y": 4}]}
    lines = [l for l in templates.render_csv(data).splitlines() if l]
    assert "x,y" in lines
    assert "1,2" in lines
    assert "3,4" in lines


def test_render_csv_breakdown_only_no_summary():
    csv_text = templates.render_csv({"breakdown": [{"x": 1}]})
    assert "metric,value" not in csv_text
    assert "x" in csv_text.splitlines()[0]


def test_render_markdown_includes_all_sections():
    data = {
        "summary": {"metric_a": 10},
        "breakdown": [{"col1": "v1", "col2": "v2"}],
        "placeholders": {
            "module6_risk_scoring_engine": {
                "module": "module6_risk_scoring_engine",
                "description": "not ready",
            }
        },
    }
    md = templates.render_markdown(ReportType.RISK_ASSESSMENT, "DHARWAD", data)
    assert "# Risk Assessment" in md
    assert "**Scope:** DHARWAD" in md
    assert "metric_a" in md
    assert "| col1 | col2 |" in md
    assert "Not Yet Available" in md
    assert "module6_risk_scoring_engine" in md


def test_render_markdown_national_scope_label():
    md = templates.render_markdown(ReportType.FINANCIAL_SUMMARY, None, {"summary": {}})
    assert "ALL (national)" in md
