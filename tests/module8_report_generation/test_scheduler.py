import pytest

from mplads.module8_report_generation import scheduler
from mplads.module8_report_generation.exceptions import UnknownSubscriptionError


@pytest.fixture(autouse=True)
def isolated_run_history(tmp_path, monkeypatch):
    monkeypatch.setattr(scheduler, "MODULE8_SUBSCRIPTION_RUNS_FILE", tmp_path / "subscription_runs.json")


def test_list_subscriptions_returns_all_catalog_entries_with_no_run_history():
    subscriptions = scheduler.list_subscriptions()
    ids = {s.subscription_id for s in subscriptions}
    assert ids == {"DAILY_ANOMALY_PULSE_LOG", "WEEKLY_RISK_SCORE_HEATMAP", "MONTHLY_FEDERAL_COST_AUDIT"}
    assert all(s.last_run_at is None for s in subscriptions)


def test_run_subscription_unknown_id_raises():
    with pytest.raises(UnknownSubscriptionError):
        scheduler.run_subscription("NOT_A_REAL_SUBSCRIPTION")


def test_run_subscription_succeeds_and_updates_history():
    response = scheduler.run_subscription("WEEKLY_RISK_SCORE_HEATMAP")
    assert response.status == "SUCCESS"
    assert len(response.reports) == 1
    assert response.reports[0].report_type.value == "RISK_ASSESSMENT"

    subscriptions = scheduler.list_subscriptions()
    weekly = next(s for s in subscriptions if s.subscription_id == "WEEKLY_RISK_SCORE_HEATMAP")
    assert weekly.last_run_status == "SUCCESS"
    assert weekly.last_run_at == response.triggered_at


def test_monthly_federal_cost_audit_runs_two_report_types():
    response = scheduler.run_subscription("MONTHLY_FEDERAL_COST_AUDIT")
    assert response.status == "SUCCESS"
    report_types = {r.report_type.value for r in response.reports}
    assert report_types == {"FINANCIAL_SUMMARY", "AUDIT_COMPLIANCE"}
