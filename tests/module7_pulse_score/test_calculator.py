import pytest

from mplads.module7_pulse_score import calculator


def test_completion_rate_normal():
    assert calculator.completion_rate(100, 85) == pytest.approx(0.85)


def test_completion_rate_zero_total_guarded():
    assert calculator.completion_rate(0, 0) == 0.0


def test_fund_utilization_pct_normal():
    assert calculator.fund_utilization_pct(1_000_000, 780_000) == pytest.approx(78.0)


def test_fund_utilization_pct_zero_sanctioned_guarded():
    assert calculator.fund_utilization_pct(0, 500) == 0.0


def test_fund_utilization_pct_not_capped_for_display():
    """Raw utilization is reported uncapped so an aggregation anomaly stays
    visible; only the pulse-score blend caps it."""
    assert calculator.fund_utilization_pct(100, 150) == pytest.approx(150.0)


def test_delay_rate_normal():
    assert calculator.delay_rate(120, 12) == pytest.approx(0.1)


def test_delay_rate_zero_denominator_guarded():
    assert calculator.delay_rate(0, 0) == 0.0


def test_velocity_score_faster_than_benchmark_scores_high():
    # actual duration half the benchmark -> ratio capped at 1.5 -> max score
    score = calculator.velocity_score(actual_avg_duration_days=100, benchmark_duration_days=200)
    assert score == 100.0


def test_velocity_score_slower_than_benchmark_scores_low():
    score = calculator.velocity_score(actual_avg_duration_days=200, benchmark_duration_days=100)
    assert 0 < score < 50


def test_velocity_score_equal_to_benchmark_is_midband():
    score = calculator.velocity_score(actual_avg_duration_days=150, benchmark_duration_days=150)
    assert score == pytest.approx(100 / 1.5)


def test_velocity_score_none_inputs_return_neutral_fallback():
    assert calculator.velocity_score(None, None) == 50.0
    assert calculator.velocity_score(None, 150) == 50.0
    assert calculator.velocity_score(150, None) == 50.0


def test_velocity_score_zero_duration_guarded():
    assert calculator.velocity_score(0, 150) == 50.0
    assert calculator.velocity_score(150, 0) == 50.0


def test_compute_pulse_score_perfect_project_scores_100():
    score = calculator.compute_pulse_score(
        completion_rate_value=1.0,
        fund_utilization_pct_value=100.0,
        delay_rate_value=0.0,
        velocity_score_value=100.0,
    )
    assert score == 100.0


def test_compute_pulse_score_worst_project_scores_0():
    score = calculator.compute_pulse_score(
        completion_rate_value=0.0,
        fund_utilization_pct_value=0.0,
        delay_rate_value=1.0,
        velocity_score_value=0.0,
    )
    assert score == 0.0


def test_compute_pulse_score_utilization_capped_in_blend():
    """A >100% raw utilization must not push the blended score above what a
    perfect 100% utilization would give."""
    score_over = calculator.compute_pulse_score(1.0, 150.0, 0.0, 100.0)
    score_exact = calculator.compute_pulse_score(1.0, 100.0, 0.0, 100.0)
    assert score_over == score_exact == 100.0


@pytest.mark.parametrize(
    "score,expected_band",
    [
        (95.0, "EXCELLENT"),
        (85.0, "EXCELLENT"),
        (84.9, "GOOD"),
        (70.0, "GOOD"),
        (69.9, "NEEDS_ATTENTION"),
        (50.0, "NEEDS_ATTENTION"),
        (49.9, "CRITICAL"),
        (0.0, "CRITICAL"),
    ],
)
def test_health_band_thresholds(score, expected_band):
    assert calculator.health_band(score) == expected_band
