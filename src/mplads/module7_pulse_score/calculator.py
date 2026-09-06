"""Pure rollup math for Module 7 (Development Pulse Score).

No I/O here -- every function takes plain numbers (already aggregated by
service.py) and returns a score or band. Kept side-effect free so each
dimension's math is independently testable.
"""
from __future__ import annotations

from typing import Optional

from mplads.config import (
    MODULE7_DIMENSION_WEIGHTS,
    MODULE7_HEALTH_BAND_EXCELLENT,
    MODULE7_HEALTH_BAND_GOOD,
    MODULE7_HEALTH_BAND_NEEDS_ATTENTION,
    MODULE7_VELOCITY_NEUTRAL_SCORE,
    MODULE7_VELOCITY_RATIO_CAP,
)
from mplads.module7_pulse_score.schemas import HealthBand


def completion_rate(total_works: int, completed_works: int) -> float:
    """completed_works / total_works, guarded against a zero-works entity
    (should not occur in practice -- an entity only exists in the index if
    it has at least one recommended work -- but guarded defensively)."""
    if total_works <= 0:
        return 0.0
    return completed_works / total_works


def fund_utilization_pct(total_sanctioned: float, total_disbursed: float) -> float:
    """Raw utilization percentage for display -- intentionally NOT capped,
    so a genuine over-100% aggregation anomaly stays visible rather than
    being silently hidden."""
    if total_sanctioned <= 0:
        return 0.0
    return (total_disbursed / total_sanctioned) * 100.0


def delay_rate(denominator: int, delayed_count: int) -> float:
    """delayed_count / denominator, guarded against a zero denominator
    (e.g. the fallback path when there are no completed works and no model
    is available to score in-progress ones)."""
    if denominator <= 0:
        return 0.0
    return delayed_count / denominator


def velocity_score(
    actual_avg_duration_days: Optional[float],
    benchmark_duration_days: Optional[float],
    ratio_cap: float = MODULE7_VELOCITY_RATIO_CAP,
    neutral_score: float = MODULE7_VELOCITY_NEUTRAL_SCORE,
) -> float:
    """0-100 score from benchmark/actual duration ratio (higher is better:
    finishing faster than the peer-group benchmark scores higher).

    Falls back to a neutral score when duration data is unavailable (e.g.
    zero completed works in the entity) rather than dividing by zero or
    inventing a duration.
    """
    if (
        actual_avg_duration_days is None
        or benchmark_duration_days is None
        or actual_avg_duration_days <= 0
        or benchmark_duration_days <= 0
    ):
        return neutral_score

    ratio = min(benchmark_duration_days / actual_avg_duration_days, ratio_cap)
    return (ratio / ratio_cap) * 100.0


def compute_pulse_score(
    completion_rate_value: float,
    fund_utilization_pct_value: float,
    delay_rate_value: float,
    velocity_score_value: float,
    weights: dict = MODULE7_DIMENSION_WEIGHTS,
) -> float:
    """Weighted blend of the 4 dimension sub-scores into a single 0-100 pulse
    score. Utilization is capped at 100 for scoring purposes only (the raw,
    uncapped value is still reported separately via fund_utilization_pct)."""
    completion_score = completion_rate_value * 100.0
    utilization_score = min(fund_utilization_pct_value, 100.0)
    delay_score = (1.0 - delay_rate_value) * 100.0

    pulse = (
        weights["completion"] * completion_score
        + weights["utilization"] * utilization_score
        + weights["delay"] * delay_score
        + weights["velocity"] * velocity_score_value
    )
    return round(pulse, 1)


def health_band(pulse_score: float) -> HealthBand:
    if pulse_score >= MODULE7_HEALTH_BAND_EXCELLENT:
        return "EXCELLENT"
    if pulse_score >= MODULE7_HEALTH_BAND_GOOD:
        return "GOOD"
    if pulse_score >= MODULE7_HEALTH_BAND_NEEDS_ATTENTION:
        return "NEEDS_ATTENTION"
    return "CRITICAL"
