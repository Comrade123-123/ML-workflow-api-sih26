"""Pydantic response contract for Module 7 (Development Pulse Score)."""
from typing import Literal

from pydantic import BaseModel

HealthBand = Literal["EXCELLENT", "GOOD", "NEEDS_ATTENTION", "CRITICAL"]


class PulseScoreResponse(BaseModel):
    """Matches the MP Profile UI card exactly."""

    entity_id: str
    pulse_score: float
    delta_vs_last_period: float
    fund_utilization_rate_pct: float
    total_works: int
    completed_works: int
    delayed_works_count: int
    health_band: HealthBand
