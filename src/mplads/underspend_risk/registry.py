"""Locates and loads the persisted underspend_risk model artifact, if any.

Same no-fallback contract as Module 1 and Module 2's registries:
unavailable means unavailable, never an untrained/placeholder model. As of
this implementation, NO artifact has been trained and persisted yet (see
the accompanying implementation report) -- is_model_available() will
correctly return False and every prediction request will receive an
honest 503 until a legitimate training run produces one.
"""
from __future__ import annotations

from typing import Optional

from mplads.config import (
    UNDERSPEND_RISK_METADATA_FILE,
    UNDERSPEND_RISK_MODEL_FILE,
    UNDERSPEND_RISK_RATE_LOOKUP_FILE,
)
from mplads.underspend_risk.model import UnderspendRiskModel


def is_model_available() -> bool:
    return (
        UNDERSPEND_RISK_MODEL_FILE.exists()
        and UNDERSPEND_RISK_RATE_LOOKUP_FILE.exists()
        and UNDERSPEND_RISK_METADATA_FILE.exists()
    )


def load_active_model() -> Optional[UnderspendRiskModel]:
    if not is_model_available():
        return None
    return UnderspendRiskModel.load(UNDERSPEND_RISK_MODEL_FILE, UNDERSPEND_RISK_RATE_LOOKUP_FILE, UNDERSPEND_RISK_METADATA_FILE)
