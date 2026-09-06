"""Locates and loads the persisted Module 2 model artifact, if any.

Same no-fallback contract as Module 1's registry: unavailable means
unavailable, never an untrained placeholder.
"""
from __future__ import annotations

from typing import Optional

from mplads.config import MODULE2_METADATA_FILE, MODULE2_MODEL_FILE
from mplads.module2_delay_prediction.model import DelayRiskModel


def is_model_available() -> bool:
    return MODULE2_MODEL_FILE.exists() and MODULE2_METADATA_FILE.exists()


def load_active_model() -> Optional[DelayRiskModel]:
    if not is_model_available():
        return None
    return DelayRiskModel.load(MODULE2_MODEL_FILE, MODULE2_METADATA_FILE)
