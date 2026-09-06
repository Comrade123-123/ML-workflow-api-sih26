"""Locates and loads the persisted Module 1 model artifact, if any.

Deliberately has no fallback: if the artifact files aren't on disk, it
reports unavailability rather than constructing an untrained/placeholder
model. This is what lets the API return an honest 503 instead of a
fabricated prediction.
"""
from __future__ import annotations

from typing import Optional

from mplads.config import MODULE1_METADATA_FILE, MODULE1_MODEL_FILE
from mplads.module1_cost_overrun.model import CostOverrunModel


def is_model_available() -> bool:
    return MODULE1_MODEL_FILE.exists() and MODULE1_METADATA_FILE.exists()


def load_active_model() -> Optional[CostOverrunModel]:
    """Returns the currently persisted model, or None if none has been
    legitimately trained and saved yet."""
    if not is_model_available():
        return None
    return CostOverrunModel.load(MODULE1_MODEL_FILE, MODULE1_METADATA_FILE)
