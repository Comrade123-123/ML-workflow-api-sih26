"""The Module 1 model interface: a thin, swappable wrapper around a
scikit-learn-compatible regressor plus the preprocessing pipeline.

This class defines the contract the rest of the system depends on
(train/predict/save/load). No regressor is trained or persisted by this
module itself — callers (scripts/train_module1.py) decide when training is
actually legitimate, per the target-validation gate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from mplads.module1_cost_overrun.preprocessing import (
    add_duration_features,
    build_preprocessing_pipeline,
    select_features,
)


@dataclass
class TrainingResult:
    n_samples: int
    trained_at: str
    model_version: str


class CostOverrunModel:
    """Wraps preprocessing + regressor into a single fit/predict unit."""

    def __init__(self, regressor: Optional[object] = None) -> None:
        self._regressor = regressor if regressor is not None else HistGradientBoostingRegressor(random_state=42)
        self._pipeline: Optional[Pipeline] = None
        self.trained_at: Optional[str] = None
        self.model_version: Optional[str] = None

    @property
    def is_fitted(self) -> bool:
        return self._pipeline is not None

    def train(self, features_df: pd.DataFrame, target: pd.Series) -> TrainingResult:
        """Fit the preprocessing pipeline + regressor on already-validated
        features/target. Callers are responsible for having run
        target_validation.validate_overrun_target on the source data first —
        this method does not re-check target legitimacy, it only fits."""
        prepared = select_features(add_duration_features(features_df))
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessing_pipeline()),
                ("regress", self._regressor),
            ]
        )
        pipeline.fit(prepared, target)
        self._pipeline = pipeline
        self.trained_at = datetime.now(timezone.utc).isoformat()
        self.model_version = self.trained_at
        return TrainingResult(
            n_samples=len(prepared), trained_at=self.trained_at, model_version=self.model_version
        )

    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("CostOverrunModel.predict() called before the model was trained/loaded.")
        prepared = select_features(add_duration_features(features_df))
        return pd.Series(self._pipeline.predict(prepared), index=prepared.index)

    def save(self, model_path: Path, metadata_path: Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save a model that has not been trained.")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, model_path)
        metadata_path.write_text(
            json.dumps(
                {"trained_at": self.trained_at, "model_version": self.model_version},
                indent=2,
            )
        )

    @classmethod
    def load(cls, model_path: Path, metadata_path: Path) -> "CostOverrunModel":
        instance = cls()
        instance._pipeline = joblib.load(model_path)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            instance.trained_at = metadata.get("trained_at")
            instance.model_version = metadata.get("model_version")
        return instance
