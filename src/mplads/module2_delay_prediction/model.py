"""The Module 2 model interface: preprocessing + a swappable sklearn-compatible
classifier, matching Module 1's train/predict/save/load contract.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from mplads.config import MODULE2_STATUS_LABELS
from mplads.module2_delay_prediction.preprocessing import (
    add_processing_lag_feature,
    build_preprocessing_pipeline,
    select_features,
)


@dataclass
class TrainingResult:
    n_samples: int
    trained_at: str
    model_version: str
    classes: List[str]


class DelayRiskModel:
    """Wraps preprocessing + a RandomForestClassifier (default) into a single
    fit/predict unit. Swappable for LightGBM/XGBoost by passing a different
    classifier instance in -- the rest of the pipeline is unaffected."""

    def __init__(self, classifier: Optional[object] = None) -> None:
        self._classifier = classifier if classifier is not None else RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42, class_weight="balanced"
        )
        self._pipeline: Optional[Pipeline] = None
        self.trained_at: Optional[str] = None
        self.model_version: Optional[str] = None
        self.classes_: Optional[List[str]] = None

    @property
    def is_fitted(self) -> bool:
        return self._pipeline is not None

    def train(self, features_df: pd.DataFrame, target: pd.Series) -> TrainingResult:
        """Fit on already-labeled, already-validated data. Callers are
        responsible for having run target_validation.validate_class_balance
        first -- this method only fits."""
        prepared = select_features(add_processing_lag_feature(features_df))
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessing_pipeline()),
                ("classify", self._classifier),
            ]
        )
        pipeline.fit(prepared, target)
        self._pipeline = pipeline
        self.trained_at = datetime.now(timezone.utc).isoformat()
        self.model_version = self.trained_at
        self.classes_ = list(pipeline.named_steps["classify"].classes_)
        return TrainingResult(
            n_samples=len(prepared),
            trained_at=self.trained_at,
            model_version=self.model_version,
            classes=self.classes_,
        )

    def predict_proba(self, features_df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("DelayRiskModel.predict_proba() called before the model was trained/loaded.")
        prepared = select_features(add_processing_lag_feature(features_df))
        probabilities = self._pipeline.predict_proba(prepared)
        return pd.DataFrame(probabilities, columns=self.classes_, index=prepared.index)

    def predict(self, features_df: pd.DataFrame) -> pd.Series:
        if not self.is_fitted:
            raise RuntimeError("DelayRiskModel.predict() called before the model was trained/loaded.")
        prepared = select_features(add_processing_lag_feature(features_df))
        return pd.Series(self._pipeline.predict(prepared), index=prepared.index)

    def save(self, model_path: Path, metadata_path: Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save a model that has not been trained.")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, model_path, compress=3)
        metadata_path.write_text(
            json.dumps(
                {
                    "trained_at": self.trained_at,
                    "model_version": self.model_version,
                    "classes": self.classes_,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, model_path: Path, metadata_path: Path) -> "DelayRiskModel":
        instance = cls()
        instance._pipeline = joblib.load(model_path)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            instance.trained_at = metadata.get("trained_at")
            instance.model_version = metadata.get("model_version")
            instance.classes_ = metadata.get("classes", MODULE2_STATUS_LABELS)
        else:
            instance.classes_ = list(instance._pipeline.named_steps["classify"].classes_)
        return instance
