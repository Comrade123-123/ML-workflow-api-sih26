"""The underspend_risk model interface: preprocessing + a swappable
sklearn-compatible classifier, plus the historical-rate lookup table it
depends on -- matching Module 1/2's train/predict/save/load contract.

The classifier default (RandomForestClassifier, class_weight="balanced")
matches the best-PR-AUC configuration actually evaluated in
OPTION2_MPLADS_TARGET_FEASIBILITY_REPORT.md section 11 -- not a new choice
made here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from mplads.underspend_risk.feature_pipeline import (
    add_derived_features,
    build_preprocessing_pipeline,
    select_features,
)
from mplads.underspend_risk.historical_rates import HistoricalRateLookup


@dataclass
class TrainingResult:
    n_samples: int
    trained_at: str
    model_version: str


class UnderspendRiskModel:
    """Wraps preprocessing + a RandomForestClassifier (default) + the
    historical-rate lookup into a single fit/predict unit."""

    def __init__(self, classifier: Optional[object] = None, rate_lookup: Optional[HistoricalRateLookup] = None) -> None:
        self._classifier = classifier if classifier is not None else RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=42
        )
        self._pipeline: Optional[Pipeline] = None
        self.rate_lookup: Optional[HistoricalRateLookup] = rate_lookup
        self.trained_at: Optional[str] = None
        self.model_version: Optional[str] = None

    @property
    def is_fitted(self) -> bool:
        return self._pipeline is not None and self.rate_lookup is not None

    def train(self, features_df: pd.DataFrame, target: pd.Series) -> TrainingResult:
        """Fit on already-validated data. `features_df` must contain the raw
        columns add_derived_features expects (Recommended Amount (INR),
        Recommended Date, Sanction Date, State, Work Type, Work Category);
        `self.rate_lookup` must already be set (via HistoricalRateLookup.fit
        on this same training set) before calling this."""
        if self.rate_lookup is None:
            raise RuntimeError("rate_lookup must be set (see HistoricalRateLookup.fit) before training.")
        prepared = select_features(add_derived_features(features_df, self.rate_lookup))
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
        return TrainingResult(n_samples=len(prepared), trained_at=self.trained_at, model_version=self.model_version)

    def predict_proba(self, raw_df: pd.DataFrame) -> pd.Series:
        """Returns P(has_underspend=1) per row. `raw_df` has the same raw
        columns as `train()` -- derived features are computed internally."""
        if not self.is_fitted:
            raise RuntimeError("UnderspendRiskModel.predict_proba() called before the model was trained/loaded.")
        prepared = select_features(add_derived_features(raw_df, self.rate_lookup))
        proba = self._pipeline.predict_proba(prepared)
        classes = list(self._pipeline.named_steps["classify"].classes_)
        positive_idx = classes.index(1) if 1 in classes else classes.index(True)
        return pd.Series(proba[:, positive_idx], index=prepared.index)

    def save(self, model_path: Path, rate_lookup_path: Path, metadata_path: Path) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save a model that has not been trained.")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, model_path, compress=3)
        self.rate_lookup.save(rate_lookup_path)
        metadata_path.write_text(
            json.dumps({"trained_at": self.trained_at, "model_version": self.model_version}, indent=2)
        )

    @classmethod
    def load(cls, model_path: Path, rate_lookup_path: Path, metadata_path: Path) -> "UnderspendRiskModel":
        instance = cls()
        instance._pipeline = joblib.load(model_path)
        instance.rate_lookup = HistoricalRateLookup.load(rate_lookup_path)
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            instance.trained_at = metadata.get("trained_at")
            instance.model_version = metadata.get("model_version")
        return instance
