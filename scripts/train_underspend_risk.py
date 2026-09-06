"""Trains and persists the underspend_risk model artifact, reproducing
scripts/mplads_option2_pipeline/train_underspend_model.py's already-validated
methodology exactly (same features, same RandomForestClassifier config, same
target definition) -- see OPTION2_MPLADS_TARGET_FEASIBILITY_REPORT.md for
the full validation record.

NOT executed as part of the underspend_risk implementation itself: this
script exists so the missing artifact can be produced on explicit request,
rather than being materialized silently. Running it is a one-way action
(creates a new production model file) and was deliberately left for a
separate, explicit decision.

Usage: python scripts/train_underspend_risk.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mplads.config import (  # noqa: E402
    UNDERSPEND_RISK_METADATA_FILE,
    UNDERSPEND_RISK_MODEL_FILE,
    UNDERSPEND_RISK_RATE_LOOKUP_FILE,
)
from mplads.underspend_risk.historical_rates import HistoricalRateLookup  # noqa: E402
from mplads.underspend_risk.model import UnderspendRiskModel  # noqa: E402

# Reuses the already-built, already-validated research dataset directly --
# does not rebuild it, to guarantee this artifact matches exactly what was
# validated (same 14,965 completed works, same has_underspend definition).
RESEARCH_DATASET = Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\data\mplads_option2\datasets\work_level_completed_dataset.csv")

TARGET = "has_underspend"


def main() -> None:
    if not RESEARCH_DATASET.exists():
        raise SystemExit(
            f"Research dataset not found at {RESEARCH_DATASET}. This script "
            "intentionally does not rebuild it from raw CSVs -- rerun "
            "scripts/mplads_option2_pipeline/build_dataset.py first if it is "
            "genuinely missing, so the artifact traces to the exact "
            "validated dataset, not a silently-rebuilt one."
        )

    df = pd.read_csv(RESEARCH_DATASET)
    df[TARGET] = df[TARGET].astype(int)

    # Fixed lookup table computed from the FULL training set (see
    # historical_rates.py's module docstring for why this differs from the
    # causal walk-forward computation used in the research backtest).
    rate_lookup = HistoricalRateLookup.fit(df, state_col="State", work_type_col="Work Type", target_col=TARGET)

    model = UnderspendRiskModel(rate_lookup=rate_lookup)
    result = model.train(df, df[TARGET])
    print(f"Trained on {result.n_samples} samples at {result.trained_at} (version {result.model_version})")

    model.save(UNDERSPEND_RISK_MODEL_FILE, UNDERSPEND_RISK_RATE_LOOKUP_FILE, UNDERSPEND_RISK_METADATA_FILE)
    print(f"Saved model to {UNDERSPEND_RISK_MODEL_FILE}")
    print(f"Saved rate lookup to {UNDERSPEND_RISK_RATE_LOOKUP_FILE}")
    print(f"Saved metadata to {UNDERSPEND_RISK_METADATA_FILE}")


if __name__ == "__main__":
    main()
