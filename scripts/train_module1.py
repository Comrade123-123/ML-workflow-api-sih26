#!/usr/bin/env python
"""Training entrypoint for Module 1 — Cost Overrun Detection.

Order of operations (each stage must pass before the next runs):
  1. Load + structurally validate the source CSVs (data_validation).
  2. Run the fail-fast target-validation gate (target_validation). If the
     historical data does not contain enough genuine positive-overrun
     examples, this raises TargetValidationError and the script stops here
     — no preprocessing, no fitting, no artifact is written, and no
     alternative target is substituted.
  3. Only if validation passes: fit CostOverrunModel and persist it.

Run with:  python scripts/train_module1.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mplads.config import (  # noqa: E402
    COL_AMOUNT_DISBURSED,
    MIN_OVERRUN_RATIO,
    MIN_OVERRUN_SAMPLES,
    MODULE1_METADATA_FILE,
    MODULE1_MODEL_FILE,
)
from mplads.module1_cost_overrun.data_validation import load_joined_training_frame  # noqa: E402
from mplads.module1_cost_overrun.exceptions import DataValidationError, TargetValidationError  # noqa: E402
from mplads.module1_cost_overrun.model import CostOverrunModel  # noqa: E402
from mplads.module1_cost_overrun.target_validation import validate_overrun_target  # noqa: E402


def main() -> int:
    print("=" * 80)
    print("Module 1 - Cost Overrun Detection: training pipeline")
    print("=" * 80)

    print("\n[1/3] Loading and structurally validating source data...")
    try:
        frame = load_joined_training_frame()
    except DataValidationError as exc:
        print(f"\nFAILED at data validation: {exc}", file=sys.stderr)
        return 1
    print(f"  -> Loaded {len(frame)} sanctioned/actual-cost pairs (Work ID join).")

    print("\n[2/3] Running fail-fast target validation...")
    try:
        report = validate_overrun_target(
            frame,
            min_positive_examples=MIN_OVERRUN_SAMPLES,
            min_positive_ratio=MIN_OVERRUN_RATIO,
        )
    except TargetValidationError as exc:
        print("\nFAILED target validation - training will NOT proceed.", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        print(
            "\nNo model has been trained or saved. This is expected behavior: "
            "the current MPLADS dataset does not contain legitimate cost-overrun "
            "history. Supply a dataset with real overrun examples "
            "(e.g. a revised-sanction/cost-escalation table) to proceed.",
            file=sys.stderr,
        )
        return 2

    print(f"  -> {report.message}")

    print("\n[3/3] Training model...")
    target = frame[COL_AMOUNT_DISBURSED]
    model = CostOverrunModel()
    result = model.train(frame, target)
    model.save(MODULE1_MODEL_FILE, MODULE1_METADATA_FILE)
    print(f"  -> Trained on {result.n_samples} samples. Saved to {MODULE1_MODEL_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
