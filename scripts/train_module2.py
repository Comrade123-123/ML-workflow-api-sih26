#!/usr/bin/env python
"""Training entrypoint for Module 2 -- Delayed Project Prediction.

Order of operations (each stage must pass before the next runs):
  1. Load + structurally validate the source CSVs, join on Work ID
     (data_validation).
  2. Assign peer-relative delay labels from real Sanction->Completion
     durations (labeling) -- ON_TRACK / AT_RISK / LIKELY_DELAYED.
  3. Run the fail-fast class-balance gate (target_validation). If any class
     is under-populated, this raises TargetValidationError and the script
     stops here -- no fitting, no artifact, no class merged/dropped to force
     a pass.
  4. Only if validation passes: fit DelayRiskModel and persist it.

Run with:  python scripts/train_module2.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mplads.config import (  # noqa: E402
    MODULE2_METADATA_FILE,
    MODULE2_MIN_CLASS_SAMPLES,
    MODULE2_MODEL_FILE,
)
from mplads.module2_delay_prediction.data_validation import load_joined_training_frame  # noqa: E402
from mplads.module2_delay_prediction.exceptions import DataValidationError, TargetValidationError  # noqa: E402
from mplads.module2_delay_prediction.labeling import assign_status_labels  # noqa: E402
from mplads.module2_delay_prediction.model import DelayRiskModel  # noqa: E402
from mplads.module2_delay_prediction.target_validation import validate_class_balance  # noqa: E402


def main() -> int:
    print("=" * 80)
    print("Module 2 - Delayed Project Prediction: training pipeline")
    print("=" * 80)

    print("\n[1/4] Loading and structurally validating source data...")
    try:
        frame = load_joined_training_frame()
    except DataValidationError as exc:
        print(f"\nFAILED at data validation: {exc}", file=sys.stderr)
        return 1
    print(f"  -> Loaded {len(frame)} sanction-to-completion records (Work ID join).")

    print("\n[2/4] Assigning peer-relative delay labels...")
    labeled = assign_status_labels(frame)
    print(f"  -> Class distribution: {labeled['status'].value_counts().to_dict()}")

    print("\n[3/4] Running fail-fast class-balance validation...")
    try:
        report = validate_class_balance(labeled, min_samples_per_class=MODULE2_MIN_CLASS_SAMPLES)
    except TargetValidationError as exc:
        print("\nFAILED target validation - training will NOT proceed.", file=sys.stderr)
        print(f"Reason: {exc}", file=sys.stderr)
        return 2
    print(f"  -> {report.message}")

    print("\n[4/4] Training model...")
    target = labeled["status"]
    model = DelayRiskModel()
    result = model.train(labeled, target)
    model.save(MODULE2_MODEL_FILE, MODULE2_METADATA_FILE)
    print(f"  -> Trained on {result.n_samples} samples, classes={result.classes}.")
    print(f"  -> Saved to {MODULE2_MODEL_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
