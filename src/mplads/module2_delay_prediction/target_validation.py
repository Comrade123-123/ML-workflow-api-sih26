"""The fail-fast target-validation gate for Module 2.

After peer-relative labeling (labeling.assign_status_labels), this verifies
every one of the 3 classes has enough examples to train on. It never
merges/drops a class or substitutes a different scheme to force a pass --
if a class is under-populated, training must stop.
"""
from __future__ import annotations

import pandas as pd

from mplads.config import MODULE2_MIN_CLASS_SAMPLES, MODULE2_STATUS_LABELS
from mplads.module2_delay_prediction.exceptions import TargetValidationError
from mplads.module2_delay_prediction.schemas import TargetValidationReport


def validate_class_balance(
    labeled_df: pd.DataFrame,
    min_samples_per_class: int = MODULE2_MIN_CLASS_SAMPLES,
) -> TargetValidationReport:
    """Verify the peer-relative labels support training a 3-class classifier.

    Raises TargetValidationError if any of ON_TRACK/AT_RISK/LIKELY_DELAYED
    has fewer than min_samples_per_class examples.
    """
    if "status" not in labeled_df.columns:
        raise TargetValidationError("Labeled data has no 'status' column to validate.")

    total = len(labeled_df)
    counts = labeled_df["status"].value_counts()
    class_counts = {label: int(counts.get(label, 0)) for label in MODULE2_STATUS_LABELS}

    under_populated = {
        label: count for label, count in class_counts.items() if count < min_samples_per_class
    }

    if under_populated:
        message = (
            f"Target validation FAILED: class(es) {under_populated} have fewer than "
            f"{min_samples_per_class} examples out of {total} total labeled records "
            f"(full class distribution: {class_counts}). A 3-class delay classifier "
            "cannot be trained reliably on an under-populated class -- training will "
            "not proceed, and no classes will be merged or dropped to force a pass."
        )
        raise TargetValidationError(message)

    message = (
        f"Target validation PASSED: class distribution {class_counts} out of {total} "
        f"total records, each meeting the minimum of {min_samples_per_class} samples."
    )
    return TargetValidationReport(
        is_valid=True,
        total_examples=total,
        class_counts=class_counts,
        min_required_samples_per_class=min_samples_per_class,
        message=message,
    )


def try_validate_class_balance(
    labeled_df: pd.DataFrame,
    min_samples_per_class: int = MODULE2_MIN_CLASS_SAMPLES,
) -> TargetValidationReport:
    """Non-raising variant for status reporting."""
    try:
        return validate_class_balance(labeled_df, min_samples_per_class)
    except TargetValidationError as exc:
        total = len(labeled_df)
        if "status" in labeled_df.columns:
            counts = labeled_df["status"].value_counts()
            class_counts = {label: int(counts.get(label, 0)) for label in MODULE2_STATUS_LABELS}
        else:
            class_counts = {label: 0 for label in MODULE2_STATUS_LABELS}
        return TargetValidationReport(
            is_valid=False,
            total_examples=total,
            class_counts=class_counts,
            min_required_samples_per_class=min_samples_per_class,
            message=str(exc),
        )
