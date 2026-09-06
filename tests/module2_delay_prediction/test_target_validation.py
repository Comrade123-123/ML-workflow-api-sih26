import pandas as pd
import pytest

from mplads.module2_delay_prediction.data_validation import load_joined_training_frame
from mplads.module2_delay_prediction.exceptions import TargetValidationError
from mplads.module2_delay_prediction.labeling import assign_status_labels
from mplads.module2_delay_prediction.target_validation import (
    try_validate_class_balance,
    validate_class_balance,
)


def test_current_real_data_passes_class_balance():
    """Unlike Module 1, the peer-relative percentile construction guarantees
    a real minority class, so validation should PASS on the actual dataset
    and training should be able to proceed."""
    frame = load_joined_training_frame()
    labeled = assign_status_labels(frame)

    report = validate_class_balance(labeled)
    assert report.is_valid is True
    assert all(count >= 50 for count in report.class_counts.values())


def test_try_validate_returns_report_without_raising():
    frame = load_joined_training_frame()
    labeled = assign_status_labels(frame)
    report = try_validate_class_balance(labeled)
    assert report.is_valid is True


def test_validate_class_balance_raises_on_under_populated_class():
    df = pd.DataFrame(
        {
            "status": ["ON_TRACK"] * 200 + ["AT_RISK"] * 190 + ["LIKELY_DELAYED"] * 2,
        }
    )
    with pytest.raises(TargetValidationError):
        validate_class_balance(df, min_samples_per_class=50)


def test_validate_class_balance_passes_when_all_classes_sufficient():
    df = pd.DataFrame(
        {
            "status": ["ON_TRACK"] * 200 + ["AT_RISK"] * 100 + ["LIKELY_DELAYED"] * 60,
        }
    )
    report = validate_class_balance(df, min_samples_per_class=50)
    assert report.is_valid is True
    assert report.class_counts == {"ON_TRACK": 200, "AT_RISK": 100, "LIKELY_DELAYED": 60}
