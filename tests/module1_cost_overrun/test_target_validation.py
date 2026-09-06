import pandas as pd
import pytest

from mplads.module1_cost_overrun.data_validation import load_joined_training_frame
from mplads.module1_cost_overrun.exceptions import TargetValidationError
from mplads.module1_cost_overrun.target_validation import (
    compute_overrun_pct,
    try_validate_overrun_target,
    validate_overrun_target,
)


def test_current_real_data_fails_target_validation():
    """The actual MPLADS dataset has zero genuine cost-overrun examples
    (disbursed never exceeds sanctioned). The fail-fast gate must reject it
    rather than silently substituting another target."""
    frame = load_joined_training_frame()

    with pytest.raises(TargetValidationError) as exc_info:
        validate_overrun_target(frame)

    message = str(exc_info.value)
    assert "Target validation FAILED" in message
    assert "positive-overrun" in message.lower() or "overrun" in message.lower()


def test_try_validate_returns_failing_report_without_raising():
    frame = load_joined_training_frame()
    report = try_validate_overrun_target(frame)
    assert report.is_valid is False
    assert report.positive_overrun_examples == 0
    assert report.total_examples > 0


def test_compute_overrun_pct_on_real_data_never_positive():
    frame = load_joined_training_frame()
    overrun_pct = compute_overrun_pct(frame)
    assert (overrun_pct <= 0).all()


def test_validate_overrun_target_passes_on_synthetic_data_with_enough_overrun():
    """Sanity check: the gate itself is not broken — it passes when given
    data that genuinely does contain enough positive-overrun examples."""
    n = 200
    sanctioned = [100_000.0] * n
    # 60% genuine overrun (well above default thresholds), rest at/under budget
    actual = [120_000.0] * 120 + [100_000.0] * 80
    df = pd.DataFrame(
        {
            "Recommended Amount (INR)": sanctioned,
            "Amount Disbursed (INR)": actual,
        }
    )
    report = validate_overrun_target(df, min_positive_examples=50, min_positive_ratio=0.01)
    assert report.is_valid is True
    assert report.positive_overrun_examples == 120


def test_validate_overrun_target_raises_below_absolute_threshold():
    df = pd.DataFrame(
        {
            "Recommended Amount (INR)": [100_000.0] * 100,
            "Amount Disbursed (INR)": [110_000.0] * 5 + [100_000.0] * 95,
        }
    )
    with pytest.raises(TargetValidationError):
        validate_overrun_target(df, min_positive_examples=50, min_positive_ratio=0.01)
