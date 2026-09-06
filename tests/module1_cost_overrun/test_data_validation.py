import pandas as pd
import pytest

from mplads.module1_cost_overrun.data_validation import (
    load_joined_training_frame,
    validate_completed_frame,
    validate_join,
    validate_recommended_frame,
)
from mplads.module1_cost_overrun.exceptions import DataValidationError


def test_load_joined_training_frame_on_real_data():
    """The real Data/ CSVs should pass structural validation and join."""
    frame = load_joined_training_frame()
    assert len(frame) > 0
    assert "Recommended Amount (INR)" in frame.columns
    assert "Amount Disbursed (INR)" in frame.columns


def test_validate_recommended_frame_missing_column_raises():
    df = pd.DataFrame({"Work ID": ["a"], "Work Category": ["x"]})
    with pytest.raises(DataValidationError):
        validate_recommended_frame(df)


def test_validate_recommended_frame_non_numeric_amount_raises():
    df = pd.DataFrame(
        {
            "Work ID": ["a"],
            "Work Category": ["x"],
            "Work Type": ["y"],
            "State": ["s"],
            "IDA District": ["d"],
            "Recommended Date": ["2024-01-01"],
            "Recommended Amount (INR)": ["not-a-number"],
            "Sanction Date": ["2024-01-02"],
        }
    )
    with pytest.raises(DataValidationError):
        validate_recommended_frame(df)


def test_validate_completed_frame_empty_raises():
    with pytest.raises(DataValidationError):
        validate_completed_frame(pd.DataFrame())


def test_validate_join_no_overlap_raises():
    recommended = pd.DataFrame({"Work ID": ["a"], "Recommended Amount (INR)": [100.0]})
    completed = pd.DataFrame(
        {
            "Work ID": ["z"],
            "Work Category": ["x"],
            "Work Type": ["y"],
            "State": ["s"],
            "IDA District": ["d"],
            "Completion Date": ["2024-01-01"],
            "Amount Disbursed (INR)": [100.0],
        }
    )
    with pytest.raises(DataValidationError):
        validate_join(recommended, completed)
