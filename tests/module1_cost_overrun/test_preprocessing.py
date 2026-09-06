import pandas as pd

from mplads.module1_cost_overrun.preprocessing import (
    add_duration_features,
    build_preprocessing_pipeline,
    select_features,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Work Category": ["Normal/Others", "Repair and Renovation"],
            "Work Type": ["Roads", None],
            "State": ["Bihar", "Kerala"],
            "IDA District": ["ARARIA", "KOLLAM"],
            "Recommended Amount (INR)": [500000.0, 300000.0],
            "Recommended Date": ["2024-07-08", "2024-08-01"],
            "Sanction Date": ["2024-07-09", None],
            "Completion Date": ["2024-09-05", "2024-10-01"],
        }
    )


def test_add_duration_features_computes_day_deltas():
    out = add_duration_features(sample_frame())
    assert out.loc[0, "days_recommended_to_sanction"] == 1
    assert pd.isna(out.loc[1, "days_recommended_to_sanction"])
    assert out.loc[0, "days_sanction_to_completion"] == 58


def test_select_features_adds_missing_columns_as_na():
    minimal = pd.DataFrame({"Work Category": ["Normal/Others"]})
    out = select_features(add_duration_features(minimal))
    assert list(out.columns) == [
        "Work Category",
        "Work Type",
        "State",
        "IDA District",
        "Recommended Amount (INR)",
        "days_recommended_to_sanction",
        "days_sanction_to_completion",
    ]
    assert out["Work Type"].isna().all()


def test_pipeline_fits_and_transforms_without_a_target():
    prepared = select_features(add_duration_features(sample_frame()))
    pipeline = build_preprocessing_pipeline()
    transformed = pipeline.fit_transform(prepared)
    assert transformed.shape[0] == 2
    assert transformed.shape[1] > 0
