import pandas as pd
import pytest

from mplads.underspend_risk.feature_pipeline import (
    ALL_FEATURES,
    FEATURE_COLUMNS_CATEGORICAL,
    FEATURE_COLUMNS_NUMERIC,
    add_derived_features,
    select_features,
)
from mplads.underspend_risk.historical_rates import HistoricalRateLookup
from mplads.underspend_risk.schemas import PredictRequest

LOOKUP = HistoricalRateLookup(
    state_rates={"Bihar": 0.10, "Telangana": 0.28},
    work_type_rates={"Roads": 0.05},
    global_prior=0.0571,
)


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Work Category": ["Normal/Others"],
            "Work Type": ["Roads"],
            "State": ["Bihar"],
            "Recommended Amount (INR)": [500_000.0],
            "Recommended Date": ["2024-07-08"],
            "Sanction Date": ["2024-07-20"],
        }
    )


def test_add_derived_features_computes_expected_values():
    out = add_derived_features(_sample_frame(), LOOKUP)
    assert out.loc[0, "recommendation_to_sanction_days"] == 12
    assert out.loc[0, "state_historical_underspend_rate"] == 0.10
    assert out.loc[0, "work_type_historical_underspend_rate"] == 0.05
    assert out.loc[0, "recommended_amount_log"] == pytest.approx(13.122, abs=1e-3)


def test_unknown_state_and_work_type_fall_back_to_global_prior():
    df = _sample_frame()
    df["State"] = "Nowhereland"
    df["Work Type"] = "Nonexistent Type"
    out = add_derived_features(df, LOOKUP)
    assert out.loc[0, "state_historical_underspend_rate"] == LOOKUP.global_prior
    assert out.loc[0, "work_type_historical_underspend_rate"] == LOOKUP.global_prior


def test_feature_ordering_is_deterministic_and_matches_research():
    out = select_features(add_derived_features(_sample_frame(), LOOKUP))
    assert list(out.columns) == ALL_FEATURES
    assert ALL_FEATURES == FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL


def test_select_features_is_deterministic_across_repeated_calls():
    df = _sample_frame()
    out1 = select_features(add_derived_features(df, LOOKUP))
    out2 = select_features(add_derived_features(df, LOOKUP))
    pd.testing.assert_frame_equal(out1, out2)


def test_select_features_fills_missing_columns_deterministically():
    minimal = pd.DataFrame({"State": ["Bihar"]})
    out = select_features(minimal)
    assert list(out.columns) == ALL_FEATURES
    assert out["Recommended Amount (INR)"].isna().all()


def test_missing_dates_produce_null_duration_not_an_error():
    df = _sample_frame()
    df["Sanction Date"] = [None]
    out = add_derived_features(df, LOOKUP)
    assert pd.isna(out.loc[0, "recommendation_to_sanction_days"])


def test_predict_request_schema_has_no_leakage_fields():
    """Structural guarantee: it must be impossible to even submit a
    post-outcome field (Completion Date, Amount Disbursed, has_underspend,
    Has Image) -- not just that the code ignores them if present."""
    field_names = set(PredictRequest.model_fields.keys())
    leakage_terms = {"completion", "disbursed", "has_underspend", "has_image", "underspend_pct", "ratio_disbursed"}
    for field in field_names:
        lowered = field.lower()
        assert not any(term in lowered for term in leakage_terms), f"leakage-shaped field found: {field}"


def test_predict_request_rejects_non_positive_recommended_amount():
    with pytest.raises(Exception):
        PredictRequest(
            work_id="WS/MP1/2024-2025/000001",
            recommended_amount=0,
            work_category="Normal/Others",
            work_type="Roads",
            state="Bihar",
            recommended_date="2024-07-08",
            sanction_date="2024-07-20",
        )
