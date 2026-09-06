import pandas as pd
import pytest

from mplads.module2_delay_prediction.exceptions import ModelNotAvailableError
from mplads.module2_delay_prediction.inference import predict_delay_risk
from mplads.module2_delay_prediction.model import DelayRiskModel
from mplads.module2_delay_prediction.registry import is_model_available, load_active_model
from mplads.module2_delay_prediction.schemas import PredictRequest


def _sample_request() -> PredictRequest:
    return PredictRequest(
        project_id="AIP-802",
        sanctioned_cost=2_000_000,
        work_category="Normal/Others",
        state="Bihar",
    )


def test_registry_state_is_consistent():
    """Before scripts/train_module2.py has run, no artifact exists and the
    registry must report unavailability rather than inventing a model.
    After a legitimate training run, it must report availability."""
    available = is_model_available()
    model = load_active_model()
    assert available == (model is not None)


def test_predict_delay_risk_matches_registry_state():
    """If no model is available, prediction must raise rather than
    fabricate a result. If a model IS available (a real one was legitimately
    trained by scripts/train_module2.py), prediction must succeed and
    return a well-formed response."""
    request = _sample_request()
    if not is_model_available():
        with pytest.raises(ModelNotAvailableError):
            predict_delay_risk(request)
    else:
        response = predict_delay_risk(request)
        assert response.project_id == "AIP-802"
        assert 0.0 <= response.delay_probability <= 1.0
        assert response.status in {"ON_TRACK", "AT_RISK", "LIKELY_DELAYED"}


def test_model_predict_before_fit_raises():
    model = DelayRiskModel()
    with pytest.raises(RuntimeError):
        model.predict(pd.DataFrame({"Work Category": ["x"]}))
    with pytest.raises(RuntimeError):
        model.predict_proba(pd.DataFrame({"Work Category": ["x"]}))


def test_model_train_and_predict_roundtrip_in_memory():
    """Exercises the train/predict mechanics with synthetic data that has a
    clean 3-class split -- never persisted to models/."""
    n = 30
    df = pd.DataFrame(
        {
            "Work Category": ["Normal/Others"] * (n * 3),
            "Work Type": ["Roads"] * (n * 3),
            "State": ["Bihar"] * (n * 3),
            "IDA District": ["ARARIA"] * (n * 3),
            "Recommended Amount (INR)": [100_000.0] * (n * 3),
            "Recommended Date": ["2024-01-01"] * (n * 3),
            "Sanction Date": ["2024-01-05"] * (n * 3),
        }
    )
    target = pd.Series(["ON_TRACK"] * n + ["AT_RISK"] * n + ["LIKELY_DELAYED"] * n)
    model = DelayRiskModel()
    result = model.train(df, target)
    assert result.n_samples == n * 3
    assert set(result.classes) == {"ON_TRACK", "AT_RISK", "LIKELY_DELAYED"}

    probabilities = model.predict_proba(df)
    assert list(probabilities.columns.sort_values()) == ["AT_RISK", "LIKELY_DELAYED", "ON_TRACK"]
    assert (probabilities.sum(axis=1).round(4) == 1.0).all()
