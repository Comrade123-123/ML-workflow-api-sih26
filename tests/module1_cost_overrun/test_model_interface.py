import pandas as pd
import pytest

from mplads.module1_cost_overrun.exceptions import ModelNotAvailableError
from mplads.module1_cost_overrun.inference import predict_cost_overrun
from mplads.module1_cost_overrun.model import CostOverrunModel
from mplads.module1_cost_overrun.registry import is_model_available, load_active_model
from mplads.module1_cost_overrun.schemas import PredictRequest


def test_registry_reports_unavailable_when_no_artifact_persisted():
    """No training has been run in this repo (per instructions), so the
    registry must report unavailability rather than inventing a model."""
    assert is_model_available() is False
    assert load_active_model() is None


def test_predict_cost_overrun_raises_when_no_model_available():
    request = PredictRequest(
        project_id="AIP-802",
        sanctioned_cost=2_000_000,
        work_category="Normal/Others",
        state="Bihar",
    )
    with pytest.raises(ModelNotAvailableError):
        predict_cost_overrun(request)


def test_model_predict_before_fit_raises():
    model = CostOverrunModel()
    with pytest.raises(RuntimeError):
        model.predict(pd.DataFrame({"Work Category": ["x"]}))


def test_model_train_and_predict_roundtrip_in_memory():
    """The interface itself (train/predict) works correctly given a
    legitimate target — this uses synthetic data purely to exercise the
    mechanics, and is never persisted to models/."""
    df = pd.DataFrame(
        {
            "Work Category": ["Normal/Others"] * 20,
            "Work Type": ["Roads"] * 20,
            "State": ["Bihar"] * 20,
            "IDA District": ["ARARIA"] * 20,
            "Recommended Amount (INR)": [100_000.0] * 20,
            "Recommended Date": ["2024-01-01"] * 20,
            "Sanction Date": ["2024-01-05"] * 20,
            "Completion Date": ["2024-03-01"] * 20,
        }
    )
    target = pd.Series([105_000.0] * 20)
    model = CostOverrunModel()
    result = model.train(df, target)
    assert result.n_samples == 20
    assert model.is_fitted is True

    predictions = model.predict(df)
    assert len(predictions) == 20
