import numpy as np
import pandas as pd
import pytest

from mplads.underspend_risk.exceptions import ModelNotAvailableError
from mplads.underspend_risk.historical_rates import HistoricalRateLookup
from mplads.underspend_risk.inference import predict_underspend_risk
from mplads.underspend_risk.model import UnderspendRiskModel
from mplads.underspend_risk.registry import is_model_available, load_active_model
from mplads.underspend_risk.schemas import PredictRequest


def test_registry_reports_available_and_loadable_after_training():
    """A legitimate artifact has been trained and persisted (see
    scripts/train_underspend_risk.py) -- the registry must load it
    correctly through the real production path."""
    assert is_model_available() is True
    model = load_active_model()
    assert model is not None
    assert model.is_fitted is True
    assert model.model_version is not None


def test_predict_underspend_risk_raises_when_no_model_available(monkeypatch):
    """Simulates unavailability via dependency injection rather than
    ambient repo state, since a real artifact is now legitimately persisted
    (see test_registry_reports_available_and_loadable_after_training)."""
    import mplads.underspend_risk.inference as inference_module

    monkeypatch.setattr(inference_module, "load_active_model", lambda: None)

    request = PredictRequest(
        work_id="WS/MP1/2024-2025/000001",
        recommended_amount=500_000,
        work_category="Normal/Others",
        work_type="Roads",
        state="Bihar",
        recommended_date="2024-07-08",
        sanction_date="2024-07-20",
    )
    with pytest.raises(ModelNotAvailableError):
        predict_underspend_risk(request)


def test_model_predict_before_fit_raises():
    model = UnderspendRiskModel()
    with pytest.raises(RuntimeError):
        model.predict_proba(pd.DataFrame({"State": ["Bihar"]}))


def test_model_train_without_rate_lookup_raises():
    model = UnderspendRiskModel()
    df = pd.DataFrame({"State": ["Bihar"], "Work Type": ["Roads"], "Work Category": ["Normal/Others"],
                        "Recommended Amount (INR)": [100_000.0], "Recommended Date": ["2024-01-01"],
                        "Sanction Date": ["2024-01-05"]})
    with pytest.raises(RuntimeError):
        model.train(df, pd.Series([0]))


def _synthetic_training_frame(n: int = 40) -> pd.DataFrame:
    """Purely synthetic, in-memory-only data used to exercise the model
    interface's mechanics (train/predict_proba). Never persisted to
    models/ -- mirrors Module 1's test_model_train_and_predict_roundtrip_in_memory."""
    rng = np.random.default_rng(42)
    states = rng.choice(["Bihar", "Telangana"], size=n)
    return pd.DataFrame(
        {
            "Work Category": ["Normal/Others"] * n,
            "Work Type": ["Roads"] * n,
            "State": states,
            "Recommended Amount (INR)": rng.uniform(100_000, 900_000, size=n),
            "Recommended Date": ["2024-07-08"] * n,
            "Sanction Date": ["2024-07-20"] * n,
        }
    )


def test_model_train_and_predict_roundtrip_in_memory():
    """The interface itself (fit rate lookup -> train -> predict_proba)
    works correctly given a legitimate target shape. Synthetic data only,
    exercising mechanics -- not a claim of a validated production model."""
    df = _synthetic_training_frame()
    target = pd.Series([0, 1] * (len(df) // 2))

    rate_lookup = HistoricalRateLookup.fit(
        df.assign(has_underspend=target), state_col="State", work_type_col="Work Type", target_col="has_underspend"
    )
    model = UnderspendRiskModel(rate_lookup=rate_lookup)
    result = model.train(df, target)

    assert result.n_samples == len(df)
    assert model.is_fitted is True

    scores = model.predict_proba(df)
    assert len(scores) == len(df)
    assert scores.between(0, 1).all()


def test_predict_proba_is_deterministic_for_identical_input():
    df = _synthetic_training_frame()
    target = pd.Series([0, 1] * (len(df) // 2))
    rate_lookup = HistoricalRateLookup.fit(df.assign(has_underspend=target), "State", "Work Type", "has_underspend")
    model = UnderspendRiskModel(rate_lookup=rate_lookup)
    model.train(df, target)

    one_row = df.iloc[[0]]
    score1 = model.predict_proba(one_row).iloc[0]
    score2 = model.predict_proba(one_row).iloc[0]
    assert score1 == score2


def test_injected_model_produces_valid_prediction_via_inference_layer():
    """Exercises the full inference.predict_underspend_risk path with an
    explicitly injected (test-only, in-memory, never-persisted) model --
    the same dependency-injection pattern Module 1's inference layer uses
    (`model` param) to make the HTTP-independent prediction logic testable
    without a real deployed artifact."""
    df = _synthetic_training_frame()
    target = pd.Series([0, 1] * (len(df) // 2))
    rate_lookup = HistoricalRateLookup.fit(df.assign(has_underspend=target), "State", "Work Type", "has_underspend")
    model = UnderspendRiskModel(rate_lookup=rate_lookup)
    model.train(df, target)

    request = PredictRequest(
        work_id="WS/MP1/2024-2025/000001",
        recommended_amount=500_000,
        work_category="Normal/Others",
        work_type="Roads",
        state="Bihar",
        recommended_date="2024-07-08",
        sanction_date="2024-07-20",
    )
    response = predict_underspend_risk(request, model=model)
    assert response.work_id == "WS/MP1/2024-2025/000001"
    assert 0.0 <= response.risk_score <= 1.0
    assert response.risk_level in ("TYPICAL", "ELEVATED")
    assert "NOT" in response.disclaimer
    assert "fraud" in response.disclaimer.lower()
