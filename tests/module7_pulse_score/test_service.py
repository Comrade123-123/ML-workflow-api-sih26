import json

import pandas as pd
import pytest

from mplads.config import ALLOCATED_LIMIT_RAJYASABHA_CSV, WORKS_RECOMMENDED_CSV
from mplads.module7_pulse_score import service
from mplads.module7_pulse_score.exceptions import EntityNotFoundError


@pytest.fixture
def real_entity_pair():
    """A genuine (constituency, mp_name) pair pulled from the actual data,
    rather than hardcoded, so the test stays valid if the CSV changes."""
    df = pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)
    row = df.iloc[0]
    return row["Constituency"], row["Hon'ble Members of Parliament"]


@pytest.fixture(autouse=True)
def isolated_history(tmp_path, monkeypatch):
    """Redirect the snapshot store to a temp file so tests never read/write
    the real production history and are independent of run order."""
    history_path = tmp_path / "pulse_history.json"
    monkeypatch.setattr(service, "MODULE7_HISTORY_FILE", history_path)
    return history_path


def test_resolve_entity_accepts_constituency_name(real_entity_pair):
    constituency, _ = real_entity_pair
    recommended = pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)
    resolved = service.resolve_entity(constituency.lower(), recommended)
    assert resolved == constituency


def test_resolve_entity_accepts_mp_name(real_entity_pair):
    constituency, mp_name = real_entity_pair
    recommended = pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)
    resolved = service.resolve_entity(mp_name.lower(), recommended)
    assert resolved == constituency


def test_resolve_entity_unknown_raises():
    recommended = pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)
    with pytest.raises(EntityNotFoundError):
        service.resolve_entity("Definitely Not A Real Constituency", recommended)


def test_resolve_entity_rajya_sabha_mp_raises():
    """Rajya Sabha MPs have zero overlap with Works_Recommended's MP names
    (confirmed during inspection) -- must 404, not silently match."""
    raj = pd.read_csv(ALLOCATED_LIMIT_RAJYASABHA_CSV, low_memory=False)
    recommended = pd.read_csv(WORKS_RECOMMENDED_CSV, low_memory=False)
    rajya_sabha_mp = raj.iloc[0]["MP Name"]
    with pytest.raises(EntityNotFoundError):
        service.resolve_entity(rajya_sabha_mp, recommended)


def test_get_pulse_score_on_real_entity_returns_well_formed_response(real_entity_pair):
    constituency, _ = real_entity_pair
    response = service.get_pulse_score(constituency)

    assert response.entity_id == constituency
    assert 0.0 <= response.pulse_score <= 100.0
    assert response.total_works >= response.completed_works >= 0
    assert 0 <= response.delayed_works_count <= response.total_works
    assert response.health_band in {"EXCELLENT", "GOOD", "NEEDS_ATTENTION", "CRITICAL"}


def test_get_pulse_score_first_call_has_zero_delta(real_entity_pair):
    constituency, _ = real_entity_pair
    response = service.get_pulse_score(constituency)
    assert response.delta_vs_last_period == 0.0


def test_get_pulse_score_second_call_reflects_snapshot_delta(real_entity_pair, isolated_history):
    constituency, _ = real_entity_pair

    first = service.get_pulse_score(constituency)
    assert first.delta_vs_last_period == 0.0

    # Tamper with the stored snapshot to simulate a prior period with a
    # different score, then confirm the delta reflects that difference.
    history = json.loads(isolated_history.read_text())
    history[constituency]["last_score"] = first.pulse_score - 5.0
    isolated_history.write_text(json.dumps(history))

    second = service.get_pulse_score(constituency)
    assert second.delta_vs_last_period == pytest.approx(5.0, abs=0.15)


def test_get_pulse_score_unknown_entity_raises():
    with pytest.raises(EntityNotFoundError):
        service.get_pulse_score("Nonexistent Constituency XYZ")
