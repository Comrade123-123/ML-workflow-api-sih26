import pytest
from pydantic import ValidationError

from mplads.module1b_gemini_risk.schemas import AssessRequest, StatusResponse


def _valid_payload(**overrides):
    payload = {
        "project_id": "WS/MP620/2024-2025/133166",
        "work_category": "Normal/Others",
        "work_type": "Construction of buildings for community cultural activities",
        "state": "Karnataka",
        "recommended_amount": 497185.0,
        "recommendation_date": "2024-07-08",
        "sanction_date": "2024-07-09",
    }
    payload.update(overrides)
    return payload


def test_valid_request_with_only_required_fields():
    req = AssessRequest(**_valid_payload())
    assert req.project_id == "WS/MP620/2024-2025/133166"
    assert req.work_description is None
    assert req.constituency is None
    assert req.sanctioned_amount is None


def test_valid_request_with_all_optional_fields():
    req = AssessRequest(**_valid_payload(
        work_description="Construction of Community Bhavan",
        constituency="DHARWAD",
        sanctioned_amount=497185.0,
    ))
    assert req.work_description == "Construction of Community Bhavan"
    assert req.sanctioned_amount == 497185.0


@pytest.mark.parametrize("field", ["project_id", "work_category", "work_type", "state", "recommended_amount", "recommendation_date", "sanction_date"])
def test_missing_required_field_rejected(field):
    payload = _valid_payload()
    del payload[field]
    with pytest.raises(ValidationError):
        AssessRequest(**payload)


def test_non_positive_recommended_amount_rejected():
    with pytest.raises(ValidationError):
        AssessRequest(**_valid_payload(recommended_amount=0))
    with pytest.raises(ValidationError):
        AssessRequest(**_valid_payload(recommended_amount=-100))


def test_non_positive_sanctioned_amount_rejected():
    with pytest.raises(ValidationError):
        AssessRequest(**_valid_payload(sanctioned_amount=-5))


def test_status_response_schema_shape():
    status = StatusResponse(underspend_model_available=True, gemini_api_key_configured=False, gemini_model="gemini-3.6-flash")
    assert status.module == "module1b"
