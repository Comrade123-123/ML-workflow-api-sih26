from mplads.module1b_gemini_risk.comparable_projects import find_comparable_projects
from mplads.module1b_gemini_risk.schemas import ComparableProject


def test_finds_comparables_for_a_known_work_type_and_state():
    comparables, criteria = find_comparable_projects(
        work_type="Construction of buildings for community cultural activities",
        work_category="Normal/Others",
        state="Karnataka",
        recommended_amount=497185.0,
    )
    assert len(comparables) > 0
    assert len(comparables) <= 5
    assert all(isinstance(c, ComparableProject) for c in comparables)
    assert "Construction of buildings for community cultural activities" in criteria


def test_comparables_ranked_by_closeness_of_recommended_amount():
    target_amount = 500000.0
    comparables, _ = find_comparable_projects(
        work_type="Construction of buildings for community cultural activities",
        work_category="Normal/Others",
        state="Karnataka",
        recommended_amount=target_amount,
    )
    distances = [abs(c.recommended_amount - target_amount) for c in comparables]
    assert distances == sorted(distances)


def test_falls_back_to_work_category_when_work_type_has_too_few_matches():
    comparables, criteria = find_comparable_projects(
        work_type="A Work Type That Definitely Does Not Exist In The Dataset",
        work_category="Normal/Others",
        state="Karnataka",
        recommended_amount=500000.0,
    )
    assert "Work Category" in criteria
    assert len(comparables) > 0


def test_no_matches_returns_empty_list_not_an_error():
    comparables, criteria = find_comparable_projects(
        work_type="Nonexistent Type",
        work_category="Nonexistent Category",
        state="Nonexistent State",
        recommended_amount=100.0,
    )
    assert comparables == []
    assert "No completed works found" in criteria


def test_comparable_project_only_contains_real_dataset_fields():
    """No field is invented -- every value traces to an actual dataset column."""
    comparables, _ = find_comparable_projects(
        work_type="Construction of buildings for community cultural activities",
        work_category="Normal/Others",
        state="Karnataka",
        recommended_amount=497185.0,
    )
    first = comparables[0]
    assert first.work_id.startswith("WS/")
    assert first.recommended_amount > 0
