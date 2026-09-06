# Module 1B — AI-Powered Project Cost & Risk Assessment

**Module 1B does not claim to predict final project cost and does not mathematically predict cost overrun.** It combines the existing, independently-validated quantitative underspend-risk signal with historical project evidence and Gemini-based explainable risk assessment. This positioning is deliberate: MPLADS expenditure is structurally capped at the recommended/sanctioned amount, so cost-overrun prediction is not a valid target for this data (see `OPTION2_MPLADS_TARGET_FEASIBILITY_REPORT.md` and `scripts/esakshi_costoverrun_research/ESAKSHI_COST_OVERRUN_FEASIBILITY_REPORT.md`).

**Status:** implemented, isolated, fully tested with mocked Gemini calls. Not committed. Requires a `GEMINI_API_KEY` to serve real assessments — without one, the endpoint honestly returns 503.

## Purpose

Give a reviewer (MP staff, District Authority, Ministry) a single, explainable view of a project that combines:
1. A real, validated ML signal (underspend risk) — not invented for this module.
2. Real historical comparable projects — not invented, not fetched from the internet.
3. Gemini's structured, evidence-grounded interpretation of the above — not an independent prediction.

## Architecture

```
MPLADS Project (AssessRequest)
        |
        v
Existing Underspend Risk Model (src/mplads/underspend_risk -- REUSED, never retrained/duplicated)
        |
        v
Historical Comparable Projects (data/mplads_option2/datasets/work_level_completed_dataset.csv)
        |
        v
Evidence Package (observed data + model signal + historical comparisons, explicitly separated)
        |
        v
Gemini (google-genai SDK) -- explainable reasoning over the evidence ONLY
        |
        v
AI Project Risk Assessment (validated Pydantic schema) -> AssessResponse
```

Each arrow is a real function call in `src/mplads/module1b_gemini_risk/assessment.py::assess_project`, not a conceptual diagram — see that file for the exact orchestration.

## API Endpoint

### `POST /api/v1/module1b/assess`

**Request** (`AssessRequest`, `src/mplads/module1b_gemini_risk/schemas.py`):

| Field | Type | Required | Notes |
|---|---|---|---|
| `project_id` | string | Yes | Work ID |
| `work_category` | string | Yes | |
| `work_type` | string | Yes | Used for comparable-project matching and forwarded to the underspend model |
| `work_description` | string | No | Context only |
| `state` | string | Yes | Used for comparable-project matching and forwarded to the underspend model |
| `constituency` | string | No | Context only |
| `recommended_amount` | float, >0 | Yes | Forwarded to the underspend model |
| `sanctioned_amount` | float, >0 | No | Context only — see note below |
| `recommendation_date` | ISO date string | Yes | |
| `sanction_date` | ISO date string | Yes | The prediction point |

**Note on `sanctioned_amount`:** this session's own fresh research (`scripts/esakshi_costoverrun_research/`) directly verified against the live official MPLADS eSAKSHI portal that Sanction Amount equals Recommended Amount in every case checked (15/15 exact matches). This field is accepted for completeness/evidence transparency but is not used as a separate model feature.

**Response** (`AssessResponse`):

```json
{
  "project_id": "WS/MP620/2024-2025/133166",
  "model": {
    "module": "module1b",
    "name": "AI-Powered Project Cost & Risk Assessment",
    "provider": "Google Gemini",
    "gemini_model": "gemini-3.6-flash"
  },
  "quantitative_signal": {
    "risk_score": 0.5319,
    "risk_level": "ELEVATED",
    "model_version": "2026-09-05T23:26:41.972859+00:00",
    "target_definition": "has_underspend = 1 if Amount Disbursed (INR) < 0.999 * Recommended Amount (INR) ..."
  },
  "historical_evidence": {
    "match_criteria": "Same Work Type ('...') and State ('Karnataka'), ranked by closeness of Recommended Amount (target: 497,185).",
    "comparable_projects": [
      {
        "work_id": "WS/...",
        "work_category": "Normal/Others",
        "work_type": "...",
        "work_description": "...",
        "state": "Karnataka",
        "recommended_amount": 500000.0,
        "sanction_date": "2024-07-09",
        "completion_date": "2024-09-12",
        "amount_disbursed": 500000.0,
        "underspend_pct": 0.0
      }
    ]
  },
  "ai_assessment": {
    "overall_risk_level": "MEDIUM",
    "executive_summary": "...",
    "key_risk_factors": ["..."],
    "underspend_signal_interpretation": "...",
    "historical_comparison": ["..."],
    "data_quality_observations": ["..."],
    "recommended_review_actions": ["..."],
    "confidence": "MEDIUM",
    "disclaimer": "..."
  },
  "disclaimer": "This is an AI-Powered Project Cost & Risk Assessment. It does NOT predict a project's final cost ..."
}
```

**Errors:**

| Status | `detail.error` | Cause |
|---|---|---|
| 422 | (standard Pydantic error) | Invalid request body |
| 503 | `underspend_model_unavailable` | The existing underspend model artifact is missing |
| 503 | `gemini_unavailable` | Missing `GEMINI_API_KEY`, network/timeout, or a non-2xx Gemini API error |
| 503 | `gemini_response_invalid` | Gemini responded, but not with valid JSON matching the expected schema |

### `GET /api/v1/module1b/status`

Reports `underspend_model_available`, `gemini_api_key_configured` (boolean only — never the key value), and `gemini_model`.

## Gemini's Role — and What It Is Forbidden From Doing

Gemini receives the full evidence package and a system instruction that explicitly states:

> "Assess project cost and execution risk using ONLY the supplied evidence. Do not invent facts, historical projects, costs, or government rules. Treat the existing ML score strictly as an underspend-risk signal — it is NOT a cost-overrun probability, and you must not convert it into one or claim any probability of cost overrun. Do not predict or state a final project cost. Do not claim fraud, corruption, or misconduct based on this data."

Gemini's output is validated against the `AIAssessment` Pydantic schema before it is ever returned — if it doesn't conform, the endpoint returns 503 rather than passing through anything unvalidated.

## The Existing Underspend Model's Role

`src/mplads/underspend_risk/inference.py::predict_underspend_risk` is called directly and unmodified. Module 1B does not retrain it, does not duplicate its logic, and does not reinterpret its output as anything other than what it is: a probability of historical underspend, carried through to Gemini with an explicit note that it is not a cost-overrun probability.

## Comparable-Project Methodology

`src/mplads/module1b_gemini_risk/comparable_projects.py` reuses `data/mplads_option2/datasets/work_level_completed_dataset.csv` — the exact dataset the underspend model itself is trained on, not a new or downloaded source. Matching: same Work Type + State (falling back to Work Type alone, then Work Category, if too few matches exist), ranked by closeness of Recommended Amount, top 5 returned. Every field in a `ComparableProject` is a real dataset value; nothing is invented, and a field absent from the dataset is `null`, never guessed.

## Evidence Flow

The evidence package handed to Gemini explicitly separates:
- `project.observed_data` — exactly what the caller submitted.
- `underspend_model` — the existing model's output, with an explicit `interpretation_note` warning against treating it as a cost-overrun probability.
- `historical_context` — the comparable projects and how they were selected.

Gemini's own output (`ai_assessment`) is the only AI-generated part of the final response.

## Environment Variables

| Variable | Required | Default | Notes |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes (for real assessments) | — | Never hardcoded, never logged, never returned in any response — verified by dedicated tests |
| `GEMINI_MODEL` | No | `gemini-3.6-flash` | Current Gemini model per official Google documentation |

See `.env.example`. **Never commit a real `.env` file.**

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then edit .env and set your own GEMINI_API_KEY
```

## Running the API

```bash
uvicorn mplads.api.main:app --reload
```

## Demo Example

```bash
curl -X POST http://127.0.0.1:8000/api/v1/module1b/assess \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "WS/MP620/2024-2025/133166",
    "work_category": "Normal/Others",
    "work_type": "Construction of buildings for community cultural activities",
    "state": "Karnataka",
    "recommended_amount": 497185.0,
    "recommendation_date": "2024-07-08",
    "sanction_date": "2024-07-09"
  }'
```

## Testing

```bash
pytest tests/module1b_gemini_risk/ -v   # Module 1B only
pytest -q                                # complete suite
```

**No test in this repository makes a real Gemini API call.** Every test either injects a fake `call_fn` into `get_ai_assessment`/`assess_project`, or monkeypatches `gemini_client._call_gemini_raw` directly — the one function that touches the network.

## Limitations

- **Does not predict final project cost.** Not attempted, not claimed.
- **Does not calculate a cost-overrun probability.** Cost overrun does not occur in MPLADS data by mechanism design — see the linked research reports.
- **The underspend signal itself has known limitations** (documented in `docs/underspend_risk.md`): its predictive power comes largely from State/Work-Type group historical rates, not individual project characteristics, and precision at the deployed threshold is ~15-17%.
- **Gemini's assessment is reasoning over supplied evidence, not an independent verification.** It cannot know anything about the project beyond what's in the evidence package, and is explicitly instructed not to invent facts.
- **Comparable-project matching is a simple, deterministic rule** (Work Type/Category + State + amount proximity), not a learned similarity model — it is intentionally transparent rather than sophisticated.
- **Not fraud/corruption detection.** Explicitly out of scope, and Gemini is explicitly instructed against making such claims.

## Security

- `GEMINI_API_KEY` is read only from the environment (`os.environ.get`) — never a request parameter, never hardcoded, never logged.
- Verified by tests: the actual key value is asserted to never appear in any successful or failing API response body, nor in any raised exception's message.
- The status endpoint reports only a boolean (`gemini_api_key_configured`), never the key itself.
- No new secrets, credentials, or `.env` file were added to version control by this implementation.
