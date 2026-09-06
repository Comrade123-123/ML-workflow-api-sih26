# MPLADS Expenditure Utilization Risk — Backend Integration Handoff

**Status: ML implementation complete, isolated, tested. NOT integrated with the main Empowered Indian backend or frontend.** This document is the single source of truth for the backend developer performing that integration. Nothing in this document changes any ML production code, model, artifact, or API behavior — it is documentation only.

---

## 1. ML API Contract

Base URL is whatever host/port the ML service is deployed at (see §6). All paths below are relative to that base URL.

### `POST /api/v1/ml/underspend-risk`

- **Method:** POST
- **URL:** `/api/v1/ml/underspend-risk`
- **Content-Type:** `application/json`

**Request JSON schema:**

| Field | Type | Required | Validation |
|---|---|---|---|
| `work_id` | string | Yes | non-empty |
| `recommended_amount` | number (float) | Yes | must be `> 0` |
| `work_category` | string | Yes | non-empty |
| `work_type` | string | Yes | non-empty |
| `state` | string | Yes | non-empty |
| `recommended_date` | string | Yes | ISO 8601 date, e.g. `"2024-07-08"` |
| `sanction_date` | string | Yes | ISO 8601 date, e.g. `"2024-07-20"` |

**There are no optional fields.** All seven fields are required — this is a deliberately minimal, leakage-free contract (see §2).

**Response JSON schema (HTTP 200):**

| Field | Type | Notes |
|---|---|---|
| `work_id` | string | echoed from the request |
| `risk_score` | number (float, 0.0-1.0) | predicted probability of `has_underspend` |
| `risk_level` | string enum: `"TYPICAL"` \| `"ELEVATED"` | derived from `risk_score` at the 0.5 threshold |
| `model_version` | string | ISO 8601 timestamp identifying the trained artifact |
| `target_definition` | string | the exact, full target definition (see §3) — always include this verbatim wherever the score is surfaced |
| `disclaimer` | string | mandatory interpretive disclaimer (see §4) — always include this verbatim wherever the score is surfaced |

**Example request:**
```json
{
  "work_id": "WS/MP620/2024-2025/133166",
  "recommended_amount": 497185.0,
  "work_category": "Normal/Others",
  "work_type": "Construction of buildings for community cultural activities",
  "state": "Karnataka",
  "recommended_date": "2024-07-08",
  "sanction_date": "2024-07-09"
}
```

**Example successful response (HTTP 200)** — this is a real, actually-observed response from the persisted artifact, not a fabricated example:
```json
{
  "work_id": "WS/MP620/2024-2025/133166",
  "risk_score": 0.5319,
  "risk_level": "ELEVATED",
  "model_version": "2026-09-05T23:26:41.972859+00:00",
  "target_definition": "has_underspend = 1 if Amount Disbursed (INR) < 0.999 * Recommended Amount (INR), else 0. Defined only for completed works. Research base rate: 5.71% (854 / 14,965 completed works).",
  "disclaimer": "This is a limited statistical signal derived from historical MPLADS expenditure data. It is NOT cost-overrun detection, NOT fraud detection, NOT corruption or misconduct detection, and NOT a definitive assessment of project risk. Feature-ablation testing (see OPTION2_MPLADS_TARGET_FEASIBILITY_REPORT.md section 12) found the underlying signal comes almost entirely from historical State and Work Type group rates rather than this specific project's own characteristics -- it does not reliably differentiate risk WITHIN a State/Work-Type category. Precision at the decision threshold below was approximately 16-17% in backtesting: most works flagged will NOT end up underspending. This score does not predict or prove future behavior."
}
```

**HTTP 422 response** (invalid input — missing field, wrong type, or `recommended_amount <= 0`) — standard FastAPI/Pydantic validation error body, e.g.:
```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "state"],
      "msg": "Field required",
      "input": { "...": "..." }
    }
  ]
}
```

**HTTP 503 response** (model artifact unavailable) — this is a real, actually-observed response (produced by physically removing the artifact and calling the live endpoint), not a fabricated example:
```json
{
  "detail": {
    "error": "model_unavailable",
    "reason": "No trained underspend-risk model is currently available. Run scripts/train_underspend_risk.py to train and persist one -- see GET /api/v1/ml/underspend-risk/status."
  }
}
```

### `GET /api/v1/ml/underspend-risk/status`

- **Method:** GET
- **URL:** `/api/v1/ml/underspend-risk/status`
- No request body / query parameters.

**Response JSON schema (HTTP 200, always 200 — this endpoint itself never fails just because the model is unavailable):**

| Field | Type | Notes |
|---|---|---|
| `model_available` | boolean | poll this before assuming predictions will succeed |
| `model_version` | string \| null | null when `model_available` is false |
| `trained_at` | string \| null | null when `model_available` is false |
| `target_definition` | string | always present, regardless of availability |
| `disclaimer` | string | always present, regardless of availability |

**Example response, model available:**
```json
{
  "model_available": true,
  "model_version": "2026-09-05T23:26:41.972859+00:00",
  "trained_at": "2026-09-05T23:26:41.972859+00:00",
  "target_definition": "has_underspend = 1 if Amount Disbursed (INR) < 0.999 * Recommended Amount (INR), else 0. Defined only for completed works. Research base rate: 5.71% (854 / 14,965 completed works).",
  "disclaimer": "This is a limited statistical signal derived from historical MPLADS expenditure data. It is NOT cost-overrun detection, NOT fraud detection, NOT corruption or misconduct detection, and NOT a definitive assessment of project risk. ..."
}
```

**Example response, model unavailable** (real, actually-observed — produced during this session's own controlled test):
```json
{
  "model_available": false,
  "model_version": null,
  "trained_at": null,
  "target_definition": "has_underspend = 1 if Amount Disbursed (INR) < 0.999 * Recommended Amount (INR), else 0. Defined only for completed works. Research base rate: 5.71% (854 / 14,965 completed works).",
  "disclaimer": "This is a limited statistical signal derived from historical MPLADS expenditure data. ..."
}
```

---

## 2. Field Mapping — Backend MPLADS Fields → ML API Request Fields

**Prediction point: SANCTION TIME.** A work must already be sanctioned (have a real Work ID, not a not-yet-sanctioned placeholder) before calling this API. All fields below must reflect information known at or before the work's Sanction Date.

| ML field | Source dataset/backend field | Type | Required | Known at | Safe at prediction point (Sanction time)? |
|---|---|---|---|---|---|
| `work_id` | `Works_Recommended_Cleaned.csv` / `Works_Completed_Cleaned.csv` → `Work ID` | string | Yes | Assigned at sanction | Yes — this is the prediction point itself |
| `recommended_amount` | `Works_Recommended_Cleaned.csv` → `Recommended Amount (INR)` | float | Yes | Recommendation (before sanction) | Yes |
| `work_category` | `Works_Recommended_Cleaned.csv` → `Work Category` | string | Yes | Recommendation | Yes |
| `work_type` | `Works_Recommended_Cleaned.csv` → `Work Type` | string | Yes | Recommendation | Yes |
| `state` | `Works_Recommended_Cleaned.csv` → `State` | string | Yes | Recommendation | Yes |
| `recommended_date` | `Works_Recommended_Cleaned.csv` → `Recommended Date` | ISO date string | Yes | Recommendation | Yes |
| `sanction_date` | `Works_Recommended_Cleaned.csv` → `Sanction Date` | ISO date string | Yes | Sanction (the prediction point itself) | Yes |

**Fields that must NEVER be sent to this API, because they are not knowable at Sanction time and would be leakage if they were somehow available:** `Completion Date`, `Amount Disbursed (INR)`, `Has Image`, any expenditure-ledger field (`Fund Disbursed Amount (INR)`, `Payment Status`, `Expenditure Date`, `Vendor Name`). The API's request schema has no fields for any of these — it is structurally impossible to submit them, not merely a convention to avoid.

**On the "NA-" placeholder Work ID case:** roughly 8% of `Works_Recommended_Cleaned.csv` rows represent works recommended but not yet sanctioned, and carry a placeholder Work ID of the form `"NA-<Work Type>"` instead of a real one. **Do not call this API for such rows** — wait until the work has a real Work ID and a real Sanction Date.

---

## 3. Model Details

| Property | Value |
|---|---|
| Product name | **MPLADS Expenditure Utilization Risk** |
| Target | `has_underspend` |
| Target definition | `has_underspend = 1 if Amount Disbursed (INR) < 0.999 * Recommended Amount (INR), else 0`, defined only for completed works |
| Model type | `RandomForestClassifier` (scikit-learn) |
| Exact hyperparameters | `n_estimators=300, max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=42` |
| Artifact files | `models/underspend_risk/underspend_risk_model.joblib` (the fitted sklearn Pipeline), `models/underspend_risk/historical_rate_lookup.json` (peer-rate lookup table), `models/underspend_risk/metadata.json` (version/trained_at) |
| Model version (current) | `2026-09-05T23:26:41.972859+00:00` |
| Training dataset | `data/mplads_option2/datasets/work_level_completed_dataset.csv` — 14,965 completed works, deterministic Work-ID join of `Works_Recommended_Cleaned.csv` and `Works_Completed_Cleaned.csv` |
| Decision threshold | **0.5** — the one and only threshold ever evaluated during research (used to compute the precision/recall/F1 reported in §5); no additional tiers were validated |
| Risk-level mapping | `risk_score >= 0.5` → `"ELEVATED"`; else `"TYPICAL"` |

**Feature list and exact ordering** (as fed to the preprocessing pipeline — order matters for reproducibility, though the API abstracts this away from the caller):
1. `Recommended Amount (INR)` (numeric)
2. `recommended_amount_log` = `log1p(Recommended Amount (INR))` (numeric, derived)
3. `recommendation_to_sanction_days` = `Sanction Date - Recommended Date` in days (numeric, derived)
4. `state_historical_underspend_rate` (numeric, derived — see below)
5. `work_type_historical_underspend_rate` (numeric, derived — see below)
6. `Work Category` (categorical)
7. `State` (categorical)
8. `Work Type` (categorical)

**Preprocessing pipeline:** numeric features → median imputation → standard scaling; categorical features → constant `"UNKNOWN"` imputation → one-hot encoding (`handle_unknown="ignore"`, so an unseen category at inference time does not error, it is encoded as all-zeros).

**Historical rate lookup:** a fixed table (`historical_rate_lookup.json`) mapping each `State` and each `Work Type` seen in training to its historical `has_underspend` rate, computed once from the full training set. Contains 32 State entries and 95 Work Type entries. An unseen `State` or `Work Type` at inference time falls back to the global training-population prior rate (**0.05707**, i.e. the overall base rate). This lookup is applied automatically inside the ML service — the backend does not need to compute or supply these rates.

---

## 4. Interpretation — Read This Before Displaying Anything to a User

`risk_score` is a **statistical model output derived from historical MPLADS expenditure patterns**, nothing more. Specifically:

- It is **NOT** cost-overrun detection (MPLADS expenditure is structurally capped at the recommended amount — cost overrun cannot occur in this data, confirmed empirically).
- It is **NOT** fraud detection.
- It is **NOT** corruption detection.
- It is **NOT** misconduct detection.
- It is **NOT** proof of wrongdoing of any kind.
- It is **NOT** a certainty or a forecast guarantee — it is a backward-looking statistical pattern applied prospectively, with all the uncertainty that implies (see §5 for exactly how much, or little, certainty that is).
- A large share of its predictive signal comes from **which State and Work Type category** a work belongs to, not from the specific project's own individual characteristics (confirmed via feature-ablation testing — removing State/Work Type drops performance below random). Do not present this as "this specific project has been individually assessed as risky."

**Every place this score is surfaced to a human — API response, UI, report, alert — must carry the `target_definition` and `disclaimer` fields verbatim.** Do not paraphrase them into something more alarming or more certain-sounding.

---

## 5. Model Performance (latest independently reproduced chronological validation — not exaggerated)

Chronological (not random) 70/15/15 split by Sanction Date, zero Work-ID overlap between splits, reproduced through the actual production code path:

| Split | n | Positive rate |
|---|---|---|
| Train | 10,475 | 4.82% |
| Test | 2,245 | 9.53% |

| Metric | Value | Baseline (test-set base rate) |
|---|---|---|
| ROC-AUC | 0.6317 | 0.5 |
| PR-AUC | 0.1764 | 0.0953 |
| Precision (at threshold 0.5) | 0.1489 | — |
| Recall (at threshold 0.5) | 0.5093 | — |
| F1 (at threshold 0.5) | 0.2304 | — |
| Confusion matrix `[[TN,FP],[FN,TP]]` | `[[1408,623],[105,109]]` | — |

**In plain terms:** the model beats random guessing by a real, reproducible margin (ROC-AUC ~0.63, PR-AUC ~1.85x base rate) — but at the one validated operating threshold, **only about 15% of works flagged `"ELEVATED"` will actually turn out to have underspent.** This is a prioritization signal for further review, not a reliable individual verdict.

**Known limitations:**
1. The signal is dominated by State/Work-Type group base rates, not individual project features (feature-ablation-confirmed).
2. Modest precision (~15%) at the only validated threshold.
3. Training data covers only the current Lok Sabha term (~2 years) — no multi-term historical validation.
4. Only one threshold (0.5) has ever been validated; no LOW/MEDIUM/HIGH tiering exists or should be invented downstream.
5. The persisted artifact is trained on 100% of the available validated data (standard practice for a deployed model) — the numbers above come from a separate, temporary, chronologically-held-out evaluation model, not the persisted artifact itself, since the persisted one has no held-out data left to score honestly.

Full research record: `OPTION2_MPLADS_TARGET_FEASIBILITY_REPORT.md` (target discovery and validation) and `OPTION2A_MPLADS_EXPENDITURE_ANOMALY_REPORT.md` (why a stronger alternative was not found).

---

## 6. Deployment Requirements

| Requirement | Value |
|---|---|
| Python version | 3.14 (developed/tested on 3.14.7) |
| Dependencies | See `requirements.txt`: `fastapi>=0.115`, `uvicorn[standard]>=0.30`, `pydantic>=2.7`, `pandas>=2.2`, `scikit-learn>=1.5`, `joblib>=1.4` (the ML-serving-relevant subset; the file also includes `pytest`, `httpx`, `xgboost`, `pypdf` used by other parts of this repo, not required just to serve this endpoint) |
| Model artifact requirements | `models/underspend_risk/underspend_risk_model.joblib`, `models/underspend_risk/historical_rate_lookup.json`, and `models/underspend_risk/metadata.json` must all be present and readable at the paths configured in `src/mplads/config.py` (`UNDERSPEND_RISK_MODEL_FILE`, `UNDERSPEND_RISK_RATE_LOOKUP_FILE`, `UNDERSPEND_RISK_METADATA_FILE`) — these paths are relative to the repo root (`PROJECT_ROOT`), not currently configurable via environment variable |
| Environment variables | **None.** This service reads no environment variables anywhere in its code path (verified by inspection — no `os.environ`/`os.getenv` calls exist in `src/mplads/`). All paths are derived from the repo's own directory structure. |
| Startup command | `uvicorn mplads.api.main:app --host 0.0.0.0 --port 8000` (run from the repo root with `src/` on `PYTHONPATH`, or `pip install -e .` if a package install step is added later — this repo does not currently define one) |
| Health check | `GET /health` → `{"status": "ok"}` (confirms the process is up; does **not** confirm the model is loaded) |
| Model-availability check | `GET /api/v1/ml/underspend-risk/status` → `model_available: true/false` (confirms whether predictions will succeed) |
| Expected failure behavior | If the artifact files are missing/corrupted, the service **starts successfully** (model loading is lazy, per-request) and `/health` still returns OK, but `/api/v1/ml/underspend-risk/status` reports `model_available: false` and every `POST /api/v1/ml/underspend-risk` call returns 503. The service never crashes or fabricates a prediction in this case. |

---

## 7. Error Handling

| Code | Meaning | Backend must... |
|---|---|---|
| **422** | Invalid input — a required field is missing, has the wrong type, or fails a validation rule (e.g. `recommended_amount <= 0`) | Treat as a backend bug (the request was built incorrectly) — fix the request construction, do not retry as-is |
| **503** | Model artifact is genuinely unavailable | Treat this as "ML service temporarily has no answer," not as a risk verdict of any kind |
| Network timeout / connection failure | The ML service is unreachable or slow | Treat identically to 503 for risk-interpretation purposes |

**Critical rule: the backend must NEVER convert a 503, a timeout, or any other ML-unavailability condition into a fabricated "low risk" / "typical" / default score.** Model availability and risk result are two entirely separate concerns:
- If the ML call fails for any reason, the correct backend behavior is to surface "risk assessment unavailable" (or simply omit the field) — never silently substitute a value that looks like a real prediction.
- Do not retry-with-fallback-to-a-guess. Retry the actual call if appropriate, or surface unavailability; do not invent a number.

Recommended timeout: a single prediction call should complete in well under 1 second once the model is loaded (in-memory Random Forest inference on 8 features); a generous client-side timeout of 2-5 seconds is more than sufficient, with no retry-with-fabrication fallback on expiry.

---

## 8. Integration Example (documentation only — not applied to any backend repository)

```typescript
// Documentation-only example. Not executed, not applied to any backend
// repository. Illustrates the request/response contract and, critically,
// the required "unavailable != low risk" error handling.

interface UnderspendRiskRequest {
  work_id: string;
  recommended_amount: number;
  work_category: string;
  work_type: string;
  state: string;
  recommended_date: string; // ISO date, e.g. "2024-07-08"
  sanction_date: string;    // ISO date, e.g. "2024-07-20"
}

interface UnderspendRiskResponse {
  work_id: string;
  risk_score: number;
  risk_level: "TYPICAL" | "ELEVATED";
  model_version: string;
  target_definition: string;
  disclaimer: string;
}

type UnderspendRiskResult =
  | { status: "ok"; data: UnderspendRiskResponse }
  | { status: "unavailable"; reason: string }
  | { status: "invalid_request"; details: unknown };

async function getUnderspendRisk(
  req: UnderspendRiskRequest,
  mlBaseUrl: string
): Promise<UnderspendRiskResult> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 5000); // no fabricated fallback on timeout

  try {
    const res = await fetch(`${mlBaseUrl}/api/v1/ml/underspend-risk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(req),
      signal: controller.signal,
    });

    if (res.status === 200) {
      const data = (await res.json()) as UnderspendRiskResponse;
      return { status: "ok", data };
    }
    if (res.status === 503) {
      const body = await res.json();
      return { status: "unavailable", reason: body.detail?.reason ?? "model unavailable" };
    }
    if (res.status === 422) {
      const body = await res.json();
      return { status: "invalid_request", details: body.detail };
    }
    // Any other unexpected status: treat as unavailable, never as a score.
    return { status: "unavailable", reason: `unexpected status ${res.status}` };
  } catch (err) {
    // Network error or timeout: unavailable, NOT a fabricated low-risk result.
    return { status: "unavailable", reason: String(err) };
  } finally {
    clearTimeout(timeout);
  }
}

// Caller-side usage: the UI/report layer must handle "unavailable" as its
// own distinct state -- e.g. "Risk assessment temporarily unavailable" --
// and must always render `target_definition` + `disclaimer` verbatim
// alongside any "ok" result, never a bare number.
```

---

## 9. Security / Network Notes

- **Internal-only service.** This ML API has no authentication of its own and should not be exposed directly to the public internet — deploy it as an internal service reachable only from the main backend (e.g. private network / VPC / internal DNS), with the main backend acting as the sole authenticated gateway to end users.
- **Timeouts.** See §7 — a short client-side timeout (2-5s) with no fabricated fallback is required on the backend's calling code.
- **Request validation.** The ML API already validates its own input strictly (422 on anything malformed) — the backend does not need to duplicate that validation, but should still avoid sending obviously-wrong data (e.g. don't send a not-yet-sanctioned work, see §2).
- **No secrets in source control.** This service currently requires no API keys, tokens, or credentials — nothing to leak. If network-level auth (e.g. a shared internal token) is added later for service-to-service calls, it must go through the deploying environment's secret management, never committed to this repository.
- **Logging.** Log request outcomes (status code, `work_id`, `model_version`, latency) for observability. Do **not** log the full disclaimer/target_definition text repeatedly at high volume (adds noise, no security concern, just verbosity) — logging the `risk_score` and `risk_level` per request is fine and useful for monitoring drift over time. Avoid logging anything beyond what's already in the request itself (no additional PII is present in the request schema — it is Work ID/State/Work Type/amounts, all already public MPLADS data).

---

## 10. Handoff Checklist

```
[ ] ML service running (uvicorn process healthy, GET /health returns {"status": "ok"})
[ ] GET /api/v1/ml/underspend-risk/status returns model_available: true
[ ] Backend sends all 7 required fields, correctly mapped per section 2 (Sanction-time data only)
[ ] Backend never calls this endpoint for a not-yet-sanctioned work (no real Work ID / Sanction Date yet)
[ ] Prediction response parsed correctly (risk_score, risk_level, model_version, target_definition, disclaimer)
[ ] HTTP 503 handled explicitly as "unavailable", not defaulted to a score
[ ] HTTP 422 handled as a request-construction bug, logged and fixed, not retried blindly
[ ] Timeout/network failure handled as "unavailable", not defaulted to a score
[ ] No fabricated fallback score anywhere in the backend's integration code
[ ] target_definition and disclaimer preserved verbatim everywhere risk_score is surfaced to a human
[ ] model_version captured and stored/logged alongside any prediction that is persisted or displayed
[ ] Backend integration tests written and passing (including an explicit "ML service unavailable" test case)
```

---

## Appendix: What the Backend Developer Still Needs From the ML Side

- A decision on where this ML service will actually run (host/port/network placement) — not an ML-repository concern, needs infrastructure input.
- If usage reveals the 0.5/TYPICAL-ELEVATED framing is too coarse for the product's needs, that requires a **new, properly validated** threshold study — not an ad hoc change on the backend side.
- If backend usage patterns reveal a need for batch prediction (many works at once) rather than one-at-a-time, that is a new API surface to design and validate, not present today.
