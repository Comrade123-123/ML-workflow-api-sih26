# MPLADS Expenditure Utilization Risk

**Product name:** MPLADS Expenditure Utilization Risk
**Status:** Implemented, isolated, **model trained and persisted** (see [Current Status](#current-status)). Backend/frontend integration remains explicitly out of scope and has not been done.
**Not one of the 9 numbered MPLADS modules.** A separate, isolated capability retained from the Option 2 MPLADS-specific ML target investigation.

## What this is NOT

This capability must never be described, in code, documentation, or API responses, as:
- cost-overrun detection
- fraud detection
- corruption detection
- misconduct detection
- a definitive assessment of project risk
- a probability that a project *will* underspend (it is a historical-pattern-based statistical score, not a forecast guarantee)

Every API response includes a `disclaimer` field stating this explicitly (see `mplads.config.UNDERSPEND_RISK_DISCLAIMER`).

## Exact Target Definition

```
has_underspend = 1  if  Amount Disbursed (INR) < 0.999 * Recommended Amount (INR)
has_underspend = 0  otherwise
```

Defined only for **completed** works. The 0.999 cutoff is not a tuned percentage threshold — it separates the verified exact-match point mass (91.8% of completed works disburse exactly the recommended amount, confirmed to the cent) from any departure from it at all. See `OPTION2_MPLADS_TARGET_FEASIBILITY_REPORT.md` section 7 for the full derivation.

**Research base rate: 5.71%** (854 positive / 14,965 completed works).

## Dataset Used

`data/mplads_option2/datasets/work_level_completed_dataset.csv` — the deterministic Work-ID join of `Data/Works_Recommended_Cleaned.csv` and `Data/Works_Completed_Cleaned.csv`, excluding the 5,440 not-yet-sanctioned placeholder rows (Work ID literally `"NA-<Work Type>"`, confirmed to be exactly the rows missing a Sanction Date). No new dataset was built for this implementation — the exact research dataset is reused by `scripts/train_underspend_risk.py`.

## Prediction Point

**Sanction time** — the moment a work receives a real Work ID and becomes trackable. A work must already be sanctioned before a prediction is meaningful (the `sanction_date` field is required in the API request).

## Features (exact, matching the validated research)

| Feature | Type | Source |
|---|---|---|
| `Recommended Amount (INR)` | numeric | direct input |
| `recommended_amount_log` | numeric | `log1p(Recommended Amount (INR))` |
| `recommendation_to_sanction_days` | numeric | `Sanction Date - Recommended Date` |
| `state_historical_underspend_rate` | numeric | lookup table, see below |
| `work_type_historical_underspend_rate` | numeric | lookup table, see below |
| `Work Category` | categorical | direct input |
| `State` | categorical | direct input |
| `Work Type` | categorical | direct input |

**Explicitly excluded (leakage):** Completion Date, Amount Disbursed, `ratio_disbursed_to_recommended`, `underspend_pct`, `days_recommended_to_completion`, `days_sanction_to_completion`, `Has Image` (this last field structurally only exists in the Works_Completed table, i.e. is unavailable before completion by table design). None of these fields exist in the API's `PredictRequest` schema at all — this is a structural guarantee, not just an implementation choice, and is directly asserted by a test (`test_predict_request_schema_has_no_leakage_fields`).

### Research-vs-production distinction for the historical-rate features

During research validation (`OPTION2_MPLADS_TARGET_FEASIBILITY_REPORT.md` section 8), these two rates were computed with a strictly-causal expanding window (each row uses only strictly-earlier-sanctioned works) — necessary for an honest, leakage-free chronological backtest. For this production implementation, a deployed model scoring genuinely new future works instead uses a **fixed lookup table computed once from the full training set** — the same standard practice Module 2's peer-relative percentile thresholds already use (fixed at train time, applied at inference time). This is not a leakage regression: the causal computation's purpose was protecting *backtest* integrity, not production model construction. See `src/mplads/underspend_risk/historical_rates.py`'s module docstring for the full reasoning. An unseen State or Work Type at inference time falls back to the global training-population prior rate.

## Model Type

`RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=42)` — the exact configuration that achieved the best PR-AUC (0.185, vs. a 0.095 base rate) in the validated research's primary evaluation, not a new choice made for this implementation.

## Validation Methodology (summary — full detail in `OPTION2_MPLADS_TARGET_FEASIBILITY_REPORT.md`)

- Chronological (not random) train/val/test split by Sanction Date, zero Work-ID overlap.
- Robust across 4 different chronological cutoffs (ROC-AUC 0.64-0.79, PR-AUC 1.6-3.5x base rate at every cutoff).
- Consistent across 3 model families (Logistic Regression, Random Forest, XGBoost).
- Independently re-verified: every reported metric recomputed by hand from raw saved predictions; the causal historical-rate features re-derived via a second, differently-implemented method with zero mismatches.
- **Known limitation, load-bearing for interpretation:** feature ablation showed the signal comes almost entirely from State/Work-Type group membership — removing them entirely drops performance *below* random (ROC-AUC 0.427). Individual project features (amount, timing) carry no standalone signal. This means the score should be read as "this State/Work-Type category has historically shown elevated underspend rates," not "this specific project's own characteristics indicate risk."

## Interpreting the Risk Score

- `risk_score`: the model's raw predicted probability of `has_underspend`, in [0, 1].
- `risk_level`: `"ELEVATED"` if `risk_score >= 0.5`, else `"TYPICAL"`. **0.5 is not an invented threshold** — it is the one and only decision threshold actually evaluated in the validated research (used to compute the reported precision ~16-17%, recall ~50-58%, F1 ~0.25-0.26). No additional tiers (e.g. a 3-way LOW/MEDIUM/HIGH split) were validated, so none are offered.
- At this threshold, **most flagged works will not end up underspending** (precision ~17%) — the score is a prioritization signal for further (human) review, not a verdict.

## Current Status

**A model artifact has been trained and persisted** (`scripts/train_underspend_risk.py`, run with explicit approval). `GET /api/v1/ml/underspend-risk/status` reports `model_available: true`, and `POST /api/v1/ml/underspend-risk` returns real predictions.

- Artifact: `models/underspend_risk/underspend_risk_model.joblib` + `historical_rate_lookup.json` + `metadata.json`
- Trained on all 14,965 rows of the validated research dataset (`data/mplads_option2/datasets/work_level_completed_dataset.csv`) — standard practice: the chronological split was used during research to validate the methodology; the deployed artifact uses all validated data rather than withholding a permanent holdout that will never be used again.
- Independent post-training validation (a temporary, never-persisted model fit only on a chronological 70/15/15 train split, scored on its held-out test split, via these exact production classes) reproduced the research finding: ROC-AUC 0.632, PR-AUC 0.176 (vs. 0.095 test-set base rate), Recall 0.509 — closely matching, though not bit-identical to, the original research script's 0.644/0.185/0.584 (a small, expected, and documented difference: production's historical-rate lookup is a single fixed value per group rather than the research backtest's within-train walk-forward computation — see `historical_rates.py`'s docstring).
- **Backend/frontend integration has not been done and remains explicitly out of scope.**

## API

### `GET /api/v1/ml/underspend-risk/status`

```json
{
  "model_available": false,
  "model_version": null,
  "trained_at": null,
  "target_definition": "has_underspend = 1 if Amount Disbursed (INR) < 0.999 * Recommended Amount (INR) ...",
  "disclaimer": "This is a limited statistical signal ..."
}
```

### `POST /api/v1/ml/underspend-risk`

Request:
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

Actual response (production artifact, real inference — recorded 2026-09-05):
```json
{
  "work_id": "WS/MP620/2024-2025/133166",
  "risk_score": 0.5319,
  "risk_level": "ELEVATED",
  "model_version": "2026-09-05T23:26:41.972859+00:00",
  "target_definition": "has_underspend = 1 if Amount Disbursed (INR) < 0.999 * Recommended Amount (INR) ...",
  "disclaimer": "This is a limited statistical signal ..."
}
```

If the model artifact were ever unavailable (e.g. removed, or in a fresh environment before training) — HTTP 503:
```json
{
  "detail": {
    "error": "model_unavailable",
    "reason": "No trained underspend-risk model is currently available. Run scripts/train_underspend_risk.py to train and persist one -- see GET /api/v1/ml/underspend-risk/status."
  }
}
```

Invalid request (e.g. missing required field, non-positive `recommended_amount`) — HTTP 422, standard FastAPI/Pydantic validation error body.

## Known Limitations

1. Signal is dominated by State/Work-Type group base rates, not individual project characteristics (see Validation Methodology above).
2. Modest precision (~16-17%) at the only validated operating threshold.
3. Research dataset covers only the current Lok Sabha term (~2 years) — no multi-term historical validation exists.
4. The persisted artifact is trained on 100% of the research dataset (standard for a deployed model); its independent chronological validation numbers (see Current Status) come from a separate, temporary model, not the persisted one itself.
5. Backend/frontend integration has not been done.
