# Option 2 — MPLADS-Specific ML Target Feasibility Report

**Scope reminder:** This uses only the official MPLADS CSVs already in `Data/`. Option 3 (MoSPI/IPMD infrastructure cost-overrun) remains abandoned and untouched, and its dataset/target/conclusions are not reused here. No changes were made to Modules 2/7/8/9, the existing Module 1 API, the frontend, the backend, or any existing dataset. All Option 2 work lives in `scripts/mplads_option2_pipeline/` and `data/mplads_option2/` — separate, versioned research directories.

---

## 1. Executive Summary

**Final decision: B — Limited MPLADS ML Target.**

Starting from the data with no target assumed, the investigation ruled out "cost overrun" immediately and empirically: across 14,965 deterministically-joined completed works, **zero** ever had a disbursed amount exceeding the recommended amount (91.8% match it exactly). It also ruled out raw "completion probability" as a prospective target: completion rate is ~0% for works sanctioned under 11 months ago and rises to 65-83% for works sanctioned 18+ months ago (correlation with sanction-age = 0.57) — the same right-censoring/survivorship-bias mechanism that sank Option 3.

The target that survived scrutiny is **`has_underspend`**: whether a completed work's disbursed amount falls below its recommended amount at all (a natural point-mass-vs-tail split found directly in the data, not an arbitrary percentage). This target:
- Is a real, observed, non-fabricated outcome (5.7% base rate).
- Has zero data leakage (verified via explicit feature/target separation and an independent re-derivation of every engineered feature).
- Shows genuine, non-trivial predictive signal, **robust across four different chronological train/test cutoffs** and **consistent across three model families** (Logistic Regression, Random Forest, XGBoost) — ROC-AUC 0.64-0.79, PR-AUC 1.6-3.5x the base rate at every cutoff tested.
- But that signal comes **almost entirely from State/Work-Type group membership**, not from individual-project characteristics — feature ablation shows amount/timing features alone perform *worse than random* (ROC-AUC 0.43) with State/Work-Type category information removed.

This is a real, useful, but narrow finding: a legitimate group-level (state/category) risk-flagging signal, not a fine-grained per-project prediction tool. It is recommended for research/limited civic use (e.g., feeding into Module 7/9's existing aggregation), not a standalone confident prediction API.

---

## 2. Complete MPLADS Data Inventory

| File | Rows | Columns | Grain |
|---|---|---|---|
| `Works_Recommended_Cleaned.csv` | 67,000 | 13 | one row per recommendation (61,650 real Work IDs + 5,440 not-yet-sanctioned placeholder rows) |
| `Works_Completed_Cleaned.csv` | 15,000 | 13 | one row per completed work (15,000 unique Work IDs, zero duplicates) |
| `Expenditure_Cleaned.csv` | 73,000 | 12 | one row per vendor payment transaction (49,104 unique Work IDs — ledger grain, many rows per work) |
| `Allocated_Limit_LokSabha_Cleaned.csv` | 543 | 5 | one row per Lok Sabha MP (allocation ceiling) |
| `Allocated_Limit_RajyaSabha_Cleaned.csv` | 231 | 6 | one row per Rajya Sabha MP |
| `Amount_Consented_Calamity_Cleaned.csv` | 20 | 7 | one row per calamity-fund consent (too small to model; contextual only) |

**Date coverage across every file: ~2024-07 to ~2026-09** — roughly two years, corresponding to the current Lok Sabha term. No historical multi-term data is present. This narrow window is the root cause of the completion-target censoring problem (§5) and constrains how far any chronological split can push into the future.

### Works_Recommended_Cleaned.csv — full column dictionary

| Column | Type | Missing | Known at |
|---|---|---|---|
| Work Category | categorical (4 values) | 0% | Recommendation |
| Work ID | string, deterministic key | 0% (8.12% are the "NA-\<type\>" not-yet-sanctioned placeholder, see §4) | Assigned at Sanction |
| Work Type | categorical (107 values) | 8.12% | Recommendation |
| Work Description | free text | 0.15% | Recommendation |
| State / IDA District / IDA Office | categorical | 0% | Recommendation |
| Hon'ble Members of Parliament / Constituency | categorical | 0% | Recommendation |
| Recommended Date | date | 0% | Recommendation |
| Recommended Amount (INR) | numeric | 0% | Recommendation |
| Sanction Date | date | 8.12% (exactly the not-yet-sanctioned rows) | Sanction |

### Works_Completed_Cleaned.csv — full column dictionary

| Column | Type | Missing | Known at |
|---|---|---|---|
| Work Category / Work Type / Work Description | as above | ≤0.37% | Recommendation |
| Work ID | deterministic key, 100% unique | 0% | Sanction |
| State / IDA District / IDA Office / MP / Constituency | categorical | 0% | Recommendation |
| Has Image | boolean | 0% | **After completion** (structurally only exists in this table) |
| Completion Date | date | 0% | **After completion (outcome)** |
| Amount Disbursed (INR) | numeric | 0.03% | **After completion (outcome)** |

### Expenditure_Cleaned.csv — full column dictionary

| Column | Type | Missing | Known at |
|---|---|---|---|
| Work Type / State / District / Office / MP / Constituency | categorical | 0% | Recommendation |
| Work ID | deterministic key (ledger grain — repeats per work) | 0% | Sanction |
| Expenditure Date | date | 0% | During implementation |
| Vendor Name | categorical (22,601 values) | 0% | During implementation |
| Payment Status | categorical (2 values: Success / In-Progress) | 0% | During implementation |
| Fund Disbursed Amount (INR) | numeric | 0% | During implementation |

**Relationship between datasets:** `Work ID` is a genuinely deterministic join key across all three work-level files once the not-yet-sanctioned placeholder rows are excluded (§4). `Expenditure` is a ledger (many rows per Work ID); `Works_Completed.Amount Disbursed` is the aggregate final figure for that work.

---

## 3. Dataset Relationships (Summary)

```
Works_Recommended (67,000 rows, 61,650 real Work IDs + 5,440 not-yet-sanctioned)
        |  Work ID (deterministic)
        v
Works_Completed (15,000 rows, 15,000 unique Work IDs)  <-- terminal outcome table
        ^
        |  Work ID (deterministic, ledger grain)
Expenditure (73,000 rows, 49,104 unique Work IDs)  <-- granular payment ledger, most rows belong to still-open works
```

Allocated_Limit_* and Amount_Consented_Calamity are MP-level/aggregate tables, not work-level, and were not joined into the modeling dataset — they could serve as future contextual features (e.g., "MP's total allocation") but are not needed for the target validated here.

---

## 4. Deterministic Join Analysis

**Critical data-quality finding, fully explained (not a bug worth silently patching around):** 5,440 of 67,000 `Works_Recommended` rows carry a placeholder Work ID of the form `"NA-<Work Type description>"` instead of a real ID (e.g. `NA-Construction of community centers and community halls`, appearing 1,321 times). Investigation confirmed these are **exactly** the same 5,440 rows that are also missing `Sanction Date` (100% overlap, both directions) — i.e., a real Work ID is only assigned once a work is sanctioned; unsanctioned recommendations share a category-level placeholder string. These rows were **excluded** from the deterministic-join dataset with this reasoning documented (not silently dropped). After exclusion, **Works_Recommended has zero genuine duplicate Work IDs** — the join key is fully clean.

| Join | Match rate | Detail |
|---|---|---|
| Works_Completed → Works_Recommended | **99.77%** | 14,965 / 15,000 matched; 35 completed works have no matching recommendation record (likely predate this data snapshot) |
| Expenditure → Works_Recommended | 87.66% | remaining 12.3% likely belong to works recommended before this data's start |
| Expenditure → Works_Completed | 25.28% | expected — most expenditure ledger rows belong to works still in progress, not yet in the Completed table |

**Amount consistency check (`Amount Disbursed (INR)` vs `Recommended Amount (INR)`, n=14,965 joined completed works):**
- 91.82% disburse **exactly** the recommended amount (to the cent).
- **0% (zero) ever disburse more than the recommended amount.**
- 8.15% disburse less, with the shortfall ratio ranging continuously from 99.9% down to 18.2% of the recommended amount — no natural threshold/gap exists in this tail (§6), it is a smooth decay.

**Expenditure-ledger reconciliation:** for the 82.76% of completed works with matching ledger rows, 98.1% reconcile exactly (`Amount Disbursed` == sum of that work's `Fund Disbursed Amount` rows); the remaining 1.9% show `Amount Disbursed` slightly higher than the ledger sum (consistent with incomplete ledger coverage in this snapshot, not a contradiction) — zero cases go the other way.

**Date-order violations found: zero** across every checked pair (Sanction before Recommended, Completion before Recommended, Completion before Sanction, Expenditure before Recommended). The data is clean on temporal ordering.

---

## 5. Candidate Target Analysis

| # | Candidate | Investigated using | Outcome |
|---|---|---|---|
| A | Completion probability | `Works_Recommended` + `Works_Completed`, completion rate by sanction-age | **Ruled out** — severe right-censoring (§ below), same failure mode as Option 3 |
| B | Time-to-completion / delay | `Works_Recommended` + `Works_Completed` durations | **Already solved by Module 2** (peer-relative percentile classification on Sanction→Completion duration) — not duplicated here |
| C | Cost overrun (final > sanctioned) | `Works_Completed.Amount Disbursed` vs `Works_Recommended.Recommended Amount` | **Ruled out — empirically impossible in this data.** 0 of 14,965 completed works ever exceed the recommended amount |
| D | High final-expenditure risk | same as C | **Ruled out**, same reason as C — there is no "high" tail above 100% of recommended, by mechanism design |
| E | Expenditure utilization / underspend risk | `Works_Completed` vs `Works_Recommended`, ratio distribution | **Selected** — the only genuinely variable financial outcome in the data |
| F | Sanction probability (will a recommendation get sanctioned) | `Works_Recommended` NA- placeholder rate by recommendation-age | Valid but **low civic value** and shows the same (milder) censoring pattern: not-yet-sanctioned rate is 14.6% for the youngest quartile of recommendations vs. 2.8% for the oldest — not pursued further |
| G | Abnormal expenditure / anomaly pattern | `Expenditure` ledger (Payment Status, vendor patterns) | Not modeled in this pass — flagged as a plausible future direction, not validated here (see §17) |

### Special investigation: sanctioned vs. disbursed amount

The data model does **not** carry a separate "sanctioned amount" distinct from "recommended amount" — there is only `Recommended Amount (INR)` (at recommendation) and `Amount Disbursed (INR)` (at completion); `Sanction Date` exists but no corresponding sanctioned-amount field. Empirically, sanction evidently ratifies the recommended amount as-is (disbursement is capped at exactly that figure in 91.8% of cases, never above it). This directly confirms, from the MPLADS side, the same structural payment-cap mechanism that was the original reason Module 1 could not be an MPLADS-native cost-overrun model. `actual − recommended` was **not** casually labeled "cost overrun" per instruction; given disbursement can only be at or below the recommended figure, the only mathematically supportable framing is **underspend / utilization shortfall**, which is what was formally defined and pursued.

---

## 6. Target Validity Ranking

| Target | Validity | Data coverage | Learnability | Leakage risk | Civic value | Recommendation |
|---|---|---|---|---|---|---|
| A. Completion probability | Poor — confounded by recommendation recency, not project risk | 61,560 sanctioned works | Untestable prospectively (age dominates) | High (age-since-sanction is a de facto label proxy) | Would be high **if** valid | **Reject** |
| C/D. Cost overrun / high-expenditure risk | Invalid — outcome does not occur | 14,965 | N/A | N/A | N/A | **Reject** |
| E. Underspend risk (`has_underspend`) | **Valid** — real, observed, well-defined | 14,965 completed works | **Confirmed** — robust signal across 4 cutoffs, 3 models | Low (verified, §9) | Moderate — group-level flag, not per-project | **Select — B (limited use)** |
| F. Sanction probability | Valid but low value | 67,000 recommendations | Likely learnable (not deeply tested) | Moderate (same censoring family) | Low — administrative-speed only | Not pursued |
| G. Expenditure anomaly | Untested | 73,000 ledger rows | Unknown | Unknown | Potentially high | Future work, not validated |

---

## 7. Recommended Target — Exact Definition

```
has_underspend = 1  if  Amount Disbursed (INR) < 0.999 * Recommended Amount (INR)
has_underspend = 0  otherwise (i.e., disbursed == recommended, the 91.8% point mass)
```

Defined only for **completed** works (Work ID present in `Works_Completed`, deterministically joined to `Works_Recommended`). The 0.999 cutoff is not a tuned percentage threshold — it separates the exact-match point mass (verified to the cent) from any departure from it at all; it is not chosen to produce better metrics, and no other percentage (5%, 10%, etc.) was tested or substituted for this reason. Base rate: **5.71%** (854 / 14,965).

---

## 8. Prediction Point and Feature Classification

**Prediction point: Sanction time** (the moment a real Work ID exists and the work becomes trackable) — chosen because "recommended but not-yet-sanctioned" works lack a stable identifier to attach later outcomes to, and Sanction is the natural point at which MPLADS commits a work to the pipeline.

| Feature | Classification | Rationale |
|---|---|---|
| Recommended Amount (INR) | **SAFE** | known at recommendation, before sanction |
| `recommended_amount_log` (derived) | **SAFE** | pure transform of the above |
| `recommendation_to_sanction_days` (derived) | **SAFE** | both endpoints known at sanction time |
| Work Category | **SAFE** | known at recommendation |
| Work Type | **SAFE** | known at recommendation |
| State / IDA District / IDA Office / Constituency | **SAFE** | known at recommendation |
| `state_historical_underspend_rate` (derived) | **SAFE** | strictly-causal expanding-window mean using only strictly-earlier-sanctioned works' outcomes (verified, §9) |
| `work_type_historical_underspend_rate` (derived) | **SAFE** | same causal construction, grouped by Work Type |
| Completion Date | **LEAKAGE** | defines when the outcome is observed |
| Amount Disbursed (INR) | **LEAKAGE** | this *is* the numerator of the target |
| `ratio_disbursed_to_recommended`, `underspend_pct` | **LEAKAGE** | direct target derivatives |
| `days_recommended_to_completion`, `days_sanction_to_completion` | **LEAKAGE** | require Completion Date |
| Has Image | **LEAKAGE** | structurally exists only in `Works_Completed` — unavailable before completion by table design |
| Vendor Name / Payment Status / Expenditure Date (Expenditure ledger) | **UNCERTAIN / not used** | these accrue *during* implementation, after sanction; excluded from this SAFE feature set entirely to keep the prediction point strictly at sanction |

Only the SAFE features were used for modeling (asserted in code — see `train_underspend_model.py`).

---

## 9. Data Quality and Distribution (target population, n=14,965)

- Unique works: 14,965 (100% — one row per work, Work ID is the dataset's row key)
- Sanction Date range: 2024-07-09 to 2025-08-31 (train+val+test combined)
- Recommended Amount: min ₹10, median ₹300,000, mean ₹537,497, max ₹47,472,048 — no negative or zero values
- Amount Disbursed: min ₹10,000, median ₹269,328, mean ₹501,044, max ₹46,470,400 — no negative or zero values
- `has_underspend` base rate: **5.71%** (854 positive / 14,111 negative) — a genuinely rare, imbalanced event
- Ratio distribution (Disbursed / Recommended): P50=P75=P90=P95=P99=1.0 (the exact-match mass dominates every upper percentile); P5=0.9986, P1=0.9093 — the entire "interesting" tail lives below the 5th percentile
- Underspend magnitude (among the 854 below-exact cases): heavily right-skewed decay — 650/854 fall in the smallest bin (0.1%-4.2% underspend), thinning out smoothly to a max of 81.8%; **no natural gap or bimodal split exists in this tail**, confirming a percentage-threshold classification (e.g. "underspend >10%") would have been an arbitrary, metric-chasing choice, correctly avoided
- Underspend rate by **State**: ranges from **0%** (Nagaland, Meghalaya) to **28.5%** (Telangana) — substantial, non-random variation (n≥30 per state)
- Underspend rate by **Work Type**: ranges from 0% (several types, e.g. "Setting up of laboratories") to ~20% (IT-systems purchases) — real but smaller variation than by State
- Correlation between underspend magnitude and Recommended Amount size: **0.002** — essentially none; project size alone tells you nothing about underspend risk
- Data contradictions checked and found: **zero** negative amounts, **zero** zero-amounts, **zero** completion-before-recommendation or completion-before-sanction or sanction-before-recommendation or expenditure-before-recommendation cases

---

## 10. Temporal Split

Chronological split by Sanction Date (day-level, tie-broken by Work ID — not just year-month, to avoid even within-month leakage in the causal features), 70/15/15:

| Split | n | Sanction date range | Positive rate |
|---|---|---|---|
| Train | 10,475 | 2024-07-09 – 2025-04-28 | 4.82% |
| Val | 2,245 | 2025-04-28 – 2025-06-20 | 6.01% |
| Test | 2,245 | 2025-06-20 – 2025-09-24 | 9.53% |

Zero Work ID overlap between any pair of splits (verified). Note the positive rate *rises* from train to test (4.8% → 9.5%) — a real but much milder drift than Option 3's collapse, and (unlike Option 3) it works in the model's favor for recall-oriented metrics rather than confounding the whole result; still noted honestly as a distributional difference across time, not hidden.

---

## 11. Baseline and Model Results (primary 70/15/15 split, test set, n=2,245)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Baseline (majority class) | 0.905 | 0.000 | 0.000 | 0.000 | 0.500 | 0.095 |
| Logistic Regression | 0.703 | 0.173 | 0.561 | 0.265 | 0.663 | 0.169 |
| Random Forest | 0.677 | 0.164 | 0.584 | 0.256 | 0.644 | 0.185 |
| XGBoost | 0.721 | 0.170 | 0.495 | 0.253 | 0.656 | 0.148 |

All three real models beat the baseline on ROC-AUC (0.5→0.64-0.66) and PR-AUC (0.095→0.15-0.19, i.e. **1.6-1.9x the base rate**) — a genuine, non-degenerate signal, unlike Option 3 where every model failed to beat baseline. Precision remains modest (~16-17%): most flagged works do **not** end up underspending, so this is useful as a *risk-flagging* signal (worth extra scrutiny), not a confident individual-project forecast.

---

## 12. Robustness Results

### (a) Alternative chronological cutoffs (Random Forest, fixed config, no tuning)

| Cutoff | Train n | Test n | Test positive rate | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| 50/25/25 | 7,482 | 3,742 | 7.64% | 0.666 | 0.167 |
| 60/20/20 | 8,979 | 2,993 | 8.59% | 0.680 | 0.188 |
| **70/15/15 (primary)** | 10,475 | 2,245 | 9.53% | 0.644 | 0.185 |
| 80/10/10 | 11,972 | 1,497 | 6.55% | **0.795** | **0.226** |

**Every single cutoff beats baseline decisively** (baseline ROC-AUC is always 0.5, PR-AUC always equals the base rate) — this is the opposite of Option 3's outcome, where every cutoff failed. Performance is not just non-negative, it is consistently in a "real but modest" band, with the smallest training set (80/10/10, oddly) doing best — plausibly because its narrower test window is more homogeneous, not because more test recency helps.

### (b) Feature ablation (Random Forest, primary split)

| Feature set | ROC-AUC | PR-AUC |
|---|---|---|
| All features | 0.644 | 0.185 |
| Drop historical-rate features | 0.629 | 0.186 |
| Drop State categorical | 0.628 | 0.151 |
| **Amount + timing only (no State/Work Type at all)** | **0.427** | **0.089** |
| Historical rates only (2 numbers) | 0.655 | 0.144 |
| State categorical only | 0.623 | 0.190 |

**This is the most important interpretability finding.** With State and Work Type completely removed, the model performs *worse than random* (ROC-AUC 0.427, PR-AUC below the base rate) — individual project characteristics (its own recommended amount, its own recommendation-to-sanction lag) carry **no** standalone predictive value. Nearly all of the model's real signal is a group base-rate effect: which State and which Work Type a work belongs to. This does not invalidate the target (the group-level effect is real, large, and stable — Telangana's 28.5% vs. Nagaland's 0% is not noise), but it reframes what the model actually does: it identifies **which categories of work carry historically higher underspend rates**, not which *individual* projects are unusually risky within a category.

### (c) Class imbalance

Base rate 5.71% (train 4.82%, test 9.53%). Addressed via `class_weight="balanced"` (Logistic Regression, Random Forest) and `scale_pos_weight` (XGBoost) — no resampling/synthetic oversampling was used (would risk fabricating labels). PR-AUC (not accuracy) is treated as the primary metric throughout, per the imbalance.

### (d) Outlier sensitivity

The underspend-magnitude tail (down to 18% of recommended amount) was inspected and found to be a smooth decay with no gap suggesting a data error; these are treated as genuine, not excluded.

### (e) Leakage re-check

Repeated explicitly: `Completion Date`, `Amount Disbursed`, all target-derived columns, and `Has Image` (completion-table-only) are asserted out of the feature set in code before training. No feature requires the outcome to compute.

### (f) Independent metric recomputation

Every test-set metric (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix) for all four models was recomputed from the raw saved `predictions_test.csv` using a separate scikit-learn call outside the training script — **all values matched exactly**. Split integrity was independently re-verified (zero Work-ID overlap between train/val/test). The two causal historical-rate features were independently re-derived using a completely different implementation (vectorized pandas `groupby().cumsum()` shift vs. the training script's manual running-total loop) — **zero mismatches across all 14,965 rows for both State and Work Type rates.**

---

## 13. Independent Verification Summary

| Check | Result |
|---|---|
| Train/Val/Test Work-ID overlap | 0 / 0 / 0 |
| Metric recomputation (4 models × 6 metrics) | Exact match, all 24 values |
| Causal feature re-derivation (2 features × 14,965 rows) | 0 mismatches |
| Leakage feature exclusion | Asserted in code, verified |

No discrepancies found anywhere.

---

## 14. Civic / Product Use Case

Given the group-level nature of the finding (§12b), the defensible use case is: **aggregate underspend-risk flagging by State and Work Type**, e.g.:
- Surface a "historically higher underspend rate" flag for new works recommended in a State/Work-Type combination with an elevated causal historical rate, as an input to Module 7's constituency-level pulse aggregation or Module 9's inspector-dispatch prioritization — not as a standalone per-project alert.
- Help prioritize which *categories* of work might warrant closer expenditure-utilization monitoring, rather than naming individual projects as "at risk" with high confidence (precision ~17% means most flagged individual works will turn out fine).

**Not supported by this target:** a confident, individually-actionable "this specific project will underspend" prediction — the ablation results (§12b) show individual project features carry no signal on their own.

---

## 15. Limitations

1. **~2-year data window** — all three core CSVs span only the current Lok Sabha term. No multi-term historical validation is possible.
2. **Signal is a group base-rate effect**, not a rich individual-project model (§12b) — the single most important caveat for how this should be described to any stakeholder.
3. **Modest precision (~16-17%)** at the tested operating threshold — a majority of flagged works will not actually underspend.
4. **Small positive-class counts** in each split (train 505, val 135, test 214 positives) — individual point metrics carry real sampling uncertainty, though the *direction* of the finding (beats baseline everywhere) is consistent across four cutoffs.
5. **State-level variation could reflect reporting/administrative practice differences rather than genuine project execution differences** — this is a plausible alternative explanation for why State dominates feature importance, and was not (and could not be, from this data alone) ruled out.
6. **Sanction-probability (Candidate F)** was found valid but was not fully modeled — flagged as available future work, not fabricated into a result here.
7. **Anomaly-pattern detection on the Expenditure ledger (Candidate G)** was not investigated in this pass — a plausible future direction, explicitly left untested rather than assumed positive.

---

## 16. Final Decision Gate

- **A (strong target):** Not selected — the group-base-rate-only nature of the signal and modest precision fall short of "data + target + temporal features + model performance justify building the model" for a standalone, confident per-project prediction feature.
- **B (limited target): SELECTED.** The target is valid, non-leaky, and shows real, robust, honestly-verified predictive signal — but its practical value is a coarser (State/Work-Type-level) risk indicator, appropriate for research/aggregate use, not a confident standalone per-project prediction API.
- **C (no defensible target):** Not selected — unlike Option 3, this target does show genuine, consistently-positive, independently-verified learnability across every cutoff tested.

---

## 17. Exact Recommendation for Next Implementation Step

1. **Do not build a new standalone Module 1-replacement API from this target yet.** Per instruction, no API changes were made (`POST /api/v1/module1/predict` and `GET /api/v1/module1/status` are untouched), no model artifact was created, and nothing was wired into the frontend.
2. If this is pursued further, the natural next step is **not** more hyperparameter tuning on the current per-project features (already shown to carry no standalone signal, §12b), but rather:
   - Formally investigating **why** State/Work-Type dominates — is it genuine execution-risk variation or a reporting-practice artifact? This requires domain input, not more modeling.
   - Considering whether the *group-level historical rate itself* (already computed here as a causal feature) is the actual deliverable — i.e., a simple, transparent "category risk table" (rate by State × Work Type) might be more honest and more useful than a black-box classifier whose real content is that same table.
   - Investigating Candidate G (expenditure-ledger anomaly detection, e.g. unusual vendor concentration or Payment-Status "In-Progress" duration patterns) as a genuinely different, not-yet-tested angle before concluding Option 2's ceiling has been reached.
3. **Preserve this research area** (`scripts/mplads_option2_pipeline/`, `data/mplads_option2/`) as-is; it is independent from Option 3's artifacts and from all locked modules.
