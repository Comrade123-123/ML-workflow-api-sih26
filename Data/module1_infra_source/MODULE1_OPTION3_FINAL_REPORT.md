# Module 1 — Option 3 Final Attempt: Government Infrastructure Cost-Overrun Prediction
## Final Report and Decision

**Scope reminder:** This is NOT an MPLADS model. It uses MoSPI/IPMD/OCMS "Flash Report on Central Sector Projects (₹150 crore and above)" data, is scoped, versioned, and documented separately, and has never been mixed with MPLADS data or wired into the live API.

---

## 1. Executive Verdict

**Decision: C — Abandon Option 3.**

Across two independently-verified dataset sizes (225 and 494 completed projects), four different chronological train/test cutoffs, five feature-ablation configurations, and two model families (Random Forest, XGBoost), **no configuration achieved a positive R² on held-out, chronologically-future data.** Doubling the dataset size did not fix this — R² got *worse* for both tree models at 494 projects than at 225. Root-cause analysis identified a **structural, non-fixable-by-more-data problem**: a severe covariate/target shift between train and test splits caused by survivorship bias in what counts as a "Completed Project" in these reports. This is a property of the data-generating process (which projects finish and get reported as "Completed" vs. remain "Ongoing"), not a modeling deficiency, so further hyperparameter tuning, feature engineering, or dataset expansion within this same data source would not be expected to fix it.

The one weak positive signal (Random Forest's MAE is modestly better than a mean-baseline in all four tested cutoffs) is real but insufficient: RMSE is worse than baseline in 3 of 4 cutoffs (meaning the model is worse on the cases that matter most — large overruns), and R² is decisively negative everywhere. Per the standing instruction that a negative R² is an acceptable, honest result and must not be tuned away, this is reported as-is: **the model has no reliable predictive value for its stated purpose (cost-overrun risk flagging).**

Module 1's existing MPLADS-scoped API is correctly left undeployed and unmodified.

---

## 2. Data Sources

Official MoSPI/IPMD/OCMS Flash Report PDFs, `ipm.mospi.gov.in/Content/ArchiveReport/flash/...` (and the newer `.../Content/PDF/FlashReport_*.pdf` naming for 2025+). 33 report-months acquired, spanning May 2020 – March 2026, of which 25 were successfully parsed for completed-project records; 8 required a download retry (all 8 eventually re-downloaded successfully at full size, but subsequent re-extraction of these specific large re-downloaded PDFs was abandoned after ~25 minutes of unproductive processing per file in this environment — see §11, Limitations). The 2025+ reports use a structurally different "PAIMANA"-portal format with a non-overlapping plain-numeric project-code namespace and no OCMS bracketed ID; a parser was drafted for this format (`parse_completed_projects_v2`) but a confirmed field-misalignment bug on multi-word Title-Case project names was found and **this format was excluded from the dataset entirely** rather than risk shipping misaligned rows.

---

## 3. Dataset Growth: 225 → 494 Projects

| | v1 (225, archived) | v2 (494, expanded) |
|---|---|---|
| Source reports (report_id) | 5 | 6 |
| Raw parsed rows | 334 | 1,140 |
| Unique project IDs (raw) | 244 | 539 |
| Duplicate-ID rows (repeat sightings, kept last-observed) | 90 | 601 |
| Excluded: missing cost/date/expenditure fields | 22 | 83 |
| Excluded: parsed original cost < ₹150cr (parser-misalignment floor) | 0 | 5 |
| **Valid labeled projects** | **225** | **494** |
| Positive-overrun count | 76 | 146 |
| Positive-overrun rate | 33.78% | 29.55% |
| Overrun % — min / median / mean / max | -99.72 / -11.89 / -2.90 / 354.13 | -100.0 / -15.46 / -4.18 / 598.87 |

The larger dataset is directionally consistent with the smaller one (similar median, similar positive-overrun rate, no new data-quality red flags), which is itself evidence the parsing/labeling methodology is stable — but see §8 for why this stability did not translate into better model performance.

**Sector distribution** (both datasets dominated by the same few sectors, new dataset adds tail coverage):

| Sector | v1 (225) | v2 (494) |
|---|---|---|
| Road Transport & Highways | 100 | 254 |
| Petroleum | 52 | 83 |
| Railways | 31 | 53 |
| Power | 23 | 48 |
| Health & Family Welfare | 14 | 14 |
| Urban Development | 1 | 16 |
| Coal | 3 | 11 |
| Water Resources | 0 | 6 |
| Steel | 0 | 3 |
| Dept. of Higher Education | 1 | 3 |
| Telecommunications | 0 | 2 |
| Civil Aviation | 0 | 1 |

Dataset B (ongoing projects) remains explicitly **blocked by source structure** — the only ID-bearing ongoing-projects table (Annexure XIV) is pre-filtered to overrun-only cases (zero ID overlap with Dataset A), and the true master ongoing-projects table (Annexure XVIII) carries no project ID at all. Per standing instruction, this was **not** worked around via fuzzy name-joins; Dataset B has 0 valid labeled rows in both versions and is not used for training.

---

## 4. Target Re-Validation

Target remains named `observed_expenditure_overrun_pct` (deliberately not "final_cost_overrun_pct") — this stands unchanged. New evidence found in this final pass **reinforces** rather than changes this naming: a Sept-2025-era PAIMANA-format report was found to carry an explicit disclaimer: *"Note: Cumulative Expenditure is as per last reporting by Central Line Ministries/Departments on the IPM portal, this may not be the actual completion cost."* This is direct, primary-source confirmation that "cumulative expenditure at last reporting" (which is what `observed_expenditure_overrun_pct` measures) is not the same claim as a true, audited final/closed-out cost — exactly the distinction the naming was chosen to preserve.

---

## 5. Feature Inventory (unchanged from the prior validation pass)

| Feature | Type | Available at prediction time? |
|---|---|---|
| `original_cost_cr` | numeric | Yes — sanctioned at approval |
| `planned_commissioning_year` | numeric | Yes — planned at approval |
| `sector_historical_overrun_rate` | numeric | Yes — strictly causal expanding-window mean of *only* strictly-earlier same-sector projects (falls back to global prior, then 0.0 neutral prior) |
| `sector` | categorical | Yes |
| `project_size_bucket` | categorical | Yes — derived from `original_cost_cr` |

No post-outcome field (cumulative expenditure, actual completion date, deleted/overrun-extent annexures) is used as a feature — confirmed again via the same audit methodology as the prior validation pass, now re-applied to the 494-project set.

---

## 6. Leakage Audit (re-run on 494 projects)

- Feature/target separation: unchanged, no leakage-risk features added.
- Multi-report duplicate handling: last-observed report kept per project_id (601 duplicate rows collapsed to 494 — the higher duplicate count vs. v1's 90 is expected, proportional to more source reports and more repeat sightings per project, not a new bug).
- `sector_historical_overrun_rate` causality: **independently re-derived from scratch** against all 494 projects — **0 mismatches** against the saved training-script value (§9).
- Split integrity: **zero project-ID overlap** confirmed across train/val/test in all four tested cutoffs (§8).

No new leakage pathway was introduced by the dataset expansion.

---

## 7. Split and Primary Results (494 projects, 70/15/15 chronological)

| Split | n | Year range |
|---|---|---|
| Train | 345 | 2000–2020 |
| Val | 74 | 2020–2022 |
| Test | 75 | 2022–2030 |

| Model | Test MAE | Test RMSE | Test R² | Test MedAE |
|---|---|---|---|---|
| Baseline (train-mean) | 40.324 | 49.404 | **-1.097** | 34.257 |
| Random Forest | 36.214 | 52.845 | **-1.399** | 25.180 |
| XGBoost | 40.965 | 62.388 | **-2.344** | 26.604 |

Compare to the archived 225-project run (same methodology): Baseline R²=-1.10-ish range, RF and XGBoost were *also* negative but XGBoost was the least-bad model there. At 494 projects, **both tree models got worse relative to baseline**, not better — the opposite of what more data should do if the underlying relationship were learnable with these features.

---

## 8. Root Cause: Train/Test Distribution Shift (Survivorship Bias)

| | Train (n=345) | Val (n=74) | Test (n=75) |
|---|---|---|---|
| Mean overrun % | **+2.88** | -8.01 | **-32.86** |
| Median overrun % | -10.27 | -13.38 | -28.52 |
| Positive-overrun rate | **34.2%** | 25.7% | **12.0%** |

This is a monotonic, severe shift: the most-recently-planned test slice (planned commissioning years 2022–2030) is dominated by projects that ran **far under** their sanctioned cost, with a positive-overrun rate less than half of train's. A model trained on train's more balanced label distribution has no way to anticipate this systematic swing — it isn't a signal in the *features*, it's a shift in what the *label distribution itself* looks like at the end of the timeline.

**Mechanism:** "Completed" is not a random sample of "all projects planned in year Y" — it is projects that have *already finished* by report date. Large, complex, or troubled projects (more prone to genuine cost overrun) take longer and are systematically more likely to still be "Ongoing," not yet in the Completed table, for the most recently-planned cohorts. The Completed-projects population for recent years is therefore biased toward fast, low-complexity, likely-under-budget projects — precisely the pattern observed (test set's extreme values are dominated by Petroleum and Road Transport & Highways projects with unusually large negative overruns, i.e., large under-spends).

**Robustness confirmation (§ below) shows this is not an artifact of the specific 70/15/15 cutpoint** — it persists, to varying degrees, at every cutoff tested.

---

## 9. Independent Verification (re-run on 494-project outputs)

All checks re-run from raw saved files (`split_*.csv`, `predictions_val_test.csv`, `baseline_constant_value.json`), independently recomputing every metric with hand-written formulas — never trusting the training script's own metric functions:

- **Split integrity:** Train (345) ∩ Val (74) ∩ Test (75) = ∅. Confirmed via set intersection.
- **Same test set across all 3 models:** 75/75 rows match `split_test.csv` exactly; all three model prediction columns present and non-null.
- **Baseline correctness:** baseline constant (2.875892) exactly equals the independently-recomputed **train** mean (not test mean, which is -32.857 — confirmed different).
- **Metric recomputation (test, n=75):**
  - Baseline: MAE 40.324, RMSE 49.404, R² -1.097 (SS_res=183,058.20, SS_tot=87,296.74) — exact match to reported.
  - Random Forest: MAE 36.214, RMSE 52.845, R² -1.399 — exact match.
  - XGBoost: MAE 40.965, RMSE 62.388, R² -2.344 — exact match.
- **`sector_historical_overrun_rate` causal re-derivation:** independently recomputed for all 494 projects from the raw sorted sequence — **0 mismatches** against the training script's saved values.

No discrepancies found. The reported negative-R² result is not a bug.

---

## 10. Robustness Checks

### (a) Alternative chronological cutoffs
Same features, same fixed model configs (no tuning), varying only the train/val/test split point:

| Cutoff | Train n | Test n | Test target mean | RF R² | XGB R² | Baseline R² |
|---|---|---|---|---|---|---|
| 50/25/25 | 247 | 124 | -23.36 | -0.197 | -0.617 | -0.569 |
| 60/20/20 | 296 | 99 | -23.66 | -1.016 | -2.442 | -0.553 |
| **70/15/15 (primary)** | 345 | 75 | -32.86 | -1.399 | -2.163 | -1.097 |
| 80/10/10 | 395 | 50 | -38.31 | -1.646 | -3.053 | -1.381 |

**Every single cutoff produces negative R² for every model.** The test-set target mean becomes more extreme as the test window shrinks to more-recent years (-23 → -38), consistent with the survivorship-bias mechanism worsening the closer you get to "now." This rules out the 70/15/15 cutpoint as a fluke — the negative result is stable across the whole plausible range of chronological splits.

### (b) Feature ablation (Random Forest, primary 70/15/15 split)

| Feature set | MAE | RMSE | R² |
|---|---|---|---|
| All 5 features | 36.214 | 52.845 | -1.399 |
| Drop `sector_historical_overrun_rate` | 35.623 | 51.057 | -1.240 |
| Drop `sector` (categorical) | 33.941 | 47.261 | -0.919 |
| `original_cost_cr` + year only | 44.623 | 52.103 | -1.332 |
| `sector_historical_overrun_rate` only | 31.900 | 43.018 | -0.590 |

No ablation reaches positive R². Interestingly, dropping the `sector` categorical (12 levels, some with very few samples) *improves* R² from -1.399 to -0.919 — a sign the model is overfitting to noisy per-sector splits given how few observations most sectors have, rather than the categorical genuinely helping. The single causal feature (`sector_historical_overrun_rate` alone) is the least-bad single-feature configuration, but still solidly negative.

### (c) Outlier sensitivity
The test set's extreme values are dominated by Petroleum and Road Transport & Highways projects with unusually large negative (under-budget) outcomes — consistent with, and a contributor to, the distribution-shift finding in §8. This was not treated as a "removable outlier" problem since these are genuine, verified project records, not parsing errors — removing them would be manipulating the evaluation to improve the reported metric, which is explicitly prohibited.

### (d) Leakage re-check
Repeated for the 494-project run (see §6) — no new leakage pathway found from the dataset expansion.

---

## 11. Limitations

1. **Extraction-infrastructure ceiling reached, not a data-availability ceiling.** All 8 previously-corrupted-download reports were successfully re-downloaded at full size in this final pass. However, re-running text extraction on these specific large (30-36MB) re-downloaded PDFs repeatedly failed to complete within a reasonable time budget in this environment (multiple independent timeout mechanisms — Python's `ProcessPoolExecutor`, PowerShell's `WaitForExit`, git-bash's `timeout` — all proved unable to reliably bound a pypdf hang on this Windows setup; a final clean, uninterfered-with attempt still produced zero output after 25 minutes on the very first file). This is a **tooling/resource limitation, not evidence that further official data doesn't exist** — the reports themselves are confirmed to be legitimate, correctly-sized, and previously partially processable (one of the 8, FR_2020_05, did complete successfully in an earlier run and revealed a genuine parser bug for that report edition, described in the second point below). Per the standing instruction not to manufacture data when acquisition is infeasible, this line of expansion was stopped here rather than continuing to sink time into fragile process-timeout tooling.
2. **A genuine parser bug was found and fixed defensively, not by force-including bad data.** The one older-edition (May 2020) report that did complete extraction revealed that its "Month wise List of Completed Projects" section lacks the "Month wise List of Deleted Projects" boundary marker the parser anchors on (present in 2021+ editions), causing a fallback scan to swallow ~416,000 characters of unrelated tables and produce 1,695 spurious pseudo-records. A plausibility ceiling (`MAX_PLAUSIBLE_COMPLETED_RECORDS_PER_REPORT = 300`, based on the highest genuine count — 183 — confirmed anywhere in the successfully-parsed corpus) was added to `extract_reports.py` to detect and discard such section-boundary failures rather than silently ship misaligned rows. This means pre-2021 report editions are **not currently supported** by this parser and were correctly excluded, not included with corrupted data.
3. **The 494-project dataset does not include any of the 8 recently-recovered reports** as a direct consequence of (1) — the dataset used for every result in this report is the same 494-project version already built and independently verified before this final extraction attempt began.
4. **Sample size remains modest** (75 test projects at the primary cutoff) — individual point-estimate metrics carry meaningful sampling uncertainty, though this does not change the qualitative conclusion given the consistency across four different cutoffs and two dataset sizes.
5. **PAIMANA-format reports (2025+) remain unusable** for this dataset due to a confirmed field-misalignment bug on multi-word project names, and a genuinely different, non-overlapping ID namespace that would require separate longitudinal-identity work in any case.

---

## 12. Comparison Summary: What Changed From 225 → 494 Projects

- Dataset more than doubled (225 → 494), sector coverage broadened (8 → 12 sectors), consistent target distribution (similar median/positive-rate) — the acquisition and parsing pipeline is validated as stable and reproducible.
- **Model performance did not improve — it got measurably worse for both tree models** (RF R²: -1.10-ish-range → -1.399; XGBoost got notably worse: → -2.344).
- The root cause was identified precisely: a severe, structural train/test distribution shift driven by survivorship bias in the "Completed Projects" reporting population, confirmed robust across four different chronological cutoffs.
- This is the single most important finding of the entire Option 3 effort: **more data from the same source cannot fix this**, because the shift is inherent to how projects transition from "Ongoing" to "Completed" over time, not a sampling-noise problem that shrinks with n.

---

## 13. Final Decision: C — Abandon Option 3

Per the mandated one-of-three framework:
- **A (production-capable):** Ruled out. No configuration, across 2 dataset sizes × 4 cutoffs × 2 model families × 5 feature sets, achieves positive R² on honest chronological held-out data.
- **B (limited research model):** Considered and rejected. The one partial positive signal (RF's MAE modestly beats baseline in all 4 cutoffs) does not rise to "defensible signal" once weighed against: R² decisively negative everywhere (worse than predicting the constant mean, in variance-explained terms), RMSE worse than baseline in 3 of 4 cutoffs (the model is *worse specifically on large-overrun cases* — the exact cases a cost-overrun-risk tool exists to catch), and a confirmed structural covariate shift that undermines any claim the model has learned a generalizable relationship rather than fit train-period noise.
- **C (abandon Option 3):** **Selected.** The government Flash Report data source, as currently accessible and parseable, cannot support a reliable cost-overrun prediction model — not due to insufficient effort or insufficient data volume, but due to a structural selection-bias property of what becomes visible as "Completed" over time. This is a genuine, evidence-based negative result, not a failure to try hard enough.

---

## 14. Recommendation / Next Steps

1. **Do not deploy, wire into the API, or create any model artifact for Option 3.** No code changes to the live Module 1 API were made in this effort, consistent with instruction.
2. **Preserve all Option 3 artifacts as research record**, clearly separated from the MPLADS-scoped Module 1: `data/module1_infra_source/` (raw reports, extracted text, both dataset versions, model_dev outputs, this report) and `scripts/module1_infra_pipeline/` (extraction, dataset-building, training, robustness-check tooling) remain in place, untouched by and not integrated into `src/mplads/`.
3. **If this data source is revisited in the future**, the highest-value next step would not be more report-months of the same "Completed Projects" table, but rather **finding a data source that reports outcomes for a fixed cohort of projects regardless of completion status** (e.g., a true longitudinal panel tracking the same set of approved projects over time, whether still ongoing or completed) — that would directly remove the survivorship-bias mechanism identified here. Absent such a source, this line of work is not worth pursuing further within Option 3's current data-acquisition strategy.
4. **Modules 2, 7, 8, and 9 remain untouched**, as instructed throughout.
