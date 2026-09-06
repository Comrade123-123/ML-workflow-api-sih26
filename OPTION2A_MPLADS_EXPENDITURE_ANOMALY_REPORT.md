# Option 2A — MPLADS Expenditure Anomaly / Financial Risk Feasibility Report

**Scope reminder:** Isolated research only. No changes were made to the Module 1 API, Modules 2/7/8/9, the frontend, the backend, or any existing dataset (including Option 2's own `has_underspend` artifacts). All work lives in `scripts/mplads_option2a_pipeline/` and `data/mplads_option2a/`.

---

## 1. Executive Verdict

**Final decision: C — No Better Target Found.** `has_underspend` (from Option 2) remains the best defensible MPLADS ML capability found so far. This investigation built a genuinely stable, reproducible unsupervised anomaly score from the expenditure ledger (payment count, vendor count, payment concentration, payment timing span), but it does not surpass `has_underspend` in defensibility: it has **no ground truth to validate against anywhere in the data** (no field in any of the six official CSVs ever means "flagged," "suspicious," or "irregular"), and the explicit scale-confound check required by the brief found that the top-ranked "anomalies" are substantially explained by legitimate project scale and multi-vendor complexity rather than a clean, complexity-controlled irregularity signal. The score is statistically well-behaved (stable across random seeds and across time) and could plausibly serve as a *secondary, transparent triage input* alongside `has_underspend` in the future, but it is not, on its own, a stronger or independently deployable capability today.

No fraud, corruption, or illegality is claimed anywhere in this report or its outputs. All scores mean, at most, "statistically unusual relative to comparable MPLADS works" and nothing more.

---

## 2. Data Inventory

Same six official CSVs as Option 2 (`Data/Works_Recommended_Cleaned.csv`, `Works_Completed_Cleaned.csv`, `Expenditure_Cleaned.csv`, plus the two allocation-ceiling files and the calamity-consent file, the latter three not used here). This report adds a **ledger-structure audit** of `Expenditure_Cleaned.csv` beyond what Option 2 examined (Option 2 used the ledger only to reconcile totals; this pass examines payment count, vendor count, and timing per work).

---

## 3. Expenditure Data Semantics

Every field's exact meaning, confirmed empirically (not assumed):

| Field | Meaning |
|---|---|
| Work ID | Deterministic key, ledger grain (repeats once per payment) |
| Expenditure Date | Date a specific payment was recorded |
| Vendor Name | The paid contractor/supplier for that specific payment (22,601 distinct vendors across the dataset) |
| Payment Status | `Payment Success` (95.3% of rows) or `Payment In-Progress` (4.7%) — no other values exist |
| Fund Disbursed Amount (INR) | The amount of that one payment (₹1 to ₹20,786,350) |

**Per-work payment structure (new finding, not examined in Option 2):**
- **74.6% of completed works have exactly ONE payment** covering their entire disbursement; only 25.4% have 2 or more, and only ~1% have 10 or more (max observed: 49 payments for one work).
- **94.5% of works use exactly ONE vendor.**
- For single-payment works, the payment's share of total disbursement is 1.0 by construction — meaningful "concentration" variation exists only within the ~25% multi-payment minority.
- **77.7% of works have a zero-day payment span** (all payments, if more than one, land on the same date) — meaningful "timing spread" variation again exists only in a minority.
- 5.74% of works have at least one payment still `In-Progress`.
- Vendor concentration: the top 1% of vendors (226 of 22,601) account for 27.3% of total disbursement value; 56.7% of vendors appear in the ledger only once. One vendor appears across 785 distinct works — a large, active contractor, not evidence of anything improper on its own. No vendor spans more than 5 States or 10 MPs — vendors are regionally confined, consistent with normal local-contractor behavior.
- **Zero** expenditure-before-sanction date violations (64,960 checked rows).
- Only 3.6% of single-payment works pay the full amount on the exact sanction date — full-amount-on-day-one is not the dominant pattern; most single payments happen at some later date, consistent with genuine (if simple) implementation timing.

**Implication:** because ~75% of works are structurally single-payment/single-vendor, any "unusual concentration" or "unusual timing" signal can only ever discriminate within the remaining ~25% — this ceiling was kept in view throughout the rest of the investigation, not discovered after the fact.

---

## 4. Deterministic Joins

Reused Option 2's already-validated deterministic Work-ID join (Works_Recommended ⋈ Works_Completed, excluding the 5,440 not-yet-sanctioned placeholder rows — see Option 2's report §4 for the full derivation). This pass adds: Works_Completed ⋈ per-work ledger aggregates (from `Expenditure_Cleaned.csv`, grouped by Work ID).

| Join | Coverage |
|---|---|
| Completed works with any ledger coverage | 12,414 / 15,000 = **82.76%** (unchanged from Option 2's reconciliation check) |
| Final anomaly-scoring dataset (completed + recommended + ledger, all three joined) | **12,386 works** |

No fuzzy matching was used anywhere. No new duplicate-ID or contradictory-date issues were found beyond what Option 2 already documented and excluded.

---

## 5. Candidate Anomaly Definitions Investigated

| # | Candidate | Data support | Outcome |
|---|---|---|---|
| A | Unusual expenditure timing | `span_days`, timing relative to sanction | Investigated — limited applicability (77.7% of works have zero span) |
| B | Unusually concentrated expenditure | `max_payment_share` (largest single payment ÷ total) | Investigated — limited applicability (94.5% single-vendor by construction) |
| C | Unusual expenditure-to-recommended ratio | `ratio_disbursed_to_recommended` | Already covered by Option 2's `has_underspend`; included here as one input feature, not re-litigated as a new target |
| D | Unusual pattern vs. peer works | Causal peer z-scores by Work Type / State | **Primary approach used** |
| E | Unusual payment frequency | `n_payments`, `n_vendors` | Investigated — included as peer-normalized features |
| F | Vendor/category/state patterns | Vendor breadth, State-level ledger stats | Investigated for face-validity only; explicitly not used to single out any vendor, MP, or state as a target, per instruction |
| G | Deviation from historical peer behavior | Causal expanding-window z-scores | **Primary approach used** |
| H | Other | — | Not found; the five raw signals above cover what the ledger structurally supports |

---

## 6. Target / Label Validity

Checked explicitly, per Phase 4:

1. **Is there an official label?** No. None of the six official CSVs contains any field meaning "flagged," "anomalous," "audited," "irregular," or similar, anywhere.
2. **Can a target be derived deterministically without circularity?** Only by defining "anomalous" as "statistically distant from peers on ledger-derived features" — which is a methodological choice, not a discovered ground truth.
3. **Does it depend on future information?** The raw signals (payment count, vendor count, concentration, span) are only fully known once the work is complete and its ledger is closed — so, like `has_underspend`, this can only be *retrospectively* scored for completed works, not prospectively predicted at sanction time from these signals themselves. (The historical peer-rate features, by contrast, are legitimately available prospectively.)
4. **Can it be predicted prospectively?** Only the causal peer-group historical rate portion; the works' own ledger signals cannot be known before the work executes.
5. **Useful for inspector prioritization?** Potentially, as a retrospective/near-real-time triage signal on works with substantial ledger activity already recorded — not as an at-sanction-time prediction.
6. **How many positive cases exist?** No fixed "positive class" exists (unsupervised); by construction, a top-1% cut yields 123 of 12,386 works.
7. **Is the target stable across time?** Yes — Spearman 0.89 between a model fit only on the early half of the data and one fit only on the late half, both scored on the same population.
8. **Is it dominated by State/Work Type?** Partially — see §10 ablation; State and Work-Type peer-groupings each individually reproduce 84-89% of the combined ranking, similar in spirit to (though less totalizing than) `has_underspend`'s finding.
9. **Does it merely rediscover an accounting rule?** No single deterministic rule reproduces it, but see §7's central caveat: it substantially rediscovers "this project is larger and more logistically complex," which is a real but unsurprising pattern, not a novel financial-irregularity signal.

**Conclusion: no supervised ground truth exists.** This is a factual finding, not a shortfall in effort — confirmed by inspecting every column of every official file. Per the brief's own contingency plan, the investigation proceeded to unsupervised methods.

---

## 7. Supervised vs. Unsupervised Decision

**Unsupervised**, by necessity (§6). Two complementary methods were used and compared:
1. **Peer-group statistical baseline** — a work's anomaly score is the maximum absolute z-score across all its causal peer-normalized features (transparent, auditable, no black box).
2. **Isolation Forest** — fit on the same causal peer z-score feature set, to check whether a genuinely multivariate method finds materially different structure than the simple transparent baseline.

The two methods agree substantially (Spearman 0.841 across the full population) but their exact top-1% lists overlap only 43.9% — meaning Isolation Forest does pick up some genuinely multivariate combinations the simple max-z-score baseline misses, but the two are fundamentally telling a similar story, not contradictory ones.

---

## 8. Feature Timing Audit

**Prediction/inspection point: ledger-closure time (i.e., once a work is complete and its full payment history is recorded)** — this is a retrospective/monitoring point, not a prospective at-sanction-time prediction point, because the ledger signals themselves (payment count, vendor count, concentration, span) only exist once payments have actually occurred.

| Feature | Classification | Rationale |
|---|---|---|
| `n_payments`, `n_vendors`, `max_payment_share`, `span_days` (raw, this work's own ledger) | **SAFE only at/after ledger closure**, NOT safe at sanction time | These are the work's own realized payment history |
| `ratio_disbursed_to_recommended` | Same as above — and is exactly Option 2's target, included here only as one input signal, never re-used as this report's own target | |
| `*_zscore_by_worktype`, `*_zscore_by_state` (causal, peer history) | **SAFE prospectively** | Strictly-expanding-window, uses only strictly-earlier-sanctioned peers' history, verified via the same causal-loop methodology used in Option 2 and Option 3 |
| Completion Date, final Amount Disbursed as standalone future values | **LEAKAGE** | Never used directly — only the already-causal `ratio_disbursed_to_recommended` (computed once, at ledger closure) enters the feature set |
| Future/post-inspection payments | **LEAKAGE** | Not applicable here since scoring happens at ledger closure, after all payments for that work are already recorded — no future payments exist relative to the scoring point for a given work |

**Important honest limitation:** because the four raw ledger signals are only knowable at ledger closure, this system is a **retrospective monitoring/triage tool for already-substantially-executed works**, not a prospective at-sanction-time risk predictor like `has_underspend`'s causal-feature portion. This is a materially different (and narrower) use case than what a "financial risk prediction" label might suggest, and is stated plainly here rather than implied away.

---

## 9. Modeling Results

Since no supervised target exists, "modeling results" here means the anomaly-scoring pipeline's output distribution, not classification/regression metrics:

- 12,386 works scored.
- Peer-group statistical baseline (max abs z-score) and Isolation Forest score, computed for every work.
- Top 20 highest-Isolation-Forest-score works saved for qualitative review (`data/mplads_option2a/model_dev/top20_anomalies_for_review.csv`) — inspected manually (§12) for face validity, not treated as confirmed findings.

---

## 10. Robustness

| Check | Result |
|---|---|
| **Seed stability** (Isolation Forest, 4 alternate seeds vs. primary seed=42) | Top-1% set overlap: 83.7%-89.4%. Spearman rank correlation: 0.961-0.976. **Highly stable** — the ranking is not seed noise. |
| **Temporal stability** (model fit on early half vs. late half of the data, both scored on the full population) | Spearman 0.89 — the pattern the model finds is consistent across time, not an artifact of one period. |
| **Feature-set ablation** (Work-Type-only vs. State-only vs. combined peer groups) | Work-Type-only vs. combined: 0.845. State-only vs. combined: 0.891. Work-Type-only vs. State-only (the two peer definitions against each other): 0.604 — moderate agreement; the two groupings capture related but not identical structure. |
| **Method agreement** (transparent peer-stat baseline vs. Isolation Forest) | Spearman 0.841 overall; only 43.9% exact top-1% overlap — broadly consistent, not identical. |
| **Scale/complexity confound check (the critical one)** | Top-1% flagged works have **median Recommended Amount ₹500,000 vs. ₹275,555** for the rest, **median 3 payments vs. 1**, **median 2 vendors vs. 1**, **median 219-day payment span vs. 0**. **The anomaly ranking is substantially explained by legitimate project scale and multi-vendor/multi-payment complexity, even after peer-normalization by Work Type and State.** This is the single most important robustness finding and directly limits how the score should be interpreted (§12). |

---

## 11. Comparison with `has_underspend`

| Dimension | `has_underspend` (Option 2) | Expenditure Anomaly Score (Option 2A) |
|---|---|---|
| Ground truth | Real, observed, verifiable outcome (5.71% base rate) | **None exists anywhere in the data** — purely statistical/relative |
| Leakage | Verified zero (independently re-checked) | Verified zero for the causal peer-features; raw ledger signals are inherently retrospective-only (§8) |
| Predictive signal | Robust across 4 chronological cutoffs, beats baseline on ROC-AUC/PR-AUC every time | Robust across seeds and time, but there is no baseline-beating claim possible without a target to predict — "robust" here means *reproducible*, not *predictively validated* |
| Feature richness | 5 features, dominated by State/Work-Type group membership (§12b of Option 2's report) | 5 ledger-derived signals × 2 peer-group definitions = 10 features, also substantially explained by group/scale effects |
| Prospective usability | Partial — causal historical-rate features are prospective; raw amount/timing are known at sanction | **Weaker** — the informative raw signals (payment count, concentration, span) are retrospective-only, unlike `has_underspend`'s sanction-time features |
| Project-level usefulness | Modest (precision ~17%, but a real event to be right or wrong about) | Weaker — no real event to validate against; "unusual" is definitionally self-referential |
| Inspector usefulness | Flag categories of work with historically elevated underspend rates | Flag already-executed works with unusual payment patterns for a closer look — useful chiefly as a scale/complexity-aware triage prompt (see §12) |
| Civic value | Moderate, clearly scoped | Moderate at best today, mostly redundant with "this is a large, multi-vendor project," which is already visible without modeling |

**Verdict on this comparison: `has_underspend` remains more defensible.** It predicts a concrete, real-world-verifiable event. The anomaly score, while methodologically sound and reproducible, currently mostly re-expresses project scale and complexity — information already visible from the Recommended Amount and payment count alone, without needing an anomaly model.

---

## 12. Inspector / Civic Use Case

**What would an inspector actually do with this?** Per the required framing, the honest answer given the confound in §10:
- Today, the top-ranked works are mostly large, multi-vendor, multi-payment projects spread over many months (e.g., an irrigation project with 23 payments across 19 vendors, a lighting project with 6 payments across 3 vendors over 343 days). An inspector looking at this list would reasonably say: *"these are bigger, more logistically involved projects — of course they look different from a simple single-payment purchase."* That is a legitimate, expected observation, not a novel discovery.
- The narrower, defensible use is: **use the score as a secondary sort key within an already-necessary review queue** (e.g., among works an inspector must review anyway for size/budget reasons, this score highlights which ones have unusually fragmented payment timing *even relative to other similarly large/complex works* — a genuinely finer-grained signal than raw size alone, since the peer-normalization is by Work Type and State, not by size).
- Appropriate inspector actions if this were ever used: review expenditure documentation for the flagged work, verify that a long payment span or many vendors corresponds to genuine phased implementation (not a red flag by itself), compare the work's pattern against its specific peers, and request supporting documentation if something still looks inconsistent after that manual comparison.
- **This system does not, and cannot, prove wrongdoing.** It also cannot yet cleanly separate "legitimately complex project" from "something worth a closer look" — that separation is exactly what the scale confound in §10 shows is not yet achieved.

---

## 13. Limitations

1. **No ground truth exists anywhere in the source data** — every claim of "usefulness" here is a plausibility argument, not a validated one (unlike `has_underspend`, which at least has a real, checkable outcome).
2. **Scale/complexity confound is not fully resolved** by Work-Type/State peer-normalization alone (§10) — a work being large and multi-vendor still pushes its score up even after normalization, likely because within-group variance in scale remains substantial.
3. **Structural ceiling**: 74.6% single-payment / 94.5% single-vendor works give the timing/concentration signals almost no variance to work with for three-quarters of the population — this system is only informative for the roughly one-quarter of works with genuinely multi-part payment histories.
4. **Retrospective, not prospective**, for its most informative features (§8) — this is a monitoring tool for substantially-executed works, not an at-sanction-time risk predictor.
5. **State/vendor-level patterns were investigated only for face validity**, not to single out any specific vendor, MP, or state — no such claim is made or intended by this report.
6. **No independent second unsupervised method beyond Isolation Forest + the peer-stat baseline** (e.g., Local Outlier Factor) was run in this pass, for time reasons — the two methods used already showed strong mutual consistency (§10), so a third method was judged unlikely to change the qualitative conclusion, but this was not empirically confirmed.

---

## 14. Final Decision: C — No Better Target Found

- **A (stronger capability):** Not selected — no ground truth exists to validate against, and the dominant scale/complexity confound means the score does not yet cleanly isolate "unusual" from "large and complex."
- **B (limited research capability):** Considered, but the comparison in §11 shows this does not clearly add *new* defensible value beyond what `has_underspend` and simple project-size/payment-count inspection already offer — it is more accurately a **reproducible but not-yet-validated** signal than an "interesting new capability."
- **C (no better target found): SELECTED.** `has_underspend` remains the best defensible MPLADS ML capability identified across both Option 2 and Option 2A. This investigation's anomaly score is real, reproducible, and worth preserving as research groundwork, but does not surpass or replace it.
- **D (no defensible capability at all):** Not selected — this would understate the genuine, if modest and unvalidated, structure found (seed/temporal stability, moderate cross-method agreement).

---

## 15. Exact Next Implementation Step

1. **Do not deploy anything from Option 2A.** No API, frontend, or backend changes were made; none are recommended at this stage.
2. If this line of work is revisited, the necessary next step is **not** more anomaly-detection tuning, but **resolving the scale confound directly** — e.g., by explicitly residualizing each ledger signal against Recommended Amount (regression-based normalization) before peer z-scoring, rather than relying on Work-Type/State grouping alone to absorb scale differences.
3. A **transparent, non-ML alternative** worth considering instead of a black-box anomaly score: a simple published rule such as "works with more than N payments across more than M vendors and a payment span exceeding P days, within their Work-Type/State peer group's top decile" — same underlying signal, fully auditable, and honestly framed as a complexity-aware triage rule rather than an ML "risk score."
4. **Preserve this research area** (`scripts/mplads_option2a_pipeline/`, `data/mplads_option2a/`) as isolated, versioned research, separate from Option 2's `has_underspend` artifacts and from all locked modules.
