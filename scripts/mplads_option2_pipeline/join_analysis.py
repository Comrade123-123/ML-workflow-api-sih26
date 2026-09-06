"""Phase 2: deterministic Work-ID joins between Works_Recommended,
Works_Completed, and Expenditure. No fuzzy matching anywhere. Reports match
rates, duplicate-ID handling, date-order violations, and -- the special
investigation -- whether Amount Disbursed ever exceeds Recommended Amount
for the same Work ID.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\Data")
OUT_DIR = Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\data\mplads_option2\inventory")

rec = pd.read_csv(DATA_DIR / "Works_Recommended_Cleaned.csv", low_memory=False)
comp = pd.read_csv(DATA_DIR / "Works_Completed_Cleaned.csv", low_memory=False)
exp = pd.read_csv(DATA_DIR / "Expenditure_Cleaned.csv", low_memory=False)

rec["Recommended Date"] = pd.to_datetime(rec["Recommended Date"], errors="coerce")
rec["Sanction Date"] = pd.to_datetime(rec["Sanction Date"], errors="coerce")
comp["Completion Date"] = pd.to_datetime(comp["Completion Date"], errors="coerce")
exp["Expenditure Date"] = pd.to_datetime(exp["Expenditure Date"], errors="coerce")

report = {}

# ---------------------------------------------------------------------
# 1. Duplicate Work IDs within Works_Recommended: exact dup rows vs amended?
# ---------------------------------------------------------------------
dup_ids = rec["Work ID"].value_counts()
dup_ids = dup_ids[dup_ids > 1]
report["works_recommended_duplicate_work_ids"] = {
    "n_work_ids_with_duplicates": int(len(dup_ids)),
    "n_rows_involved": int(dup_ids.sum()),
}
# Sample: are duplicate-ID rows byte-identical, or do they differ (amendment)?
sample_dup_id = dup_ids.index[0] if len(dup_ids) else None
if sample_dup_id:
    sample_rows = rec[rec["Work ID"] == sample_dup_id]
    report["works_recommended_duplicate_sample"] = {
        "work_id": sample_dup_id,
        "n_rows": len(sample_rows),
        "rows_fully_identical": bool(sample_rows.drop(columns=["Sr. No."]).duplicated(keep=False).all()),
        "distinct_recommended_amounts": sample_rows["Recommended Amount (INR)"].nunique(),
        "distinct_recommended_dates": sample_rows["Recommended Date"].nunique(),
    }
# Systematic check: for ALL duplicate-ID groups, are they fully identical (ignoring Sr. No.) or do values differ?
dup_groups = rec[rec["Work ID"].isin(dup_ids.index)]
non_id_cols = [c for c in rec.columns if c not in ("Sr. No.", "Work ID")]
identical_flag = dup_groups.groupby("Work ID")[non_id_cols].apply(lambda g: g.drop_duplicates().shape[0] == 1)
report["works_recommended_duplicate_groups_fully_identical_pct"] = round(100 * identical_flag.mean(), 3)
report["works_recommended_duplicate_groups_with_differing_values"] = int((~identical_flag).sum())

# ---------------------------------------------------------------------
# 2. Match rate: Works_Completed -> Works_Recommended (deterministic Work ID)
# ---------------------------------------------------------------------
rec_dedup_last = rec.sort_values("Recommended Date").drop_duplicates("Work ID", keep="last")
comp_ids = set(comp["Work ID"])
rec_ids = set(rec["Work ID"])
matched = comp_ids & rec_ids
report["match_completed_to_recommended"] = {
    "n_completed_works": len(comp_ids),
    "n_matched_in_recommended": len(matched),
    "match_rate_pct": round(100 * len(matched) / len(comp_ids), 3),
    "n_completed_not_in_recommended": len(comp_ids - rec_ids),
}

# ---------------------------------------------------------------------
# 3. Match rate: Expenditure -> Works_Recommended and -> Works_Completed
# ---------------------------------------------------------------------
exp_ids = set(exp["Work ID"])
report["match_expenditure_to_recommended"] = {
    "n_expenditure_unique_work_ids": len(exp_ids),
    "n_matched_in_recommended": len(exp_ids & rec_ids),
    "match_rate_pct": round(100 * len(exp_ids & rec_ids) / len(exp_ids), 3),
}
report["match_expenditure_to_completed"] = {
    "n_expenditure_unique_work_ids": len(exp_ids),
    "n_matched_in_completed": len(exp_ids & comp_ids),
    "match_rate_pct": round(100 * len(exp_ids & comp_ids) / len(exp_ids), 3),
}

# ---------------------------------------------------------------------
# 4. THE SPECIAL INVESTIGATION: Recommended Amount vs Amount Disbursed
#    (Works_Completed), for deterministically-joined Work IDs only.
# ---------------------------------------------------------------------
joined = comp.merge(rec_dedup_last, on="Work ID", how="inner", suffixes=("_comp", "_rec"))
joined["ratio_disbursed_to_recommended"] = joined["Amount Disbursed (INR)"] / joined["Recommended Amount (INR)"]
joined["diff_disbursed_minus_recommended"] = joined["Amount Disbursed (INR)"] - joined["Recommended Amount (INR)"]

n_exceeds = int((joined["diff_disbursed_minus_recommended"] > 0.01).sum())
n_equal = int((joined["diff_disbursed_minus_recommended"].abs() <= 0.01).sum())
n_below = int((joined["diff_disbursed_minus_recommended"] < -0.01).sum())

report["special_investigation_disbursed_vs_recommended"] = {
    "n_joined_projects": len(joined),
    "n_disbursed_exceeds_recommended": n_exceeds,
    "pct_exceeds": round(100 * n_exceeds / len(joined), 3),
    "n_disbursed_equals_recommended_exact": n_equal,
    "pct_equals_exact": round(100 * n_equal / len(joined), 3),
    "n_disbursed_below_recommended": n_below,
    "pct_below": round(100 * n_below / len(joined), 3),
    "ratio_min": round(float(joined["ratio_disbursed_to_recommended"].min()), 4),
    "ratio_median": round(float(joined["ratio_disbursed_to_recommended"].median()), 4),
    "ratio_mean": round(float(joined["ratio_disbursed_to_recommended"].mean()), 4),
    "ratio_p90": round(float(joined["ratio_disbursed_to_recommended"].quantile(0.90)), 4),
    "ratio_p99": round(float(joined["ratio_disbursed_to_recommended"].quantile(0.99)), 4),
    "ratio_max": round(float(joined["ratio_disbursed_to_recommended"].max()), 4),
    "if_exceeds_sample_diffs": joined.loc[joined["diff_disbursed_minus_recommended"] > 0.01, "diff_disbursed_minus_recommended"].head(10).round(2).tolist(),
}

# ---------------------------------------------------------------------
# 5. Expenditure-ledger reconciliation: sum(Fund Disbursed) per Work ID vs
#    Works_Completed's single Amount Disbursed for that Work ID.
# ---------------------------------------------------------------------
exp_sum = exp.groupby("Work ID")["Fund Disbursed Amount (INR)"].sum().rename("expenditure_sum")
recon = comp.merge(exp_sum, on="Work ID", how="inner")
recon["diff"] = recon["Amount Disbursed (INR)"] - recon["expenditure_sum"]
report["reconciliation_completed_amount_vs_expenditure_sum"] = {
    "n_completed_works_with_expenditure_rows": len(recon),
    "n_completed_works_total": len(comp),
    "coverage_pct": round(100 * len(recon) / len(comp), 3),
    "n_exact_match": int((recon["diff"].abs() <= 0.01).sum()),
    "n_completed_amount_higher": int((recon["diff"] > 0.01).sum()),
    "n_completed_amount_lower": int((recon["diff"] < -0.01).sum()),
    "diff_median": round(float(recon["diff"].median()), 2),
    "diff_mean": round(float(recon["diff"].mean()), 2),
    "abs_diff_p90": round(float(recon["diff"].abs().quantile(0.90)), 2),
}

# ---------------------------------------------------------------------
# 6. Date-order violations
# ---------------------------------------------------------------------
rec_valid_dates = rec.dropna(subset=["Recommended Date", "Sanction Date"])
report["date_order_sanction_before_recommended"] = int((rec_valid_dates["Sanction Date"] < rec_valid_dates["Recommended Date"]).sum())

joined_dates = joined.dropna(subset=["Recommended Date", "Completion Date"])
report["date_order_completion_before_recommended"] = int((joined_dates["Completion Date"] < joined_dates["Recommended Date"]).sum())
joined_dates2 = joined.dropna(subset=["Sanction Date", "Completion Date"])
report["date_order_completion_before_sanction"] = int((joined_dates2["Completion Date"] < joined_dates2["Sanction Date"]).sum())

exp_joined = exp.merge(rec_dedup_last[["Work ID", "Recommended Date"]], on="Work ID", how="inner").dropna(subset=["Expenditure Date", "Recommended Date"])
report["date_order_expenditure_before_recommended"] = int((exp_joined["Expenditure Date"] < exp_joined["Recommended Date"]).sum())

print(json.dumps(report, indent=2, default=str))
(OUT_DIR / "join_analysis.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
