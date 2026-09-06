"""Phase 1: expenditure-ledger-specific audit, going beyond Option 2's
inventory (which covered the same 3 CSVs at a schema level). Focuses on the
per-work PAYMENT STRUCTURE: how many payments, over what time span, how
concentrated across vendors -- the raw material for any timing/concentration
anomaly definition (Phase 3 candidates A/B/E).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\Data")
OUT_DIR = Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\data\mplads_option2a\inventory")

exp = pd.read_csv(DATA_DIR / "Expenditure_Cleaned.csv", low_memory=False)
rec = pd.read_csv(DATA_DIR / "Works_Recommended_Cleaned.csv", low_memory=False)
comp = pd.read_csv(DATA_DIR / "Works_Completed_Cleaned.csv", low_memory=False)

exp["Expenditure Date"] = pd.to_datetime(exp["Expenditure Date"], errors="coerce")
rec["Recommended Date"] = pd.to_datetime(rec["Recommended Date"], errors="coerce")
rec["Sanction Date"] = pd.to_datetime(rec["Sanction Date"], errors="coerce")
comp["Completion Date"] = pd.to_datetime(comp["Completion Date"], errors="coerce")

report = {}

# ---------------------------------------------------------------------
# Payment-count distribution per Work ID
# ---------------------------------------------------------------------
per_work = exp.groupby("Work ID").agg(
    n_payments=("Fund Disbursed Amount (INR)", "size"),
    total_disbursed=("Fund Disbursed Amount (INR)", "sum"),
    n_vendors=("Vendor Name", "nunique"),
    max_single_payment=("Fund Disbursed Amount (INR)", "max"),
    first_payment_date=("Expenditure Date", "min"),
    last_payment_date=("Expenditure Date", "max"),
    n_in_progress=("Payment Status", lambda s: (s == "Payment In-Progress").sum()),
).reset_index()
per_work["span_days"] = (per_work["last_payment_date"] - per_work["first_payment_date"]).dt.days
per_work["max_payment_share"] = per_work["max_single_payment"] / per_work["total_disbursed"]
per_work["pct_in_progress"] = per_work["n_in_progress"] / per_work["n_payments"]

report["n_payments_distribution"] = {
    "min": int(per_work["n_payments"].min()),
    "p25": float(per_work["n_payments"].quantile(0.25)),
    "median": float(per_work["n_payments"].median()),
    "p75": float(per_work["n_payments"].quantile(0.75)),
    "p95": float(per_work["n_payments"].quantile(0.95)),
    "p99": float(per_work["n_payments"].quantile(0.99)),
    "max": int(per_work["n_payments"].max()),
    "pct_single_payment_works": round(100 * (per_work["n_payments"] == 1).mean(), 3),
}

report["n_vendors_distribution"] = {
    "pct_single_vendor_works": round(100 * (per_work["n_vendors"] == 1).mean(), 3),
    "median_vendors": float(per_work["n_vendors"].median()),
    "max_vendors": int(per_work["n_vendors"].max()),
}

report["max_payment_share_distribution"] = {
    "note": "share of total work disbursement carried by its single largest payment -- 1.0 for single-payment works by construction",
    "median": round(float(per_work["max_payment_share"].median()), 4),
    "p25": round(float(per_work["max_payment_share"].quantile(0.25)), 4),
    "p10": round(float(per_work["max_payment_share"].quantile(0.10)), 4),
}

report["span_days_distribution"] = {
    "note": "0 for single-payment works by construction",
    "median": float(per_work["span_days"].median()),
    "p75": float(per_work["span_days"].quantile(0.75)),
    "p95": float(per_work["span_days"].quantile(0.95)),
    "max": float(per_work["span_days"].max()),
    "pct_zero_span": round(100 * (per_work["span_days"] == 0).mean(), 3),
}

report["payment_status_in_progress"] = {
    "overall_pct_rows_in_progress": round(100 * (exp["Payment Status"] == "Payment In-Progress").mean(), 3),
    "pct_works_with_any_in_progress": round(100 * (per_work["n_in_progress"] > 0).mean(), 3),
}

# ---------------------------------------------------------------------
# Vendor-level patterns: does a small number of vendors dominate?
# ---------------------------------------------------------------------
vendor_totals = exp.groupby("Vendor Name")["Fund Disbursed Amount (INR)"].agg(["sum", "count"]).sort_values("sum", ascending=False)
report["vendor_concentration"] = {
    "n_unique_vendors": int(exp["Vendor Name"].nunique()),
    "top_10_vendor_share_of_total_disbursement_pct": round(100 * vendor_totals["sum"].head(10).sum() / exp["Fund Disbursed Amount (INR)"].sum(), 3),
    "top_1pct_vendor_share_of_total_disbursement_pct": round(100 * vendor_totals["sum"].head(int(len(vendor_totals) * 0.01)).sum() / exp["Fund Disbursed Amount (INR)"].sum(), 3),
    "vendors_appearing_only_once": int((vendor_totals["count"] == 1).sum()),
    "vendors_appearing_only_once_pct": round(100 * (vendor_totals["count"] == 1).mean(), 3),
    "max_vendor_n_payments": int(vendor_totals["count"].max()),
    "max_vendor_n_distinct_works": int(exp.groupby("Vendor Name")["Work ID"].nunique().max()),
}

# Do any vendors work across many different States/MPs (breadth of operation)?
vendor_breadth = exp.groupby("Vendor Name").agg(n_states=("State", "nunique"), n_mps=("Hon'ble Members of Parliament", "nunique"), n_works=("Work ID", "nunique"))
report["vendor_breadth"] = {
    "vendors_spanning_gt5_states": int((vendor_breadth["n_states"] > 5).sum()),
    "vendors_spanning_gt10_mps": int((vendor_breadth["n_mps"] > 10).sum()),
    "max_states_single_vendor": int(vendor_breadth["n_states"].max()),
    "max_mps_single_vendor": int(vendor_breadth["n_mps"].max()),
}

# ---------------------------------------------------------------------
# Timing relative to recommendation/sanction: does expenditure ever precede
# sanction (should be impossible -- checked already for recommendation in
# Option 2, re-verify against Sanction Date here specifically)?
# ---------------------------------------------------------------------
rec_clean = rec[~rec["Work ID"].str.startswith("NA-", na=False)].sort_values("Recommended Date").drop_duplicates("Work ID", keep="last")
exp_joined = exp.merge(rec_clean[["Work ID", "Sanction Date", "Recommended Date"]], on="Work ID", how="inner")
exp_joined_valid = exp_joined.dropna(subset=["Expenditure Date", "Sanction Date"])
report["expenditure_before_sanction_violations"] = int((exp_joined_valid["Expenditure Date"] < exp_joined_valid["Sanction Date"]).sum())
report["n_expenditure_rows_checked_against_sanction"] = len(exp_joined_valid)

# Same-day-as-sanction full disbursement: a specific "unusual timing" pattern
# -- full recommended amount paid in a single payment on the sanction date itself.
single_payment_works = per_work[per_work["n_payments"] == 1].merge(rec_clean[["Work ID", "Sanction Date"]], on="Work ID", how="inner")
report["single_payment_on_sanction_date_pct_of_single_payment_works"] = round(
    100 * (single_payment_works["first_payment_date"] == single_payment_works["Sanction Date"]).mean(), 3
) if len(single_payment_works) else None

per_work.to_csv(Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\data\mplads_option2a\datasets") / "per_work_ledger_features.csv", index=False)

print(json.dumps(report, indent=2, default=str))
(OUT_DIR / "expenditure_audit.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
