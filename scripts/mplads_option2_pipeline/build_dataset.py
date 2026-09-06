"""Phase 2 deliverable: the documented, deterministic work-level analytical
dataset (Works_Recommended INNER JOIN Works_Completed on Work ID). No fuzzy
matching. Excludes the 5,440 not-yet-sanctioned placeholder rows (Work ID
literally "NA-<work type>", no real ID assigned pre-sanction -- confirmed
via Phase 2 to be exactly the same rows missing Sanction Date).
"""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\Data")
OUT_DIR = Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\data\mplads_option2\datasets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rec = pd.read_csv(DATA_DIR / "Works_Recommended_Cleaned.csv", low_memory=False)
comp = pd.read_csv(DATA_DIR / "Works_Completed_Cleaned.csv", low_memory=False)

not_yet_sanctioned = rec["Work ID"].str.startswith("NA-", na=False)
rec_clean = rec[~not_yet_sanctioned].copy()

# Zero genuine duplicate Work IDs after excluding placeholders (confirmed in
# join_analysis.py) -- dedup call kept only as a defensive no-op / documented
# invariant check, not because duplicates are expected.
before = len(rec_clean)
rec_clean = rec_clean.sort_values("Recommended Date").drop_duplicates("Work ID", keep="last")
assert len(rec_clean) == before, "Unexpected genuine duplicate Work IDs found -- investigate before proceeding."

rec_clean["Recommended Date"] = pd.to_datetime(rec_clean["Recommended Date"], errors="coerce")
rec_clean["Sanction Date"] = pd.to_datetime(rec_clean["Sanction Date"], errors="coerce")
comp["Completion Date"] = pd.to_datetime(comp["Completion Date"], errors="coerce")

df = comp.merge(rec_clean, on="Work ID", how="inner", suffixes=("_comp", "_rec"))

df["ratio_disbursed_to_recommended"] = df["Amount Disbursed (INR)"] / df["Recommended Amount (INR)"]
df["underspend_pct"] = (1 - df["ratio_disbursed_to_recommended"]) * 100
# Natural split observed in the data itself (not an arbitrary chosen %):
# 91.8% of projects disburse EXACTLY the recommended amount (ratio == 1.0,
# a genuine point mass, not a rounding artifact -- confirmed via exact-cent
# comparison in join_analysis.py). "has_underspend" separates that point
# mass from the continuous tail below it; no percentage threshold is chosen.
df["has_underspend"] = df["ratio_disbursed_to_recommended"] < 0.999

df["days_recommended_to_completion"] = (df["Completion Date"] - df["Recommended Date"]).dt.days
df["days_sanction_to_completion"] = (df["Completion Date"] - df["Sanction Date"]).dt.days

keep_cols = [
    "Work ID", "Work Category_rec", "Work Type_rec", "Work Description_rec",
    "State_rec", "IDA District_rec", "IDA Office_rec",
    "Hon'ble Members of Parliament_rec", "Constituency_rec",
    "Recommended Date", "Recommended Amount (INR)", "Sanction Date",
    "Completion Date", "Amount Disbursed (INR)", "Has Image",
    "ratio_disbursed_to_recommended", "underspend_pct", "has_underspend",
    "days_recommended_to_completion", "days_sanction_to_completion",
]
out = df[keep_cols].rename(columns=lambda c: c[: -len("_rec")] if c.endswith("_rec") else c)

out.to_csv(OUT_DIR / "work_level_completed_dataset.csv", index=False)

summary = {
    "n_rows": len(out),
    "n_unique_work_ids": int(out["Work ID"].nunique()),
    "date_range_recommended": [str(out["Recommended Date"].min()), str(out["Recommended Date"].max())],
    "date_range_completion": [str(out["Completion Date"].min()), str(out["Completion Date"].max())],
    "has_underspend_rate": round(float(out["has_underspend"].mean()), 4),
    "n_negative_days_recommended_to_completion": int((out["days_recommended_to_completion"] < 0).sum()),
    "n_negative_days_sanction_to_completion": int((out["days_sanction_to_completion"] < 0).sum()),
}
print(json.dumps(summary, indent=2, default=str))
(OUT_DIR / "work_level_completed_dataset_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
