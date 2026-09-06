"""Checks whether a completion-status or time-to-completion target would
suffer the same right-censoring / survivorship-bias problem that caused
Option 3's negative R2: do recently-recommended works simply look
"less likely to be complete" purely because they haven't had time yet,
regardless of any recommendation-time feature?
"""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\Data")
OUT_DIR = Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\data\mplads_option2\inventory")

rec = pd.read_csv(DATA_DIR / "Works_Recommended_Cleaned.csv", low_memory=False)
comp = pd.read_csv(DATA_DIR / "Works_Completed_Cleaned.csv", low_memory=False)

rec = rec[~rec["Work ID"].str.startswith("NA-", na=False)].copy()  # exclude not-yet-sanctioned placeholders
rec["Recommended Date"] = pd.to_datetime(rec["Recommended Date"], errors="coerce")
rec["Sanction Date"] = pd.to_datetime(rec["Sanction Date"], errors="coerce")
comp["Completion Date"] = pd.to_datetime(comp["Completion Date"], errors="coerce")

completed_ids = set(comp["Work ID"])
rec["is_completed"] = rec["Work ID"].isin(completed_ids)

# "Snapshot date" proxy = latest date seen anywhere in the data
snapshot_date = pd.concat([rec["Recommended Date"], rec["Sanction Date"], comp["Completion Date"]]).max()

rec["sanction_age_days"] = (snapshot_date - rec["Sanction Date"]).dt.days
rec["recommendation_age_days"] = (snapshot_date - rec["Recommended Date"]).dt.days

# Bucket by sanction-age month and compute completion rate per bucket.
rec["sanction_age_month_bucket"] = (rec["sanction_age_days"] // 30).clip(lower=0)
by_bucket = rec.groupby("sanction_age_month_bucket").agg(
    n=("Work ID", "size"), completion_rate=("is_completed", "mean")
).reset_index()

report = {
    "snapshot_date_proxy": str(snapshot_date),
    "n_sanctioned_works": len(rec),
    "overall_completion_rate": round(float(rec["is_completed"].mean()), 4),
    "completion_rate_by_sanction_age_month_bucket": by_bucket.to_dict(orient="records"),
}

# Correlation check: does completion rate rise monotonically with age since sanction?
# (If yes -> classic right-censoring: "not completed" mostly just means "not old enough yet".)
report["correlation_completion_vs_sanction_age_days"] = round(
    float(rec[["sanction_age_days", "is_completed"]].corr().iloc[0, 1]), 4
)

# Youngest-quartile vs oldest-quartile sanction-age completion rate comparison
q1, q3 = rec["sanction_age_days"].quantile([0.25, 0.75])
youngest = rec[rec["sanction_age_days"] <= q1]
oldest = rec[rec["sanction_age_days"] >= q3]
report["youngest_quartile_completion_rate"] = round(float(youngest["is_completed"].mean()), 4)
report["oldest_quartile_completion_rate"] = round(float(oldest["is_completed"].mean()), 4)
report["youngest_quartile_sanction_age_days_range"] = [int(rec["sanction_age_days"].min()), int(q1)]
report["oldest_quartile_sanction_age_days_range"] = [int(q3), int(rec["sanction_age_days"].max())]

print(json.dumps(report, indent=2, default=str))
(OUT_DIR / "censoring_check.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
