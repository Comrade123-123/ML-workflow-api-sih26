"""Phase 2/3/6: builds the work-level anomaly-scoring dataset by joining
Option 2's completed-works dataset with this pipeline's per-work ledger
features, then engineers CAUSAL peer-group z-scores (strictly-earlier
sanctioned works only, by Work Type and separately by State) for every raw
ledger feature. No ground-truth label exists anywhere in the source data
(confirmed -- no flag/status field means "anomalous"/"flagged"/"suspicious"
appears in any of the 6 official CSVs), so this is unsupervised by
necessity, not by choice.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OPTION2_DATASET_DIR = Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\data\mplads_option2\datasets")
OPTION2A_DATASET_DIR = Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\data\mplads_option2a\datasets")

completed = pd.read_csv(OPTION2_DATASET_DIR / "work_level_completed_dataset.csv")
ledger = pd.read_csv(OPTION2A_DATASET_DIR / "per_work_ledger_features.csv")

completed["Sanction Date"] = pd.to_datetime(completed["Sanction Date"])
ledger["first_payment_date"] = pd.to_datetime(ledger["first_payment_date"])
ledger["last_payment_date"] = pd.to_datetime(ledger["last_payment_date"])

df = completed.merge(ledger, on="Work ID", how="inner")  # only works with ledger coverage (82.76% of completed works, per Phase 1)

# Raw signals to peer-normalize (all computable causally, none require
# looking at any OTHER work's outcome, only this work's own ledger + peers' history)
RAW_SIGNALS = ["n_payments", "n_vendors", "max_payment_share", "span_days", "ratio_disbursed_to_recommended"]

df = df.sort_values(["Sanction Date", "Work ID"]).reset_index(drop=True)


def causal_zscore(df, group_col, signal_col):
    """For each row, z-score of signal_col vs the STRICTLY EARLIER-sanctioned
    peers sharing group_col. Falls back to 0.0 (neutral) when fewer than 10
    prior peers exist yet (z-score undefined/unstable with tiny samples)."""
    z = np.zeros(len(df))
    sums, sqsums, counts = {}, {}, {}
    vals = df[signal_col].values
    grps = df[group_col].values
    for i in range(len(df)):
        g = grps[i]
        c = counts.get(g, 0)
        if c >= 10:
            mean = sums[g] / c
            var = max(sqsums[g] / c - mean ** 2, 1e-9)
            std = var ** 0.5
            z[i] = (vals[i] - mean) / std if std > 0 else 0.0
        else:
            z[i] = 0.0
        sums[g] = sums.get(g, 0.0) + vals[i]
        sqsums[g] = sqsums.get(g, 0.0) + vals[i] ** 2
        counts[g] = c + 1
    return z


for signal in RAW_SIGNALS:
    df[f"{signal}_zscore_by_worktype"] = causal_zscore(df, "Work Type", signal)
    df[f"{signal}_zscore_by_state"] = causal_zscore(df, "State", signal)

out_cols = ["Work ID", "State", "Work Type", "Work Category", "Recommended Amount (INR)",
            "Sanction Date", "Completion Date", "ratio_disbursed_to_recommended",
            "has_underspend"] + RAW_SIGNALS + [c for c in df.columns if "zscore" in c]
out = df[out_cols]
out.to_csv(OPTION2A_DATASET_DIR / "anomaly_scoring_dataset.csv", index=False)

summary = {
    "n_rows": len(out),
    "n_completed_works_total": len(completed),
    "ledger_coverage_pct": round(100 * len(out) / len(completed), 3),
    "raw_signal_stats": {s: {"median": float(df[s].median()), "p95": float(df[s].quantile(0.95)), "max": float(df[s].max())} for s in RAW_SIGNALS},
}
print(json.dumps(summary, indent=2, default=str))
(OPTION2A_DATASET_DIR / "anomaly_scoring_dataset_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
