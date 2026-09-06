"""Deep-dive on the one genuinely variable financial signal found: the 8.15%
of completed works where Amount Disbursed < Recommended Amount. Checks
whether this is a real, learnable, unevenly-distributed pattern (by Work
Type / State) or scattered noise, and looks for a natural (not arbitrary)
threshold in the ratio distribution.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\Data")
OUT_DIR = Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\data\mplads_option2\inventory")

rec = pd.read_csv(DATA_DIR / "Works_Recommended_Cleaned.csv", low_memory=False)
comp = pd.read_csv(DATA_DIR / "Works_Completed_Cleaned.csv", low_memory=False)
rec = rec[~rec["Work ID"].str.startswith("NA-", na=False)].copy()
rec_dedup = rec.sort_values("Recommended Date").drop_duplicates("Work ID", keep="last")

joined = comp.merge(rec_dedup, on="Work ID", how="inner", suffixes=("_comp", "_rec"))
joined["ratio"] = joined["Amount Disbursed (INR)"] / joined["Recommended Amount (INR)"]
joined["underspend_pct"] = (1 - joined["ratio"]) * 100  # positive = spent less than recommended

report = {}

# Full distribution of the ratio (not just the <1 subset)
report["ratio_distribution_percentiles"] = {
    str(p): round(float(joined["ratio"].quantile(p / 100)), 4) for p in [1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
}
report["ratio_value_counts_rounded"] = joined["ratio"].round(2).value_counts().sort_index(ascending=False).head(20).to_dict()

# Histogram of underspend_pct among the <1.0 subset to look for a natural gap
below = joined[joined["ratio"] < 0.999]
report["n_below"] = len(below)
report["underspend_pct_histogram_below_subset"] = np.histogram(below["underspend_pct"], bins=20)[0].tolist()
report["underspend_pct_bin_edges"] = np.histogram(below["underspend_pct"], bins=20)[1].round(2).tolist()

# By Work Type -- is underspend concentrated in specific categories?
joined["is_underspend"] = joined["ratio"] < 0.999
by_type = joined.groupby("Work Type_rec").agg(n=("Work ID", "size"), underspend_rate=("is_underspend", "mean")).query("n >= 30").sort_values("underspend_rate", ascending=False)
report["underspend_rate_by_work_type_min30"] = by_type.head(15).round(4).to_dict(orient="index")
report["underspend_rate_by_work_type_min30_bottom"] = by_type.tail(10).round(4).to_dict(orient="index")

# By State
by_state = joined.groupby("State_rec").agg(n=("Work ID", "size"), underspend_rate=("is_underspend", "mean")).query("n >= 30").sort_values("underspend_rate", ascending=False)
report["underspend_rate_by_state_min30_top"] = by_state.head(10).round(4).to_dict(orient="index")
report["underspend_rate_by_state_min30_bottom"] = by_state.tail(10).round(4).to_dict(orient="index")

# Correlation with recommended amount size -- do larger/smaller works underspend more?
report["corr_underspend_pct_vs_recommended_amount"] = round(float(joined[["Recommended Amount (INR)", "underspend_pct"]].corr().iloc[0, 1]), 4)

# Overall variance check: is underspend_pct (over ALL completed works, not just <1 subset) sufficiently variable?
report["underspend_pct_full_population_stats"] = {
    "mean": round(float(joined["underspend_pct"].mean()), 4),
    "std": round(float(joined["underspend_pct"].std()), 4),
    "min": round(float(joined["underspend_pct"].min()), 4),
    "max": round(float(joined["underspend_pct"].max()), 4),
}

print(json.dumps(report, indent=2, default=str))
(OUT_DIR / "underspend_investigation.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
