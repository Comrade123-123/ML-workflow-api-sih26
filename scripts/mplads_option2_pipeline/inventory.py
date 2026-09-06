"""Phase 1: full inventory of every MPLADS CSV in Data/. Pure inspection --
no joins, no target assumptions, no modeling. Writes one JSON file per
dataset plus a combined data_dictionary.json under
data/mplads_option2/inventory/.
"""
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\Data")
OUT_DIR = Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\data\mplads_option2\inventory")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "Works_Recommended_Cleaned.csv",
    "Works_Completed_Cleaned.csv",
    "Expenditure_Cleaned.csv",
    "Allocated_Limit_LokSabha_Cleaned.csv",
    "Allocated_Limit_RajyaSabha_Cleaned.csv",
    "Amount_Consented_Calamity_Cleaned.csv",
]

DATE_LIKE = ("date",)
AMOUNT_LIKE = ("amount", "inr")


def profile_column(s: pd.Series, name: str) -> dict:
    n = len(s)
    missing = int(s.isna().sum())
    info = {
        "dtype": str(s.dtype),
        "n": n,
        "missing": missing,
        "missing_pct": round(100 * missing / n, 3) if n else None,
        "n_unique": int(s.nunique(dropna=True)),
    }
    lname = name.lower()
    if any(k in lname for k in AMOUNT_LIKE):
        numeric = pd.to_numeric(s, errors="coerce")
        info["numeric_parseable_pct"] = round(100 * numeric.notna().mean(), 3)
        info["min"] = float(numeric.min()) if numeric.notna().any() else None
        info["max"] = float(numeric.max()) if numeric.notna().any() else None
        info["mean"] = float(numeric.mean()) if numeric.notna().any() else None
        info["median"] = float(numeric.median()) if numeric.notna().any() else None
        info["n_negative"] = int((numeric < 0).sum())
        info["n_zero"] = int((numeric == 0).sum())
    if any(k in lname for k in DATE_LIKE):
        parsed = pd.to_datetime(s, errors="coerce")
        info["date_parseable_pct"] = round(100 * parsed.notna().mean(), 3)
        if parsed.notna().any():
            info["min_date"] = str(parsed.min())
            info["max_date"] = str(parsed.max())
    else:
        # small-cardinality text columns: show top values
        if info["n_unique"] <= 30:
            vc = s.value_counts(dropna=True).head(15)
            info["top_values"] = {str(k): int(v) for k, v in vc.items()}
    return info


data_dictionary = {}

for fname in FILES:
    path = DATA_DIR / fname
    df = pd.read_csv(path, low_memory=False)
    entry = {
        "file": fname,
        "row_count": len(df),
        "columns": list(df.columns),
        "column_profiles": {},
    }
    if "Work ID" in df.columns:
        entry["unique_work_ids"] = int(df["Work ID"].nunique(dropna=True))
        entry["duplicate_work_id_rows"] = int(len(df) - df["Work ID"].nunique(dropna=True))
        entry["work_id_missing"] = int(df["Work ID"].isna().sum())
    for col in df.columns:
        entry["column_profiles"][col] = profile_column(df[col], col)
    data_dictionary[fname] = entry
    (OUT_DIR / f"{fname.replace('.csv','')}_profile.json").write_text(
        json.dumps(entry, indent=2, default=str), encoding="utf-8"
    )
    print(f"Profiled {fname}: {len(df)} rows, {len(df.columns)} cols")

(OUT_DIR / "data_dictionary.json").write_text(json.dumps(data_dictionary, indent=2, default=str), encoding="utf-8")
print("\nWrote data_dictionary.json")
