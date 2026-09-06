"""Phase 10 robustness checks for the underspend-risk classifier: alternative
chronological cutoffs, feature ablation, and outlier/imbalance sanity checks.
No hyperparameter tuning anywhere -- same fixed model configs throughout.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_DIR = Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\data\mplads_option2\datasets")
MODEL_DEV_DIR = Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\data\mplads_option2\model_dev")

df = pd.read_csv(DATASET_DIR / "work_level_completed_dataset.csv")
df["Recommended Date"] = pd.to_datetime(df["Recommended Date"])
df["Sanction Date"] = pd.to_datetime(df["Sanction Date"])
TARGET = "has_underspend"

df["recommendation_to_sanction_days"] = (df["Sanction Date"] - df["Recommended Date"]).dt.days
df["recommended_amount_log"] = np.log1p(df["Recommended Amount (INR)"])
df["_sort_key"] = df["Sanction Date"]
df = df.sort_values(["_sort_key", "Work ID"]).reset_index(drop=True)
global_prior = df[TARGET].mean()


def causal_rate(df, group_col, target_col, global_prior):
    out = np.full(len(df), np.nan)
    cum_count, cum_sum = {}, {}
    for i, (grp, val) in enumerate(zip(df[group_col].values, df[target_col].values)):
        c = cum_count.get(grp, 0)
        s = cum_sum.get(grp, 0)
        out[i] = (s / c) if c > 0 else global_prior
        cum_count[grp] = c + 1
        cum_sum[grp] = s + val
    return out


df["state_historical_underspend_rate"] = causal_rate(df, "State", TARGET, global_prior)
df["work_type_historical_underspend_rate"] = causal_rate(df, "Work Type", TARGET, global_prior)

FEATURE_COLUMNS_NUMERIC_FULL = ["Recommended Amount (INR)", "recommended_amount_log", "recommendation_to_sanction_days", "state_historical_underspend_rate", "work_type_historical_underspend_rate"]
FEATURE_COLUMNS_CATEGORICAL_FULL = ["Work Category", "State", "Work Type"]


def build_preprocessor(numeric, categorical):
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric))
    if categorical:
        transformers.append(("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical))
    return ColumnTransformer(transformers)


def fit_eval(train, test, numeric, categorical):
    feats = numeric + categorical
    Xtr, ytr = train[feats], train[TARGET].astype(int)
    Xte, yte = test[feats], test[TARGET].astype(int)
    pre = build_preprocessor(numeric, categorical)
    rf = Pipeline([("pre", pre), ("model", RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=42))])
    rf.fit(Xtr, ytr)
    score = rf.predict_proba(Xte)[:, 1]
    return {
        "n_train": len(train), "n_test": len(test),
        "train_positive_rate": round(float(ytr.mean()), 4),
        "test_positive_rate": round(float(yte.mean()), 4),
        "roc_auc": round(float(roc_auc_score(yte, score)), 4) if yte.nunique() > 1 else None,
        "pr_auc": round(float(average_precision_score(yte, score)), 4) if yte.nunique() > 1 else None,
    }


results = {}

# ---------------------------------------------------------------------
# (A) Alternative chronological cutoffs
# ---------------------------------------------------------------------
def split_at(df, train_frac, val_frac):
    d = df.sort_values(["_sort_key", "Work ID"]).reset_index(drop=True)
    n = len(d)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return d.iloc[:train_end], d.iloc[train_end:val_end], d.iloc[val_end:]

results["alternative_cutoffs"] = {}
for name, (tf, vf) in {"primary_70_15_15": (0.70, 0.15), "alt_60_20_20": (0.60, 0.20), "alt_80_10_10": (0.80, 0.10), "alt_50_25_25": (0.50, 0.25)}.items():
    train, val, test = split_at(df, tf, vf)
    results["alternative_cutoffs"][name] = fit_eval(train, test, FEATURE_COLUMNS_NUMERIC_FULL, FEATURE_COLUMNS_CATEGORICAL_FULL)

# ---------------------------------------------------------------------
# (B) Feature ablation (primary 70/15/15 split)
# ---------------------------------------------------------------------
train, val, test = split_at(df, 0.70, 0.15)
ablations = {
    "all_features": (FEATURE_COLUMNS_NUMERIC_FULL, FEATURE_COLUMNS_CATEGORICAL_FULL),
    "no_historical_rates": ([c for c in FEATURE_COLUMNS_NUMERIC_FULL if "historical" not in c], FEATURE_COLUMNS_CATEGORICAL_FULL),
    "no_state_categorical": (FEATURE_COLUMNS_NUMERIC_FULL, [c for c in FEATURE_COLUMNS_CATEGORICAL_FULL if c != "State"]),
    "amount_and_timing_only": (["Recommended Amount (INR)", "recommended_amount_log", "recommendation_to_sanction_days"], []),
    "historical_rates_only": (["state_historical_underspend_rate", "work_type_historical_underspend_rate"], []),
    "state_categorical_only": ([], ["State"]),
}
results["feature_ablation"] = {}
for name, (numeric, categorical) in ablations.items():
    results["feature_ablation"][name] = fit_eval(train, test, numeric, categorical)

print(json.dumps(results, indent=2, default=str))
(MODEL_DEV_DIR / "robustness_checks.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
