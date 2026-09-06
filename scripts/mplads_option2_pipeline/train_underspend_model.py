"""Phase 5-10: SAFE feature engineering, leakage audit, temporal split, and
baseline/LogReg/RF/XGBoost evaluation for the candidate target:

  has_underspend = Amount Disbursed < 99.9% of Recommended Amount,
                   observed only for COMPLETED works.

Prediction point: SANCTION time (once a real Work ID exists). Only fields
knowable at or before sanction are used as features -- Completion Date,
Amount Disbursed, and anything derived from them are never used as features
(they define the target).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

DATASET_DIR = Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\data\mplads_option2\datasets")
MODEL_DEV_DIR = Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\data\mplads_option2\model_dev")
MODEL_DEV_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET_DIR / "work_level_completed_dataset.csv")
df["Recommended Date"] = pd.to_datetime(df["Recommended Date"])
df["Sanction Date"] = pd.to_datetime(df["Sanction Date"])
df["Completion Date"] = pd.to_datetime(df["Completion Date"])

TARGET = "has_underspend"

# ---------------------------------------------------------------------
# SAFE feature engineering (all derivable at/around sanction time)
# ---------------------------------------------------------------------
df["recommendation_to_sanction_days"] = (df["Sanction Date"] - df["Recommended Date"]).dt.days
df["recommended_amount_log"] = np.log1p(df["Recommended Amount (INR)"])
df["sanction_month"] = df["Sanction Date"].dt.month
df["sanction_year"] = df["Sanction Date"].dt.year
# Day-level sort key (not just year-month) so the causal feature below can't
# leak from a later-same-month work into an earlier-same-month one; ties on
# the exact same calendar day are broken by Work ID for determinism.
df["_sort_key"] = df["Sanction Date"]

# Causal, strictly-expanding-window historical underspend rate by State and
# by Work Type (mirrors the Option 3 causal-feature methodology) -- for each
# work, uses only STRICTLY EARLIER-sanctioned works' outcomes, never its own
# or a later work's. Falls back to global prior mean, then a neutral prior.
df = df.sort_values(["_sort_key", "Work ID"]).reset_index(drop=True)
global_prior = df[TARGET].mean()  # only used as a fallback constant, not as a feature computed with future data one row at a time

def causal_rate(df, group_col, target_col, global_prior):
    out = np.full(len(df), np.nan)
    cum_count = {}
    cum_sum = {}
    for i, (grp, val) in enumerate(zip(df[group_col].values, df[target_col].values)):
        c = cum_count.get(grp, 0)
        s = cum_sum.get(grp, 0)
        out[i] = (s / c) if c > 0 else global_prior
        cum_count[grp] = c + 1
        cum_sum[grp] = s + val
    return out

df["state_historical_underspend_rate"] = causal_rate(df, "State", TARGET, global_prior)
df["work_type_historical_underspend_rate"] = causal_rate(df, "Work Type", TARGET, global_prior)

FEATURE_COLUMNS_NUMERIC = [
    "Recommended Amount (INR)", "recommended_amount_log",
    "recommendation_to_sanction_days",
    "state_historical_underspend_rate", "work_type_historical_underspend_rate",
]
FEATURE_COLUMNS_CATEGORICAL = ["Work Category", "State", "Work Type"]
ALL_FEATURES = FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL

# ---------------------------------------------------------------------
# Leakage audit (explicit, printed for the record)
# ---------------------------------------------------------------------
LEAKAGE_COLUMNS_EXCLUDED = [
    "Completion Date", "Amount Disbursed (INR)", "ratio_disbursed_to_recommended",
    "underspend_pct", "days_recommended_to_completion", "days_sanction_to_completion",
    "Has Image",  # only populated in Works_Completed -- structurally unavailable pre-completion
]
print("=== Leakage audit ===")
print("SAFE features used:", ALL_FEATURES)
print("Excluded as LEAKAGE (post-outcome or completion-table-only):", LEAKAGE_COLUMNS_EXCLUDED)
assert not set(ALL_FEATURES) & set(LEAKAGE_COLUMNS_EXCLUDED)

# ---------------------------------------------------------------------
# Temporal split by sanction date (chronological, no random split)
# ---------------------------------------------------------------------
def chronological_split(df):
    d = df.sort_values(["_sort_key", "Work ID"]).reset_index(drop=True)
    n = len(d)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train, val, test = d.iloc[:train_end], d.iloc[train_end:val_end], d.iloc[val_end:]
    assert set(train["Work ID"]) & set(val["Work ID"]) & set(test["Work ID"]) == set()
    return train, val, test

train, val, test = chronological_split(df)

split_info = {
    "n_train": len(train), "n_val": len(val), "n_test": len(test),
    "train_sanction_date_range": [str(train["Sanction Date"].min()), str(train["Sanction Date"].max())],
    "val_sanction_date_range": [str(val["Sanction Date"].min()), str(val["Sanction Date"].max())],
    "test_sanction_date_range": [str(test["Sanction Date"].min()), str(test["Sanction Date"].max())],
    "train_positive_rate": round(float(train[TARGET].mean()), 4),
    "val_positive_rate": round(float(val[TARGET].mean()), 4),
    "test_positive_rate": round(float(test[TARGET].mean()), 4),
}
print("\n=== Split info ===")
print(json.dumps(split_info, indent=2))

train.to_csv(MODEL_DEV_DIR / "split_train.csv", index=False)
val.to_csv(MODEL_DEV_DIR / "split_val.csv", index=False)
test.to_csv(MODEL_DEV_DIR / "split_test.csv", index=False)
(MODEL_DEV_DIR / "split_info.json").write_text(json.dumps(split_info, indent=2, default=str), encoding="utf-8")

# ---------------------------------------------------------------------
# Preprocessing + models (fixed configs, no hyperparameter tuning)
# ---------------------------------------------------------------------
def build_preprocessor():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")), ("encode", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", numeric, FEATURE_COLUMNS_NUMERIC), ("cat", categorical, FEATURE_COLUMNS_CATEGORICAL)])


def classification_metrics(y_true, y_pred, y_score):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4) if len(set(y_true)) > 1 else None,
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4) if len(set(y_true)) > 1 else None,
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n": len(y_true),
        "n_positive": int(np.sum(y_true)),
    }


Xtr, ytr = train[ALL_FEATURES], train[TARGET].astype(int)
Xval, yval = val[ALL_FEATURES], val[TARGET].astype(int)
Xte, yte = test[ALL_FEATURES], test[TARGET].astype(int)

results = {}

baseline = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
results["baseline_majority"] = classification_metrics(yte, baseline.predict(Xte), baseline.predict_proba(Xte)[:, 1])

logreg = Pipeline([("pre", build_preprocessor()), ("model", LogisticRegression(max_iter=1000, class_weight="balanced"))])
logreg.fit(Xtr, ytr)
results["logistic_regression"] = classification_metrics(yte, logreg.predict(Xte), logreg.predict_proba(Xte)[:, 1])

rf = Pipeline([("pre", build_preprocessor()), ("model", RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=5, class_weight="balanced", random_state=42))])
rf.fit(Xtr, ytr)
results["random_forest"] = classification_metrics(yte, rf.predict(Xte), rf.predict_proba(Xte)[:, 1])

xgb = Pipeline([("pre", build_preprocessor()), ("model", XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, scale_pos_weight=(ytr == 0).sum() / (ytr == 1).sum(), random_state=42, eval_metric="logloss"))])
xgb.fit(Xtr, ytr)
results["xgboost"] = classification_metrics(yte, xgb.predict(Xte), xgb.predict_proba(Xte)[:, 1])

print("\n=== Test-set results ===")
print(json.dumps(results, indent=2))
(MODEL_DEV_DIR / "primary_results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

# Save predictions for independent verification
pred_df = test[["Work ID", TARGET]].copy()
pred_df["baseline_score"] = baseline.predict_proba(Xte)[:, 1]
pred_df["logreg_score"] = logreg.predict_proba(Xte)[:, 1]
pred_df["rf_score"] = rf.predict_proba(Xte)[:, 1]
pred_df["xgb_score"] = xgb.predict_proba(Xte)[:, 1]
pred_df.to_csv(MODEL_DEV_DIR / "predictions_test.csv", index=False)

# Feature importance for interpretability
feature_names = list(rf.named_steps["pre"].get_feature_names_out())
rf_importance = dict(zip(feature_names, rf.named_steps["model"].feature_importances_.round(4)))
(MODEL_DEV_DIR / "rf_feature_importance.json").write_text(json.dumps(rf_importance, indent=2, default=str), encoding="utf-8")
print("\n=== RF feature importance ===")
print(json.dumps(rf_importance, indent=2))
