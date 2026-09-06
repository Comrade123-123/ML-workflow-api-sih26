"""Phase 5/7/8: unsupervised anomaly scoring (Isolation Forest + a simple
peer-group statistical baseline), with the robustness checks Phase 8
explicitly requires: seed stability, feature ablation, peer-group
definition (Work Type vs State vs combined), and a check for whether
anomalies are just legitimate scale/complexity differences (multi-payment,
larger works) rather than something meriting review.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DATASET_DIR = Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\data\mplads_option2a\datasets")
MODEL_DEV_DIR = Path(r"C:\Users\manas\OneDrive\Desktop\ml-workflow\data\mplads_option2a\model_dev")
MODEL_DEV_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATASET_DIR / "anomaly_scoring_dataset.csv")

WORKTYPE_Z_FEATURES = [c for c in df.columns if c.endswith("_zscore_by_worktype")]
STATE_Z_FEATURES = [c for c in df.columns if c.endswith("_zscore_by_state")]
COMBINED_FEATURES = WORKTYPE_Z_FEATURES + STATE_Z_FEATURES

report = {}


def run_iforest(X, seed):
    model = IsolationForest(n_estimators=200, contamination="auto", random_state=seed)
    model.fit(X)
    # lower score = more anomalous in sklearn's convention; flip sign so higher = more anomalous
    return -model.score_samples(X)


# ---------------------------------------------------------------------
# (1) Peer-group statistical baseline: max absolute z-score across features
# ---------------------------------------------------------------------
df["peer_stat_score"] = df[COMBINED_FEATURES].abs().max(axis=1)

# ---------------------------------------------------------------------
# (2) Isolation Forest, primary run (combined features, seed=42)
# ---------------------------------------------------------------------
X_combined = df[COMBINED_FEATURES].fillna(0.0).values
df["iforest_score_primary"] = run_iforest(X_combined, seed=42)

# ---------------------------------------------------------------------
# (3) Seed stability: rerun with 4 other seeds, compare top-1% overlap
# ---------------------------------------------------------------------
top_pct = 0.01
n_top = max(int(len(df) * top_pct), 1)
seed_scores = {42: df["iforest_score_primary"].values}
for seed in [1, 7, 123, 2024]:
    seed_scores[seed] = run_iforest(X_combined, seed=seed)

top_sets = {seed: set(np.argsort(-scores)[:n_top]) for seed, scores in seed_scores.items()}
primary_top = top_sets[42]
overlap_report = {}
for seed, s in top_sets.items():
    if seed == 42:
        continue
    overlap_report[f"seed_{seed}_overlap_with_primary_pct"] = round(100 * len(s & primary_top) / n_top, 2)
report["seed_stability"] = overlap_report

# Rank correlation across seeds (Spearman) -- a stricter stability check than top-1% overlap
from scipy.stats import spearmanr
rank_corrs = {}
for seed in [1, 7, 123, 2024]:
    corr, _ = spearmanr(seed_scores[42], seed_scores[seed])
    rank_corrs[f"seed_{seed}_spearman_vs_primary"] = round(float(corr), 4)
report["seed_stability_spearman"] = rank_corrs

# ---------------------------------------------------------------------
# (4) Feature-set ablation: Work-Type-only vs State-only vs combined
# ---------------------------------------------------------------------
X_worktype = df[WORKTYPE_Z_FEATURES].fillna(0.0).values
X_state = df[STATE_Z_FEATURES].fillna(0.0).values
score_worktype = run_iforest(X_worktype, seed=42)
score_state = run_iforest(X_state, seed=42)

corr_wt_vs_combined, _ = spearmanr(score_worktype, df["iforest_score_primary"])
corr_state_vs_combined, _ = spearmanr(score_state, df["iforest_score_primary"])
corr_wt_vs_state, _ = spearmanr(score_worktype, score_state)
report["feature_ablation_agreement"] = {
    "worktype_only_vs_combined_spearman": round(float(corr_wt_vs_combined), 4),
    "state_only_vs_combined_spearman": round(float(corr_state_vs_combined), 4),
    "worktype_only_vs_state_only_spearman": round(float(corr_wt_vs_state), 4),
}

# ---------------------------------------------------------------------
# (5) Peer-group statistical baseline vs Isolation Forest agreement
# ---------------------------------------------------------------------
corr_stat_vs_iforest, _ = spearmanr(df["peer_stat_score"], df["iforest_score_primary"])
report["peer_stat_baseline_vs_iforest_spearman"] = round(float(corr_stat_vs_iforest), 4)

top_iforest = set(np.argsort(-df["iforest_score_primary"].values)[:n_top])
top_statbaseline = set(np.argsort(-df["peer_stat_score"].values)[:n_top])
report["top1pct_overlap_iforest_vs_statbaseline_pct"] = round(100 * len(top_iforest & top_statbaseline) / n_top, 2)

# ---------------------------------------------------------------------
# (6) Are top anomalies just "legitimate scale/complexity" (more payments,
# larger recommended amount)? Compare top-1% vs rest on raw (non-z-scored) attributes.
# ---------------------------------------------------------------------
df["is_top1pct_iforest"] = False
df.loc[df.index.isin(top_iforest), "is_top1pct_iforest"] = True
scale_check = df.groupby("is_top1pct_iforest").agg(
    n=("Work ID", "size"),
    median_recommended_amount=("Recommended Amount (INR)", "median"),
    median_n_payments=("n_payments", "median"),
    median_n_vendors=("n_vendors", "median"),
    median_span_days=("span_days", "median"),
).round(2)
report["top1pct_vs_rest_raw_scale_comparison"] = scale_check.to_dict(orient="index")

# ---------------------------------------------------------------------
# (7) Temporal stability: split by Sanction Date, does the top-1% overlap
# between an "early half" model and "late half" model, when both are scored
# on the SAME full population using ONLY that half's data to fit?
# ---------------------------------------------------------------------
df_sorted = df.sort_values("Sanction Date")
half = len(df_sorted) // 2
early, late = df_sorted.iloc[:half], df_sorted.iloc[half:]
model_early = IsolationForest(n_estimators=200, contamination="auto", random_state=42).fit(early[COMBINED_FEATURES].fillna(0.0).values)
model_late = IsolationForest(n_estimators=200, contamination="auto", random_state=42).fit(late[COMBINED_FEATURES].fillna(0.0).values)
score_from_early_model = -model_early.score_samples(df[COMBINED_FEATURES].fillna(0.0).values)
score_from_late_model = -model_late.score_samples(df[COMBINED_FEATURES].fillna(0.0).values)
corr_temporal, _ = spearmanr(score_from_early_model, score_from_late_model)
report["temporal_stability_spearman_early_vs_late_model"] = round(float(corr_temporal), 4)

print(json.dumps(report, indent=2, default=str))
(MODEL_DEV_DIR / "anomaly_robustness_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

df.to_csv(MODEL_DEV_DIR / "anomaly_scores_full.csv", index=False)

# Save the top 20 anomalies for qualitative inspection (Phase 9 -- face validity, not fraud-labeling)
top20 = df.nlargest(20, "iforest_score_primary")[["Work ID", "State", "Work Type", "Recommended Amount (INR)", "n_payments", "n_vendors", "max_payment_share", "span_days", "ratio_disbursed_to_recommended", "iforest_score_primary"]]
top20.to_csv(MODEL_DEV_DIR / "top20_anomalies_for_review.csv", index=False)
print("\n=== Top 20 anomaly-scored works (for qualitative review, NOT fraud claims) ===")
print(top20.to_string())
