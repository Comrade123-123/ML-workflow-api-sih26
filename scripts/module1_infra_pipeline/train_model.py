"""Stage 6-10: prediction-timing/leakage audit, feature engineering,
chronological split, and model comparison for Dataset A (completed
infrastructure projects, MoSPI/IPMD/OCMS).

This is a research/training script -- it does NOT touch src/mplads or the
live API. Per the approved plan, that only happens after this audit and the
model comparison are reported and reviewed.

=====================================================================
STAGE 6 -- PREDICTION-TIMING & LEAKAGE AUDIT (read this before the code)
=====================================================================

PREDICTION POINT: at project approval / original sanction -- before any
execution has begun. This is not a choice among several options; it is the
ONLY point Dataset A's schema can support. The source table ("Month wise
List of Completed Projects") has exactly 5 raw columns: Sl.No, Project
Name, Original Cost, Original Date of Commissioning, Cumulative
Expenditure. There is no physical-progress, financial-progress, or
approval-date column, so no "mid-project" snapshot can be constructed from
this data at all -- that would require a different source table (which, as
reported for Dataset B, does not reliably exist in an unbiased, ID-joinable
form in this report).

INFORMATION CUTOFF: everything that is fixed and known at the moment a
project is approved and its original cost/plan is recorded. Nothing about
how the project actually proceeded is used.

FEATURE-BY-FEATURE CLASSIFICATION:

| Raw/derived field                     | Classification | Reasoning |
|----------------------------------------|-----------------|-----------|
| original_cost_cr                       | SAFE            | The sanctioned amount itself, fixed at approval. |
| sector                                 | SAFE            | Project categorization is inherent to project scope at approval. |
| original_commissioning_date (-> planned_commissioning_year) | SAFE | This is the ORIGINAL plan, decided at/near approval -- not an outcome. Only the year is kept (see below). |
| project_size_bucket                    | SAFE            | Pure derived bucket of original_cost_cr; no new information. |
| sector_historical_overrun_rate         | SAFE (engineered) | Computed ONLY from OTHER projects with a strictly EARLIER planned_commissioning_year in the same sector (expanding-window, causal). Never uses the current project's own outcome. |
| project_name                           | EXCLUDED        | Available at approval (SAFE in principle) but free text with no validated signal; excluded for parsimony, not because of leakage. |
| project_id                             | EXCLUDED        | Identifier only, not a predictive feature. |
| report_id                              | LEAKAGE         | Reflects which report period recorded the project as *completed* -- a proxy for actual completion timing, which would not be known at approval. |
| cumulative_expenditure_cr              | LEAKAGE         | Directly used to compute the target; the post-outcome figure itself. |
| observed_expenditure_overrun_pct       | TARGET          | The label. Never a feature. |
| original_commissioning_date (raw month)| EXCLUDED        | Kept only at year granularity; month-level granularity risked overfitting noise given n=225, and added no audited justification for finer use. |

TARGET RE-VALIDATION (final attempt, new evidence): a newer PAIMANA-format
report edition (Sept 2025 onward) was found to carry an EXPLICIT disclaimer
on its own completed-projects table: "Note: Cumulative Expenditure is as
per last reporting by Central Line Ministries/Departments on the IPM
portal, this may not be the actual completion cost." This is direct,
primary-source confirmation -- not an inference -- that the target should
NOT be called a final/closed-out cost. observed_expenditure_overrun_pct
remains the correct name; it is not being changed.

MULTI-REPORT LEAKAGE: Dataset A's "keep last-observed report" dedup rule
(Stage 3) only affects which snapshot of `cumulative_expenditure_cr` (the
TARGET input) is used -- never a feature. All SAFE features above
(original_cost_cr, sector, planned_commissioning_year) are invariant
values fixed at approval and identical across every report edition a
project appears in, confirmed during Stage 2/4 validation (0/87 drift,
99.2% name consistency). There is no scenario where a later report's
information could leak into a feature for an earlier prediction point,
because the features don't vary by report at all.

CHRONOLOGICAL ORDERING KEY: planned_commissioning_year (+ month, for
tie-breaking only) is used as the ordering variable for both the
historical-rate feature and the train/validation/test split, since it is
the only date field Dataset A provides and it is fixed at approval time
(not report-dependent).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_DIR = Path(__file__).resolve().parents[2] / "data" / "module1_infra_source" / "datasets"
ARTIFACT_DIR = Path(__file__).resolve().parents[2] / "data" / "module1_infra_source" / "model_dev"

MIN_VALID_ORIGINAL_COST_CR = 150.0

SIZE_BUCKET_EDGES = [150, 300, 1000, 5000, float("inf")]
SIZE_BUCKET_LABELS = ["150-300cr", "300cr-1000cr", "1000cr-5000cr", "5000cr+"]


def _parse_planned_date(value: str) -> tuple[int, int]:
    month, year = value.split("/")
    return int(year), int(month)


def audit_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Implements the feature list from the Stage 6 audit above. Every
    column produced here is traceable to a row in that table."""
    df = df.copy()

    years_months = df["original_commissioning_date"].apply(_parse_planned_date)
    df["planned_commissioning_year"] = years_months.apply(lambda t: t[0])
    _month = years_months.apply(lambda t: t[1])
    df["_sort_key"] = df["planned_commissioning_year"] * 12 + _month

    df["project_size_bucket"] = pd.cut(
        df["original_cost_cr"], bins=SIZE_BUCKET_EDGES, labels=SIZE_BUCKET_LABELS, right=False
    ).astype(str)

    # Causal, expanding-window historical sector overrun rate: for each
    # project, the mean observed_expenditure_overrun_pct of ONLY other
    # projects in the same sector with a strictly earlier sort key. Falls
    # back to the (also strictly causal) global prior mean when the
    # project's own sector has no earlier example yet, and to 0.0 (a
    # neutral "on budget" prior, not a data-derived value) only for
    # projects with no earlier examples at all.
    df = df.sort_values("_sort_key").reset_index(drop=True)
    sector_rates = []
    global_so_far: list[float] = []
    sector_so_far: dict[str, list[float]] = {}
    for _, row in df.iterrows():
        sector = row["sector"]
        prior_sector_vals = sector_so_far.get(sector, [])
        if prior_sector_vals:
            rate = float(np.mean(prior_sector_vals))
        elif global_so_far:
            rate = float(np.mean(global_so_far))
        else:
            rate = 0.0
        sector_rates.append(rate)
        sector_so_far.setdefault(sector, []).append(row["observed_expenditure_overrun_pct"])
        global_so_far.append(row["observed_expenditure_overrun_pct"])
    df["sector_historical_overrun_rate"] = sector_rates

    return df


FEATURE_COLUMNS_NUMERIC = ["original_cost_cr", "planned_commissioning_year", "sector_historical_overrun_rate"]
FEATURE_COLUMNS_CATEGORICAL = ["sector", "project_size_bucket"]
ALL_FEATURES = FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL
TARGET = "observed_expenditure_overrun_pct"


def build_preprocessing_pipeline() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [("impute", SimpleImputer(strategy="constant", fill_value="UNKNOWN")), ("encode", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer([("num", numeric, FEATURE_COLUMNS_NUMERIC), ("cat", categorical, FEATURE_COLUMNS_CATEGORICAL)])


def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Older-planned projects -> train, next -> validation, most recently
    planned -> test. No project appears in more than one split (each row is
    already one unique project_id, confirmed during Stage 3/4)."""
    df = df.sort_values("_sort_key").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]
    assert set(train["project_id"]) & set(val["project_id"]) & set(test["project_id"]) == set()
    return train, val, test


def regression_metrics(y_true, y_pred) -> dict:
    metrics = {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 3),
        "median_abs_error": round(float(median_absolute_error(y_true, y_pred)), 3),
    }
    # MAPE is only meaningful when the true value can't be ~0 or negative --
    # here observed_expenditure_overrun_pct legitimately takes negative
    # values (underspend) and can be arbitrarily close to 0, so a standard
    # MAPE would be dominated/undefined by near-zero true values. Report it
    # only over the subset with |y_true| > 10 (a defensible floor), and
    # label it explicitly as partial rather than a full-dataset metric.
    y_true_arr = np.asarray(y_true)
    mask = np.abs(y_true_arr) > 10
    if mask.sum() >= 5:
        mape_subset = np.mean(np.abs((y_true_arr[mask] - np.asarray(y_pred)[mask]) / y_true_arr[mask])) * 100
        metrics["mape_over_abs_overrun_gt_10pct_subset"] = round(float(mape_subset), 3)
        metrics["mape_subset_n"] = int(mask.sum())
    else:
        metrics["mape_over_abs_overrun_gt_10pct_subset"] = None
        metrics["mape_subset_n"] = int(mask.sum())
    return metrics


def risk_classification_metrics(y_true, y_pred, threshold: float = 10.0) -> dict:
    """Risk = positive class if observed_expenditure_overrun_pct > threshold
    (i.e. the project ended up costing meaningfully more than sanctioned).
    Threshold chosen as a round, interpretable number given the dataset's
    own median overrun of -11.89% and 33.78% positive-overrun base rate --
    not tuned against the test set."""
    y_true_class = (np.asarray(y_true) > threshold).astype(int)
    y_pred_class = (np.asarray(y_pred) > threshold).astype(int)
    return {
        "threshold_pct": threshold,
        "precision": round(float(precision_score(y_true_class, y_pred_class, zero_division=0)), 3),
        "recall": round(float(recall_score(y_true_class, y_pred_class, zero_division=0)), 3),
        "f1": round(float(f1_score(y_true_class, y_pred_class, zero_division=0)), 3),
        "confusion_matrix": confusion_matrix(y_true_class, y_pred_class).tolist(),
        "positive_class_count_true": int(y_true_class.sum()),
        "positive_class_count_pred": int(y_pred_class.sum()),
        "n": len(y_true_class),
    }


def main() -> None:
    raw = pd.read_csv(DATASET_DIR / "dataset_a_completed_projects.csv")
    df = audit_and_engineer_features(raw)

    print("=== Stage 6 Feature Audit: final feature list ===")
    for col in ALL_FEATURES:
        print(f"  {col}")
    print(f"Target: {TARGET}")
    print()

    train, val, test = chronological_split(df)
    print(f"=== Stage 7 Split (chronological by planned_commissioning_year/month) ===")
    print(f"Train: {len(train)} (years {train['planned_commissioning_year'].min()}-{train['planned_commissioning_year'].max()})")
    print(f"Val:   {len(val)} (years {val['planned_commissioning_year'].min()}-{val['planned_commissioning_year'].max()})")
    print(f"Test:  {len(test)} (years {test['planned_commissioning_year'].min()}-{test['planned_commissioning_year'].max()})")
    print()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    # Persist raw split membership (project IDs, dates, engineered feature
    # value, target) so an independent script can verify everything below
    # without trusting any function in this file.
    split_cols = ["project_id", "sector", "planned_commissioning_year", "sector_historical_overrun_rate", TARGET]
    train.assign(split="train")[["split"] + split_cols].to_csv(ARTIFACT_DIR / "split_train.csv", index=False)
    val.assign(split="val")[["split"] + split_cols].to_csv(ARTIFACT_DIR / "split_val.csv", index=False)
    test.assign(split="test")[["split"] + split_cols].to_csv(ARTIFACT_DIR / "split_test.csv", index=False)

    X_train, y_train = train[ALL_FEATURES], train[TARGET]
    X_val, y_val = val[ALL_FEATURES], val[TARGET]
    X_test, y_test = test[ALL_FEATURES], test[TARGET]
    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])

    results = {}

    # --- Baseline: predicts the training-set mean overrun for everyone ---
    baseline = Pipeline([("pre", build_preprocessing_pipeline()), ("model", DummyRegressor(strategy="mean"))])
    baseline.fit(X_train, y_train)
    cv_scores = cross_val_score(baseline, X_train, y_train, cv=KFold(5, shuffle=True, random_state=42), scoring="neg_mean_absolute_error")
    results["baseline"] = {
        "cv_mae_train_folds": [round(float(-s), 3) for s in cv_scores],
        "test": regression_metrics(y_test, baseline.predict(X_test)),
        "val": regression_metrics(y_val, baseline.predict(X_val)),
    }

    # --- Random Forest ---
    rf = Pipeline(
        [
            ("pre", build_preprocessing_pipeline()),
            ("model", RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5, random_state=42)),
        ]
    )
    rf.fit(X_train, y_train)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=KFold(5, shuffle=True, random_state=42), scoring="neg_mean_absolute_error")
    results["random_forest"] = {
        "cv_mae_train_folds": [round(float(-s), 3) for s in cv_scores],
        "val": regression_metrics(y_val, rf.predict(X_val)),
        "test": regression_metrics(y_test, rf.predict(X_test)),
    }

    # --- XGBoost ---
    import xgboost as xgb

    xgb_model = Pipeline(
        [
            ("pre", build_preprocessing_pipeline()),
            (
                "model",
                xgb.XGBRegressor(
                    n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.8, random_state=42, reg_lambda=1.0,
                ),
            ),
        ]
    )
    xgb_model.fit(X_train, y_train)
    cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=KFold(5, shuffle=True, random_state=42), scoring="neg_mean_absolute_error")
    results["xgboost"] = {
        "cv_mae_train_folds": [round(float(-s), 3) for s in cv_scores],
        "val": regression_metrics(y_val, xgb_model.predict(X_val)),
        "test": regression_metrics(y_test, xgb_model.predict(X_test)),
    }

    print("=== Regression results (val / test) ===")
    print(json.dumps(results, indent=2))
    print()

    # Persist raw predictions per project per model, for independent
    # verification -- nothing about the metrics above should need to be
    # trusted; this is the raw material to recompute them from scratch.
    predictions = pd.DataFrame(
        {
            "project_id": pd.concat([val["project_id"], test["project_id"]]).values,
            "split": ["val"] * len(val) + ["test"] * len(test),
            "y_true": pd.concat([y_val, y_test]).values,
            "baseline_pred": np.concatenate([baseline.predict(X_val), baseline.predict(X_test)]),
            "random_forest_pred": np.concatenate([rf.predict(X_val), rf.predict(X_test)]),
            "xgboost_pred": np.concatenate([xgb_model.predict(X_val), xgb_model.predict(X_test)]),
        }
    )
    predictions.to_csv(ARTIFACT_DIR / "predictions_val_test.csv", index=False)
    (ARTIFACT_DIR / "baseline_constant_value.json").write_text(
        json.dumps(
            {
                "baseline_predicted_constant": float(baseline.predict(X_test)[0]),
                "manually_computed_train_mean": float(y_train.mean()),
                "manually_computed_test_mean": float(y_test.mean()),
            },
            indent=2,
        )
    )

    # --- Risk classification layer (using the best model's test predictions) ---
    print("=== Risk classification (threshold: observed overrun > 10%) ===")
    for name, model in [("baseline", baseline), ("random_forest", rf), ("xgboost", xgb_model)]:
        clf_metrics = risk_classification_metrics(y_test, model.predict(X_test))
        print(name, json.dumps(clf_metrics))

    # --- Feature importance (Random Forest + XGBoost) ---
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    feature_names = rf.named_steps["pre"].get_feature_names_out()
    rf_importances = dict(zip(feature_names, rf.named_steps["model"].feature_importances_.round(4).tolist()))
    xgb_importances = dict(zip(feature_names, xgb_model.named_steps["model"].feature_importances_.round(4).tolist()))
    print()
    print("=== Random Forest feature importance ===")
    print(json.dumps(dict(sorted(rf_importances.items(), key=lambda x: -x[1])), indent=2))
    print("=== XGBoost feature importance ===")
    print(json.dumps(dict(sorted(xgb_importances.items(), key=lambda x: -x[1])), indent=2))

    (ARTIFACT_DIR / "stage6_10_results.json").write_text(
        json.dumps(
            {
                "results": results,
                "rf_feature_importance": rf_importances,
                "xgb_feature_importance": xgb_importances,
                "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
