"""Stage 3-5: build Dataset A (completed projects) and Dataset B (ongoing
projects) from the extracted per-report JSON, with full quality and target
validation statistics. Writes clean CSVs plus a JSON quality/validation
report. Does not train anything -- pure dataset construction per the
approved plan.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

EXTRACTED_DIR = Path(__file__).resolve().parents[2] / "data" / "module1_infra_source" / "extracted"
DATASET_DIR = Path(__file__).resolve().parents[2] / "data" / "module1_infra_source" / "datasets"

# The entire IPMD/OCMS Flash Report system, by the Ministry's own published
# mandate (verified in Phase 1's research: "Monitoring of projects of Rs.150
# crores and above"), only covers projects costing >= Rs. 150 crore. A parsed
# "original cost" below this is therefore not a statistical outlier to be
# judgment-called -- it is definitionally impossible given the source
# system's own scope, and reveals a field-misalignment parsing error on that
# row (confirmed by manual inspection: such rows also have corrupted
# project_name fields containing dates/numbers instead of text).
MIN_VALID_ORIGINAL_COST_CR = 150.0


def _load_all(suffix: str) -> pd.DataFrame:
    frames = []
    for path in sorted(EXTRACTED_DIR.glob(f"*_{suffix}.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        if records:
            frames.append(pd.DataFrame(records))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_dataset_a() -> tuple[pd.DataFrame, dict]:
    """Dataset A: completed projects. One row per (report, project) as
    extracted; then collapsed to one row per project using the LAST
    (most recent report) observed cumulative expenditure, per Stage 3's
    explicit instruction not to use the first-seen value."""
    raw = _load_all("completed")
    quality: dict = {"raw_records": len(raw)}
    if raw.empty:
        return raw, quality

    quality["unique_project_ids_raw"] = raw["project_id"].nunique()
    quality["duplicate_project_id_rows"] = int(raw["project_id"].duplicated().sum())
    quality["missing_original_cost"] = int(raw["original_cost_cr"].isna().sum())
    quality["missing_cumulative_expenditure"] = int(raw["cumulative_expenditure_cr"].isna().sum())
    quality["missing_commissioning_date"] = int(raw["original_commissioning_date"].isna().sum())
    quality["negative_original_cost"] = int((raw["original_cost_cr"] < 0).sum())
    quality["zero_original_cost"] = int((raw["original_cost_cr"] == 0).sum())
    quality["negative_expenditure"] = int((raw["cumulative_expenditure_cr"] < 0).sum())

    # Report order matters for "last observed" -- report_id sorts
    # chronologically because of the FR_YYYY_MM naming convention used by
    # the download/extraction scripts.
    raw_sorted = raw.sort_values("report_id")

    # Detect projects whose original_cost_cr DISAGREES across reports --
    # this field should be invariant (it's fixed at approval time), so a
    # mismatch indicates either a genuine cost revision or (more likely
    # given the ~1% mismatch rate found during Stage 2 validation) a
    # parsing misalignment. Flagged, not silently trusted.
    cost_by_project = raw_sorted.groupby("project_id")["original_cost_cr"].nunique(dropna=True)
    inconsistent_ids = cost_by_project[cost_by_project > 1].index.tolist()
    quality["projects_with_conflicting_original_cost"] = len(inconsistent_ids)

    # Collapse to one row per project: keep the LAST report's record
    # (most recent observed cumulative expenditure), excluding projects
    # with conflicting original cost (can't trust which value is right).
    clean = raw_sorted[~raw_sorted["project_id"].isin(inconsistent_ids)]
    deduped = clean.drop_duplicates(subset="project_id", keep="last").copy()
    quality["projects_excluded_conflicting_cost"] = len(inconsistent_ids)
    quality["unique_projects_after_dedup"] = len(deduped)

    # Target validation (Stage 5).
    below_threshold = deduped[
        deduped["original_cost_cr"].notna() & (deduped["original_cost_cr"] < MIN_VALID_ORIGINAL_COST_CR)
    ]
    quality["excluded_below_150cr_threshold_parser_error"] = len(below_threshold)

    valid = deduped[
        deduped["original_cost_cr"].notna()
        & (deduped["original_cost_cr"] >= MIN_VALID_ORIGINAL_COST_CR)
        & deduped["cumulative_expenditure_cr"].notna()
        & (deduped["cumulative_expenditure_cr"] >= 0)
    ].copy()
    # NOT "final_cost_overrun_pct": Stage-1 target validation found no
    # explicit statement in the source report that "Cumulative Expenditure"
    # for a project marked Completed represents a financially closed-out
    # final cost -- "Completed" is only one of three offload categories
    # (Completed/Dropped/Frozen) describing commissioning status, not a
    # financial-closure event. Empirical check across report pairs up to 3
    # months apart shows 0/87 comparable projects with any expenditure
    # drift post-completion (strong short-term stability), but no evidence
    # exists over longer horizons (retention payments, warranty claims,
    # litigation settlements could still revise this figure years later).
    # Named accordingly per the approved fallback.
    valid["observed_expenditure_overrun_pct"] = (
        (valid["cumulative_expenditure_cr"] - valid["original_cost_cr"]) / valid["original_cost_cr"] * 100.0
    )

    quality["valid_labeled_projects"] = len(valid)
    quality["positive_overrun_count"] = int((valid["observed_expenditure_overrun_pct"] > 0).sum())
    quality["positive_overrun_pct_of_valid"] = (
        round(100.0 * quality["positive_overrun_count"] / len(valid), 2) if len(valid) else 0.0
    )
    if len(valid):
        quality["overrun_pct_min"] = round(float(valid["observed_expenditure_overrun_pct"].min()), 2)
        quality["overrun_pct_max"] = round(float(valid["observed_expenditure_overrun_pct"].max()), 2)
        quality["overrun_pct_median"] = round(float(valid["observed_expenditure_overrun_pct"].median()), 2)
        quality["overrun_pct_mean"] = round(float(valid["observed_expenditure_overrun_pct"].mean()), 2)

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    valid.to_csv(DATASET_DIR / "dataset_a_completed_projects.csv", index=False)
    return valid, quality


def build_dataset_b() -> tuple[pd.DataFrame, dict]:
    """Dataset B: ongoing projects (anticipated-cost overrun). Collapsed to
    the LATEST report's snapshot per project (most current anticipated
    cost), matching the same "last observed" principle as Dataset A."""
    raw = _load_all("ongoing")
    quality: dict = {"raw_records": len(raw)}
    if raw.empty:
        return raw, quality

    quality["unique_project_ids_raw"] = raw["project_id"].nunique()

    raw_sorted = raw.sort_values("report_id")
    deduped = raw_sorted.drop_duplicates(subset="project_id", keep="last").copy()
    quality["unique_projects_after_dedup"] = len(deduped)

    # Explicit rejection funnel: every deduped row gets exactly one
    # disposition (a specific rejection reason, or "valid"), evaluated in a
    # fixed priority order so the funnel counts are mutually exclusive and
    # sum to unique_projects_after_dedup -- this is the raw -> parsed ->
    # rejected(reason) accounting requested, not a set of overlapping flags.
    def _disposition(row) -> str:
        # A real project_name in this table is prose ("CGD PROJECT AT REWA
        # DISTRICT"); 3+ digits inside it is a strong, specific signal that a
        # blank Revised field shifted the bracket-block parse, contaminating
        # the name with numbers/dates that belong to other columns.
        name = row["project_name"] or ""
        if sum(ch.isdigit() for ch in name) >= 3:
            return "rejected_name_field_misalignment"
        if pd.isna(row["cost_original_cr"]):
            return "rejected_missing_original_cost"
        if row["cost_original_cr"] < MIN_VALID_ORIGINAL_COST_CR:
            return "rejected_below_150cr_threshold"
        if pd.isna(row["cost_anticipated_cr"]):
            return "rejected_missing_anticipated_cost"
        if row["cost_anticipated_cr"] < 0:
            return "rejected_negative_anticipated_cost"
        return "valid"

    deduped["disposition"] = deduped.apply(_disposition, axis=1)
    funnel = deduped["disposition"].value_counts().to_dict()
    quality["rejection_funnel"] = {
        "raw_rows": len(raw),
        "unique_projects_after_dedup": len(deduped),
        **{k: int(v) for k, v in funnel.items()},
    }

    valid = deduped[deduped["disposition"] == "valid"].copy()
    # NOT "final anticipated cost" either: this is the project's own
    # currently self-reported forecast (Category C per the Phase-3
    # classification), never the realized outcome -- kept as a strictly
    # separate dataset/target from Dataset A per your instruction not to mix
    # them.
    valid["anticipated_cost_overrun_pct"] = (
        (valid["cost_anticipated_cr"] - valid["cost_original_cr"]) / valid["cost_original_cr"] * 100.0
    )

    quality["valid_labeled_projects"] = len(valid)
    quality["positive_overrun_count"] = int((valid["anticipated_cost_overrun_pct"] > 0).sum())
    quality["positive_overrun_pct_of_valid"] = (
        round(100.0 * quality["positive_overrun_count"] / len(valid), 2) if len(valid) else 0.0
    )
    if len(valid):
        quality["overrun_pct_min"] = round(float(valid["anticipated_cost_overrun_pct"].min()), 2)
        quality["overrun_pct_max"] = round(float(valid["anticipated_cost_overrun_pct"].max()), 2)
        quality["overrun_pct_median"] = round(float(valid["anticipated_cost_overrun_pct"].median()), 2)
        quality["overrun_pct_mean"] = round(float(valid["anticipated_cost_overrun_pct"].mean()), 2)

    valid = valid.drop(columns=["disposition"])
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    valid.to_csv(DATASET_DIR / "dataset_b_ongoing_projects.csv", index=False)
    return valid, quality


def main() -> None:
    a_df, a_quality = build_dataset_a()
    b_df, b_quality = build_dataset_b()

    report = {"dataset_a_completed": a_quality, "dataset_b_ongoing": b_quality}
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / "quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== Dataset A: Completed Projects ===")
    print(json.dumps(a_quality, indent=2))
    print()
    print("=== Dataset B: Ongoing Projects ===")
    print(json.dumps(b_quality, indent=2))


if __name__ == "__main__":
    main()
