"""Project-wide paths and thresholds for the MPLADS ML system."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Data"

WORKS_RECOMMENDED_CSV = DATA_DIR / "Works_Recommended_Cleaned.csv"
WORKS_COMPLETED_CSV = DATA_DIR / "Works_Completed_Cleaned.csv"
EXPENDITURE_CSV = DATA_DIR / "Expenditure_Cleaned.csv"
ALLOCATED_LIMIT_LOKSABHA_CSV = DATA_DIR / "Allocated_Limit_LokSabha_Cleaned.csv"
ALLOCATED_LIMIT_RAJYASABHA_CSV = DATA_DIR / "Allocated_Limit_RajyaSabha_Cleaned.csv"
AMOUNT_CONSENTED_CALAMITY_CSV = DATA_DIR / "Amount_Consented_Calamity_Cleaned.csv"

# --- Module 1: Cost Overrun Detection -----------------------------------

MODULE1_MODEL_DIR = PROJECT_ROOT / "models" / "module1_cost_overrun"
MODULE1_MODEL_FILE = MODULE1_MODEL_DIR / "cost_overrun_model.joblib"
MODULE1_METADATA_FILE = MODULE1_MODEL_DIR / "metadata.json"

# Column names as they literally appear in the source CSVs (do not rename/guess).
COL_WORK_ID = "Work ID"
COL_WORK_CATEGORY = "Work Category"
COL_WORK_TYPE = "Work Type"
COL_STATE = "State"
COL_IDA_DISTRICT = "IDA District"
COL_RECOMMENDED_DATE = "Recommended Date"
COL_RECOMMENDED_AMOUNT = "Recommended Amount (INR)"
COL_SANCTION_DATE = "Sanction Date"
COL_COMPLETION_DATE = "Completion Date"
COL_AMOUNT_DISBURSED = "Amount Disbursed (INR)"
COL_IDA_OFFICE = "IDA Office"
COL_EXPENDITURE_DATE = "Expenditure Date"
COL_FUND_DISBURSED = "Fund Disbursed Amount (INR)"

# Fail-fast target-validation thresholds. A "positive overrun example" is a
# historical record where actual final expenditure exceeded the sanctioned
# amount (overrun_pct > 0). Both an absolute floor and a ratio floor are
# required so a handful of examples in a huge dataset can't pass by volume
# alone, and a small dataset can't pass on a tiny absolute count alone.
MIN_OVERRUN_SAMPLES = 50
MIN_OVERRUN_RATIO = 0.01

# Risk banding applied to a model's predicted overrun_pct at inference time.
# overrun_pct <= RISK_THRESHOLD_MEDIUM        -> LOW
# RISK_THRESHOLD_MEDIUM < x <= RISK_THRESHOLD_HIGH -> MEDIUM
# overrun_pct > RISK_THRESHOLD_HIGH           -> HIGH
RISK_THRESHOLD_MEDIUM = 5.0
RISK_THRESHOLD_HIGH = 15.0

# --- Module 2: Delayed Project Prediction -------------------------------

MODULE2_MODEL_DIR = PROJECT_ROOT / "models" / "module2_delay_prediction"
MODULE2_MODEL_FILE = MODULE2_MODEL_DIR / "delay_model.joblib"
MODULE2_METADATA_FILE = MODULE2_MODEL_DIR / "metadata.json"

MODULE2_STATUS_LABELS = ["ON_TRACK", "AT_RISK", "LIKELY_DELAYED"]

# A project's actual Sanction->Completion duration is compared against the
# empirical duration distribution of its own peer group (see
# MODULE2_PEER_GROUP_COL) to assign a label. This is a data-driven, relative
# definition of delay -- there is no expected/target completion date column
# in the source data, so no absolute deadline-based label is possible
# without an externally supplied SLA (not currently available).
MODULE2_PEER_GROUP_COL = COL_WORK_TYPE
MODULE2_PEER_GROUP_FALLBACK_COL = COL_WORK_CATEGORY
MODULE2_MIN_PEER_GROUP_SIZE = 30
MODULE2_AT_RISK_PERCENTILE = 0.75
MODULE2_LIKELY_DELAYED_PERCENTILE = 0.90

# Fail-fast: every one of the 3 classes must have at least this many labeled
# examples after peer-relative labeling, or training refuses to proceed.
MODULE2_MIN_CLASS_SAMPLES = 50

# --- Module 7: Development Pulse Score -----------------------------------

MODULE7_STATE_DIR = PROJECT_ROOT / "state" / "module7_pulse_score"
MODULE7_HISTORY_FILE = MODULE7_STATE_DIR / "pulse_history.json"

# Relative weights of the 4 rollup dimensions in the final 0-100 pulse score.
# Must sum to 1.0.
MODULE7_DIMENSION_WEIGHTS = {
    "completion": 0.30,
    "utilization": 0.25,
    "delay": 0.25,
    "velocity": 0.20,
}

# Health-band thresholds on the final pulse score.
MODULE7_HEALTH_BAND_EXCELLENT = 85.0
MODULE7_HEALTH_BAND_GOOD = 70.0
MODULE7_HEALTH_BAND_NEEDS_ATTENTION = 50.0
# below MODULE7_HEALTH_BAND_NEEDS_ATTENTION -> CRITICAL

# Minimum national sample count for a Work Type's median duration to be used
# as the velocity benchmark directly; smaller groups fall back to the
# Work Category-level median (mirrors MODULE2_MIN_PEER_GROUP_SIZE's rationale
# but kept independent so retuning one module doesn't silently affect the
# other).
MODULE7_MIN_PEER_GROUP_SIZE = 30

# Benchmark/actual duration ratio is capped here before normalizing to a
# 0-100 velocity score, so one exceptionally fast project doesn't dominate.
MODULE7_VELOCITY_RATIO_CAP = 1.5

# When a constituency has zero completed works, actual duration is
# undefined -- velocity is scored as this neutral fallback rather than
# dividing by zero or fabricating a duration.
MODULE7_VELOCITY_NEUTRAL_SCORE = 50.0

# --- Module 8: Report Generation -----------------------------------------

MODULE8_STATE_DIR = PROJECT_ROOT / "state" / "module8_report_generation"
MODULE8_SUBSCRIPTION_RUNS_FILE = MODULE8_STATE_DIR / "subscription_runs.json"

# --- Module 9: Alert Generation & Field Inspector Dispatch ---------------

MODULE9_STATE_DIR = PROJECT_ROOT / "state" / "module9_dispatch"
MODULE9_AUDIT_TRAIL_FILE = MODULE9_STATE_DIR / "audit_trail.json"
MODULE9_INSPECTORS_STATE_FILE = MODULE9_STATE_DIR / "inspectors_state.json"

# Module 2's delay_probability alert threshold -- a project this likely to
# be delayed triggers an alert even if its predicted class isn't the
# LIKELY_DELAYED label itself.
MODULE9_DELAY_PROBABILITY_THRESHOLD = 0.75

# --- MPLADS Expenditure Utilization Risk ("underspend_risk") ------------
#
# NOT one of the 9 numbered modules -- a separate, isolated capability
# retained from the Option 2 MPLADS-specific ML target investigation after
# Option 3 (infrastructure cost-overrun) and Option 2A (expenditure anomaly)
# were both found not viable/not stronger. See:
#   OPTION2_MPLADS_TARGET_FEASIBILITY_REPORT.md
#   OPTION2A_MPLADS_EXPENDITURE_ANOMALY_REPORT.md
# for the full research record this implementation is required to match
# exactly (no new target, dataset, or feature choices were introduced here).

UNDERSPEND_RISK_MODEL_DIR = PROJECT_ROOT / "models" / "underspend_risk"
UNDERSPEND_RISK_MODEL_FILE = UNDERSPEND_RISK_MODEL_DIR / "underspend_risk_model.joblib"
UNDERSPEND_RISK_RATE_LOOKUP_FILE = UNDERSPEND_RISK_MODEL_DIR / "historical_rate_lookup.json"
UNDERSPEND_RISK_METADATA_FILE = UNDERSPEND_RISK_MODEL_DIR / "metadata.json"

# Exact target definition validated in OPTION2_MPLADS_TARGET_FEASIBILITY_REPORT.md
# section 7. The 0.999 cutoff is not a tuned percentage -- it separates the
# verified exact-match point mass (91.8% of completed works disburse exactly
# the recommended amount) from any departure from it at all.
UNDERSPEND_TARGET_DEFINITION = (
    "has_underspend = 1 if Amount Disbursed (INR) < 0.999 * Recommended Amount (INR), "
    "else 0. Defined only for completed works. Research base rate: 5.71% "
    "(854 / 14,965 completed works)."
)

UNDERSPEND_RISK_DISCLAIMER = (
    "This is a limited statistical signal derived from historical MPLADS "
    "expenditure data. It is NOT cost-overrun detection, NOT fraud detection, "
    "NOT corruption or misconduct detection, and NOT a definitive assessment "
    "of project risk. Feature-ablation testing (see "
    "OPTION2_MPLADS_TARGET_FEASIBILITY_REPORT.md section 12) found the "
    "underlying signal comes almost entirely from historical State and Work "
    "Type group rates rather than this specific project's own "
    "characteristics -- it does not reliably differentiate risk WITHIN a "
    "State/Work-Type category. Precision at the decision threshold below was "
    "approximately 16-17% in backtesting: most works flagged will NOT end up "
    "underspending. This score does not predict or prove future behavior."
)

# The only threshold ever evaluated in the validated research (used to
# compute the reported precision/recall/F1/confusion-matrix in
# OPTION2_MPLADS_TARGET_FEASIBILITY_REPORT.md section 11) -- scikit-learn's
# own default classification threshold. No additional tiers (e.g. a 3-way
# LOW/MEDIUM/HIGH split) were validated, so none are invented here.
UNDERSPEND_RISK_DECISION_THRESHOLD = 0.5
