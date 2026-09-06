import pandas as pd

from mplads.config import MODULE2_STATUS_LABELS
from mplads.module2_delay_prediction.data_validation import load_joined_training_frame
from mplads.module2_delay_prediction.labeling import assign_status_labels


def test_assign_status_labels_on_real_data_produces_three_classes():
    frame = load_joined_training_frame()
    labeled = assign_status_labels(frame)

    assert set(labeled["status"].unique()) <= set(MODULE2_STATUS_LABELS)
    counts = labeled["status"].value_counts()
    # Peer-relative percentile labeling guarantees a real minority class.
    assert counts.get("AT_RISK", 0) > 0
    assert counts.get("LIKELY_DELAYED", 0) > 0
    assert counts.get("ON_TRACK", 0) > counts.get("LIKELY_DELAYED", 0)


def test_assign_status_labels_falls_back_for_small_peer_groups():
    """A Work Type with only a handful of rows should fall back to the
    Work Category-level thresholds rather than compute a degenerate
    percentile from too few samples."""
    n_common = 100
    df = pd.DataFrame(
        {
            "Work Type": ["CommonType"] * n_common + ["RareType"] * 3,
            "Work Category": ["CategoryA"] * (n_common + 3),
            "days_sanction_to_completion": list(range(1, n_common + 1)) + [50, 60, 70],
        }
    )
    labeled = assign_status_labels(df)
    rare_rows = labeled[labeled["Work Type"] == "RareType"]
    assert (rare_rows["peer_group_used"] == "Work Category").all()


def test_assign_status_labels_monotonic_with_duration():
    """Within one peer group, a longer duration should never map to a
    'less delayed' status than a shorter one."""
    df = pd.DataFrame(
        {
            "Work Type": ["T"] * 40,
            "Work Category": ["C"] * 40,
            "days_sanction_to_completion": list(range(1, 41)),
        }
    )
    labeled = assign_status_labels(df).sort_values("days_sanction_to_completion")
    rank = {"ON_TRACK": 0, "AT_RISK": 1, "LIKELY_DELAYED": 2}
    ranks = labeled["status"].map(rank).tolist()
    assert ranks == sorted(ranks)
