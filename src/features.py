from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


ID_COLS = ["repository", "project", "issue_key"]


def _clean_category(s: pd.Series) -> pd.Series:
    s = s.astype("string").fillna("Unknown")
    s = s.replace({"<NA>": "Unknown", "None": "Unknown", "nan": "Unknown", "": "Unknown"})
    return s


def feature_spec(mode: str = "day0", strict_no_leakage: bool = True, landmark_days: int = 7):
    numeric = [
        "created_weekday",
        "created_hour",
        "initial_assigned",
    ]
    categorical = [
        "initial_priority",
        "initial_issue_type",
    ]

    if not strict_no_leakage:
        # Chỉ dùng cho sensitivity analysis: các field snapshot có thể thay đổi
        # sau creation và có nguy cơ leakage.
        numeric += [
            "snapshot_summary_len",
            "snapshot_description_len",
            "snapshot_label_count",
            "snapshot_component_count",
        ]

    if mode == "day7":
        numeric += [
            f"changes_{landmark_days}d",
            f"status_changes_{landmark_days}d",
            f"assignee_changes_{landmark_days}d",
            f"priority_changes_{landmark_days}d",
            f"comments_{landmark_days}d",
            f"inactivity_days_at_{landmark_days}d",
        ]
    elif mode != "day0":
        raise ValueError("mode phải là 'day0' hoặc 'day7'")

    return numeric, categorical


def build_day0_table(
    cohort: pd.DataFrame,
    strict_no_leakage: bool = True,
) -> pd.DataFrame:
    df = cohort.copy()

    for c in ["initial_priority", "initial_issue_type"]:
        df[c] = _clean_category(df[c])

    numeric, categorical = feature_spec(
        mode="day0",
        strict_no_leakage=strict_no_leakage,
    )

    keep = ID_COLS + ["duration_days", "event"] + numeric + categorical
    keep = [c for c in keep if c in df.columns]

    out = df[keep].copy()
    out["event"] = out["event"].astype(bool)
    out["duration_days"] = pd.to_numeric(out["duration_days"], errors="coerce")
    out = out[out["duration_days"].notna() & (out["duration_days"] > 0)].copy()

    return out


def build_day7_table(
    cohort: pd.DataFrame,
    landmark_days: int = 7,
    strict_no_leakage: bool = True,
) -> pd.DataFrame:
    """
    Landmark dataset:
    chỉ issue còn tồn tại tại ngày landmark mới được giữ lại.
    Target mới = thời gian còn lại sau landmark.
    """
    df = cohort[cohort["duration_days"] > landmark_days].copy()
    df["duration_days"] = df["duration_days"] - float(landmark_days)

    for c in ["initial_priority", "initial_issue_type"]:
        df[c] = _clean_category(df[c])

    numeric, categorical = feature_spec(
        mode="day7",
        strict_no_leakage=strict_no_leakage,
        landmark_days=landmark_days,
    )

    keep = ID_COLS + ["duration_days", "event"] + numeric + categorical
    keep = [c for c in keep if c in df.columns]

    out = df[keep].copy()
    out["event"] = out["event"].astype(bool)
    out["duration_days"] = pd.to_numeric(out["duration_days"], errors="coerce")
    out = out[out["duration_days"].notna() & (out["duration_days"] > 0)].copy()

    return out


def sample_training_rows(
    df: pd.DataFrame,
    max_rows: int | None,
    random_state: int = 42,
) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.copy()

    # Lấy mẫu gần cân bằng theo project để project khổng lồ không nuốt hết train set.
    groups = list(df.groupby("project", sort=False))
    per_group = max(1, max_rows // max(1, len(groups)))
    chunks = []

    for _, g in groups:
        n = min(len(g), per_group)
        chunks.append(g.sample(n=n, random_state=random_state))

    out = pd.concat(chunks, ignore_index=True)

    if len(out) < max_rows:
        remaining = df.loc[~df.index.isin(out.index)]
        if len(remaining):
            extra = remaining.sample(
                n=min(max_rows - len(out), len(remaining)),
                random_state=random_state,
            )
            out = pd.concat([out, extra], ignore_index=True)

    return out.head(max_rows).copy()
