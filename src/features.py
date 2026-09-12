from __future__ import annotations

import numpy as np
import pandas as pd


ID_COLS = ["repository", "project", "issue_key"]


def _clean_category(s: pd.Series) -> pd.Series:
    s = s.astype("string").fillna("Unknown")
    s = s.str.strip()
    s = s.replace({
        "<NA>": "Unknown",
        "None": "Unknown",
        "nan": "Unknown",
        "": "Unknown",
    })
    return s


def _normalize_priority(s: pd.Series) -> pd.Series:
    """Chỉ chuẩn hóa các biến thể lexical rõ ràng, ví dụ Blocker - P1 -> Blocker."""
    out = _clean_category(s)
    out = out.str.replace(r"\s*-\s*P[1-5]\s*$", "", regex=True)
    out = out.str.strip().replace({"": "Unknown"})
    return out


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
    normalize_priority_labels: bool = True,
) -> pd.DataFrame:
    df = cohort.copy()

    if normalize_priority_labels:
        df["initial_priority"] = _normalize_priority(df["initial_priority"])
    else:
        df["initial_priority"] = _clean_category(df["initial_priority"])

    df["initial_issue_type"] = _clean_category(df["initial_issue_type"])

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
    normalize_priority_labels: bool = True,
) -> pd.DataFrame:
    df = cohort[cohort["duration_days"] > float(landmark_days)].copy()
    df["duration_days"] = df["duration_days"] - float(landmark_days)

    if normalize_priority_labels:
        df["initial_priority"] = _normalize_priority(df["initial_priority"])
    else:
        df["initial_priority"] = _clean_category(df["initial_priority"])

    df["initial_issue_type"] = _clean_category(df["initial_issue_type"])

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
    """Lấy mẫu gần cân bằng theo repository+project, không làm sai index gốc."""
    if max_rows is None or len(df) <= int(max_rows):
        return df.copy()

    max_rows = int(max_rows)
    work = df.copy()
    work["__row_id__"] = np.arange(len(work), dtype=np.int64)

    group_cols = ["repository", "project"] if {"repository", "project"}.issubset(work.columns) else ["project"]
    grouped = list(work.groupby(group_cols, sort=False, dropna=False))
    per_group = max(1, max_rows // max(1, len(grouped)))

    chunks = []
    selected_ids = set()

    for i, (_, g) in enumerate(grouped):
        n = min(len(g), per_group)
        sampled = g.sample(n=n, random_state=random_state + i)
        chunks.append(sampled)
        selected_ids.update(sampled["__row_id__"].tolist())

    out = pd.concat(chunks, ignore_index=True)

    if len(out) < max_rows:
        remaining = work[~work["__row_id__"].isin(selected_ids)]
        if len(remaining):
            extra = remaining.sample(
                n=min(max_rows - len(out), len(remaining)),
                random_state=random_state + 10000,
            )
            out = pd.concat([out, extra], ignore_index=True)

    return out.head(max_rows).drop(columns=["__row_id__"], errors="ignore").copy()
