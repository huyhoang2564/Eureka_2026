from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


DATE_COLS = ["created", "updated", "resolution_date"]


def load_flat_dataset(path: str | Path) -> pd.DataFrame:
    """Đọc một file Parquet hoặc cả thư mục Parquet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_parquet(path)

    for c in DATE_COLS:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    return df


def build_survival_cohort(
    flat_df: pd.DataFrame,
    min_positive_days: float = 1.0 / 1440.0,  # 1 phút
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    event = 1 nếu có resolution_date hợp lệ.
    event = 0 nếu chưa resolve tại cuối cửa sổ quan sát.

    Censoring cutoff được ước lượng theo từng repository+project bằng
    timestamp lớn nhất có thể quan sát (created/updated/resolution_date).
    Đây là proxy thực nghiệm; cần ghi rõ trong phần limitations.
    """
    df = flat_df.copy()

    for c in DATE_COLS:
        df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    n_input = len(df)

    # Bắt buộc phải có created.
    df = df[df["created"].notna()].copy()

    # observed_time = max(created, updated, resolution_date) của từng issue.
    observed = pd.concat(
        [
            df["created"].rename("created"),
            df["updated"].rename("updated"),
            df["resolution_date"].rename("resolution"),
        ],
        axis=1,
    ).max(axis=1)

    df["_observed_time"] = observed

    # Project-specific study end proxy.
    cutoff = (
        df.groupby(["repository", "project"], dropna=False)["_observed_time"]
        .max()
        .rename("project_cutoff")
        .reset_index()
    )
    df = df.merge(cutoff, on=["repository", "project"], how="left")

    # Resolution trước created là invalid.
    valid_resolution = (
        df["resolution_date"].notna()
        & (df["resolution_date"] >= df["created"])
    )

    df["event"] = valid_resolution.astype(bool)

    end_time = df["project_cutoff"].copy()
    end_time.loc[df["event"]] = df.loc[df["event"], "resolution_date"]

    df["duration_days"] = (
        (end_time - df["created"]).dt.total_seconds() / 86400.0
    )

    # Loại record có thời gian âm / missing.
    invalid_duration = df["duration_days"].isna() | (df["duration_days"] < 0)
    n_invalid_duration = int(invalid_duration.sum())
    df = df[~invalid_duration].copy()

    # Giữ issue resolved cùng thời điểm bằng epsilon dương.
    zero_duration = df["duration_days"] <= 0
    n_zero_adjusted = int(zero_duration.sum())
    df.loc[zero_duration, "duration_days"] = min_positive_days

    df["created_weekday"] = df["created"].dt.weekday.astype("int64")
    df["created_hour"] = df["created"].dt.hour.astype("int64")

    df.drop(columns=["_observed_time"], inplace=True, errors="ignore")

    report = pd.DataFrame(
        [
            {"metric": "input_rows", "value": n_input},
            {"metric": "rows_without_created_removed", "value": n_input - len(flat_df[flat_df["created"].notna()])},
            {"metric": "invalid_duration_removed", "value": n_invalid_duration},
            {"metric": "zero_duration_adjusted", "value": n_zero_adjusted},
            {"metric": "final_rows", "value": len(df)},
            {"metric": "events_resolved", "value": int(df["event"].sum())},
            {"metric": "censored_unresolved", "value": int((~df["event"]).sum())},
            {"metric": "censoring_rate", "value": float((~df["event"]).mean()) if len(df) else np.nan},
        ]
    )

    return df, report
