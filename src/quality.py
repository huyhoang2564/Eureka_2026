from __future__ import annotations

import pandas as pd


FORBIDDEN_PREDICTORS = {
    "resolution_date",
    "duration_days",
    "event",
    "project_cutoff",
    "updated",
    "current_status",
    "final_status",
    "resolution",
}


def summarize_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "column": df.columns,
            "missing": [int(df[c].isna().sum()) for c in df.columns],
            "missing_rate": [float(df[c].isna().mean()) for c in df.columns],
        }
    )
    return out.sort_values("missing_rate", ascending=False)


def assert_no_forbidden_predictors(feature_cols):
    bad = sorted(set(feature_cols) & FORBIDDEN_PREDICTORS)
    if bad:
        raise ValueError(f"Phát hiện leakage predictors: {bad}")


def cohort_summary(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "rows", "value": len(df)},
            {"metric": "projects", "value": df["project"].nunique()},
            {"metric": "repositories", "value": df["repository"].nunique()},
            {"metric": "events", "value": int(df["event"].sum())},
            {"metric": "censored", "value": int((~df["event"]).sum())},
            {"metric": "censoring_rate", "value": float((~df["event"]).mean())},
            {"metric": "median_duration_days", "value": float(df["duration_days"].median())},
        ]
    )
