from __future__ import annotations

from typing import Dict, Iterable
import numpy as np
import pandas as pd

from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)

from .models import make_y, transform_X


def _safe_eval_subset(train_df: pd.DataFrame, test_df: pd.DataFrame):
    max_train = float(train_df["duration_days"].max())
    # IPCW cần test follow-up không vượt support train.
    mask = test_df["duration_days"].astype(float) < max_train
    return test_df.loc[mask].copy()


def evaluate_survival_model(
    artifact: Dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizons_days: Iterable[float] = (30, 60, 90),
) -> Dict[str, float]:
    model = artifact["model"]

    X_test = transform_X(artifact, test_df)
    y_test = make_y(test_df)
    risk = model.predict(X_test)

    out = {}
    out["n_test"] = int(len(test_df))
    out["events_test"] = int(test_df["event"].sum())
    out["harrell_c"] = float(
        concordance_index_censored(
            y_test["event"],
            y_test["time"],
            risk,
        )[0]
    )

    eval_df = _safe_eval_subset(train_df, test_df)
    if len(eval_df) < 20 or eval_df["event"].sum() < 5:
        out["ipcw_c"] = np.nan
        for h in horizons_days:
            out[f"auc_{int(h)}"] = np.nan
        out["mean_dynamic_auc"] = np.nan
        out["ibs"] = np.nan
        return out

    y_train = make_y(train_df)
    y_eval = make_y(eval_df)
    X_eval = transform_X(artifact, eval_df)
    risk_eval = model.predict(X_eval)

    try:
        out["ipcw_c"] = float(
            concordance_index_ipcw(
                y_train,
                y_eval,
                risk_eval,
            )[0]
        )
    except Exception:
        out["ipcw_c"] = np.nan

    min_t = float(eval_df["duration_days"].min())
    max_t = float(eval_df["duration_days"].max())
    max_train = float(train_df["duration_days"].max())

    valid_horizons = [
        float(h)
        for h in horizons_days
        if h > min_t and h < max_t and h < max_train
    ]

    for h in horizons_days:
        out[f"auc_{int(h)}"] = np.nan

    if valid_horizons:
        try:
            aucs, mean_auc = cumulative_dynamic_auc(
                y_train,
                y_eval,
                risk_eval,
                np.asarray(valid_horizons, dtype=float),
            )
            for h, auc in zip(valid_horizons, aucs):
                out[f"auc_{int(h)}"] = float(auc)
            out["mean_dynamic_auc"] = float(mean_auc)
        except Exception:
            out["mean_dynamic_auc"] = np.nan
    else:
        out["mean_dynamic_auc"] = np.nan

    # IBS trên vùng follow-up trung tâm để tránh biên không ổn định.
    try:
        lower = max(min_t + 1e-6, float(eval_df["duration_days"].quantile(0.10)))
        upper = min(
            max_t - 1e-6,
            max_train - 1e-6,
            float(eval_df["duration_days"].quantile(0.90)),
        )

        if upper <= lower:
            raise ValueError("Không có time grid hợp lệ cho IBS.")

        times = np.linspace(lower, upper, 50)

        surv_fns = model.predict_survival_function(X_eval)
        surv_probs = np.asarray(
            [[float(fn(t)) for t in times] for fn in surv_fns],
            dtype=float,
        )

        out["ibs"] = float(
            integrated_brier_score(
                y_train,
                y_eval,
                surv_probs,
                times,
            )
        )
    except Exception:
        out["ibs"] = np.nan

    return out
