from __future__ import annotations

import numpy as np
import pandas as pd


def safe_step_value(fn, t: float) -> float:
    """
    StepFunction của scikit-survival chỉ hợp lệ trong miền thời gian của model.
    Ta clip t vào miền đó.
    """
    x = np.asarray(fn.x, dtype=float)
    t_clip = float(np.clip(t, x.min(), x.max()))
    return float(fn(t_clip))


def conditional_unresolved_probability(
    survival_fn,
    current_age_days: float,
    horizon_days: float,
) -> float:
    """
    P(T > age+h | T > age) = S(age+h)/S(age)
    """
    s_now = safe_step_value(survival_fn, current_age_days)
    s_future = safe_step_value(survival_fn, current_age_days + horizon_days)

    if s_now <= 0:
        return np.nan

    return float(np.clip(s_future / s_now, 0.0, 1.0))


def risk_band(scores: pd.Series) -> pd.Series:
    """
    Data-driven bands theo phân vị:
    bottom 50% = Low, 50-80% = Medium, top 20% = High.
    Đây là quy ước vận hành, không phải ngưỡng khoa học cố định.
    """
    q50 = scores.quantile(0.50)
    q80 = scores.quantile(0.80)

    return pd.cut(
        scores,
        bins=[-np.inf, q50, q80, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    )
