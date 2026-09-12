from __future__ import annotations

import warnings
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator

from .models import make_y, transform_X


# ============================================================
# 1. Safe subset cho censoring-aware metrics
# ============================================================

def _safe_eval_subset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    IPCW / dynamic AUC / Brier Score yêu cầu follow-up của test
    nằm trong vùng thời gian mà train có thể hỗ trợ.
    """

    max_train = float(
        train_df["duration_days"].max()
    )

    return test_df.loc[
        test_df["duration_days"].astype(float)
        < max_train
    ].copy()


# ============================================================
# 2. Deterministic sample
# ============================================================

def _deterministic_sample(
    df: pd.DataFrame,
    max_rows: int | None,
    random_state: int,
) -> pd.DataFrame:
    """
    Lấy sample reproducible.

    Dùng chủ yếu cho IBS vì việc dự đoán toàn bộ survival curve
    trên hàng chục nghìn issue rất tốn RAM.
    """

    if (
        max_rows is None
        or len(df) <= int(max_rows)
    ):
        return df.copy()

    return df.sample(
        n=int(max_rows),
        random_state=random_state,
    ).copy()


# ============================================================
# 3. MEMORY-SAFE RISK PREDICTION
# ============================================================

def _predict_risk_in_batches(
    artifact: Dict,
    df: pd.DataFrame,
    batch_size: int = 64,
) -> np.ndarray:
    """
    Dự đoán risk score theo batch.

    Rất quan trọng với RandomSurvivalForest của scikit-survival.

    Nếu gọi:

        model.predict(X)

    trên ~80.000 issue cùng lúc, RSF có thể tạo mảng nội bộ dạng:

        n_samples × n_unique_event_times × 2

    Với dữ liệu hiện tại có ~45k unique event times,
    mảng đó có thể cần >50 GiB RAM.

    Predict theo batch giúp giảm peak memory rất mạnh.

    batch_size=64 được chọn an toàn cho máy 32 GB RAM.
    """

    n_rows = len(df)

    if n_rows == 0:
        return np.asarray(
            [],
            dtype=np.float64,
        )

    batch_size = max(
        1,
        int(batch_size),
    )

    risks = np.empty(
        n_rows,
        dtype=np.float64,
    )

    position = 0

    for start in range(
        0,
        n_rows,
        batch_size,
    ):
        batch = df.iloc[
            start:start + batch_size
        ]

        X_batch = transform_X(
            artifact,
            batch,
        )

        risk_batch = (
            artifact["model"]
            .predict(X_batch)
        )

        end = position + len(batch)

        risks[position:end] = np.asarray(
            risk_batch,
            dtype=np.float64,
        )

        position = end

        # Giải phóng các object lớn của batch hiện tại
        del X_batch
        del risk_batch

    return risks


# ============================================================
# 4. MEMORY-SAFE SURVIVAL PROBABILITIES
# ============================================================

def _survival_probabilities_at_times(
    artifact: Dict,
    df: pd.DataFrame,
    times: np.ndarray,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Dự đoán survival probabilities theo batch.

    Chỉ dùng cho IBS nên df thường đã được sample xuống
    khoảng 2.000 observations.
    """

    model = artifact["model"]

    times = np.asarray(
        times,
        dtype=float,
    )

    output = np.empty(
        (
            len(df),
            len(times),
        ),
        dtype=np.float64,
    )

    row_position = 0

    for start in range(
        0,
        len(df),
        batch_size,
    ):
        batch = df.iloc[
            start:start + batch_size
        ]

        X_batch = transform_X(
            artifact,
            batch,
        )

        survival_functions = (
            model.predict_survival_function(
                X_batch
            )
        )

        for j, fn in enumerate(
            survival_functions
        ):
            x = np.asarray(
                fn.x,
                dtype=float,
            )

            # StepFunction chỉ được gọi trong domain của nó
            clipped_times = np.clip(
                times,
                x.min(),
                x.max(),
            )

            output[
                row_position + j,
                :
            ] = [
                float(fn(t))
                for t in clipped_times
            ]

        row_position += len(batch)

        del X_batch
        del survival_functions

    return output


# ============================================================
# 5. Kaplan-Meier baseline
# ============================================================

def _km_survival_at_times(
    y_train,
    times: np.ndarray,
) -> np.ndarray:
    """
    Kaplan-Meier baseline survival probability S(t).

    Dùng để so IBS của Cox / RSF với baseline đơn giản.
    """

    km_time, km_survival = (
        kaplan_meier_estimator(
            y_train["event"],
            y_train["time"],
        )
    )

    times = np.asarray(
        times,
        dtype=float,
    )

    values = np.ones(
        len(times),
        dtype=float,
    )

    indexes = np.searchsorted(
        km_time,
        times,
        side="right",
    ) - 1

    valid = indexes >= 0

    values[valid] = (
        km_survival[
            indexes[valid]
        ]
    )

    return values


# ============================================================
# 6. MAIN EVALUATION FUNCTION
# ============================================================

def evaluate_survival_model(
    artifact: Dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    horizons_days: Iterable[float] = (
        30,
        60,
        90,
    ),
    max_ibs_rows: int | None = 2000,
    survival_batch_size: int = 16,
    risk_batch_size: int = 256,
    random_state: int = 42,
) -> Dict[str, float]:
    """
    Đánh giá survival model.

    Metrics:

    1. Harrell C-index
       - dùng toàn bộ held-out test set
       - risk prediction theo batch

    2. IPCW C-index
       - dùng toàn bộ safe evaluation subset
       - risk prediction theo batch

    3. Time-dependent AUC
       - AUC @ 30 / 60 / 90 ngày nếu hợp lệ
       - dùng toàn bộ safe subset

    4. Integrated Brier Score (IBS)
       - deterministic sample để kiểm soát RAM
       - survival prediction theo batch

    5. Kaplan-Meier IBS baseline
       - so calibration của model với baseline không có features
    """

    out: Dict[str, float] = {
        "n_test": int(
            len(test_df)
        ),
        "events_test": int(
            test_df["event"].sum()
        ),
    }

    # ========================================================
    # A. HARRELL C-INDEX
    # ========================================================

    y_test = make_y(
        test_df
    )

    risk_test = (
        _predict_risk_in_batches(
            artifact=artifact,
            df=test_df,
            batch_size=risk_batch_size,
        )
    )

    out["harrell_c"] = float(
        concordance_index_censored(
            y_test["event"],
            y_test["time"],
            risk_test,
        )[0]
    )

    del risk_test


    # ========================================================
    # B. SAFE SUBSET
    # ========================================================

    eval_df = _safe_eval_subset(
        train_df,
        test_df,
    )

    out["n_ipcw_eval"] = int(
        len(eval_df)
    )

    # Không đủ dữ liệu để tính censoring-aware metrics
    if (
        len(eval_df) < 20
        or int(
            eval_df["event"].sum()
        ) < 5
    ):
        out["ipcw_c"] = np.nan

        for h in horizons_days:
            out[
                f"auc_{int(h)}"
            ] = np.nan

        out[
            "mean_dynamic_auc"
        ] = np.nan

        out["ibs"] = np.nan
        out["km_ibs"] = np.nan
        out[
            "ibs_gain_vs_km"
        ] = np.nan
        out["ibs_rows"] = 0

        return out


    # ========================================================
    # C. BUILD SURVIVAL TARGETS
    # ========================================================

    y_train = make_y(
        train_df
    )

    y_eval = make_y(
        eval_df
    )


    # ========================================================
    # D. MEMORY-SAFE RISK FOR IPCW + AUC
    # ========================================================

    risk_eval = (
        _predict_risk_in_batches(
            artifact=artifact,
            df=eval_df,
            batch_size=risk_batch_size,
        )
    )


    # ========================================================
    # E. IPCW C-INDEX
    # ========================================================

    try:
        out["ipcw_c"] = float(
            concordance_index_ipcw(
                y_train,
                y_eval,
                risk_eval,
            )[0]
        )

    except Exception as e:
        warnings.warn(
            f"IPCW C-index failed: {e!r}"
        )

        out["ipcw_c"] = np.nan


    # ========================================================
    # F. TIME-DEPENDENT AUC
    # ========================================================

    min_t = float(
        eval_df[
            "duration_days"
        ].min()
    )

    max_t = float(
        eval_df[
            "duration_days"
        ].max()
    )

    max_train = float(
        train_df[
            "duration_days"
        ].max()
    )

    valid_horizons = [
        float(h)
        for h in horizons_days
        if (
            float(h) > min_t
            and float(h) < max_t
            and float(h) < max_train
        )
    ]

    # Mặc định NaN
    for h in horizons_days:
        out[
            f"auc_{int(h)}"
        ] = np.nan

    if valid_horizons:

        try:
            aucs, mean_auc = (
                cumulative_dynamic_auc(
                    y_train,
                    y_eval,
                    risk_eval,
                    np.asarray(
                        valid_horizons,
                        dtype=float,
                    ),
                )
            )

            for h, auc in zip(
                valid_horizons,
                aucs,
            ):
                out[
                    f"auc_{int(h)}"
                ] = float(auc)

            out[
                "mean_dynamic_auc"
            ] = float(mean_auc)

        except Exception as e:
            warnings.warn(
                f"Dynamic AUC failed: {e!r}"
            )

            out[
                "mean_dynamic_auc"
            ] = np.nan

    else:
        out[
            "mean_dynamic_auc"
        ] = np.nan

    del risk_eval


    # ========================================================
    # G. INTEGRATED BRIER SCORE
    # ========================================================

    try:

        # Chỉ sample IBS để tránh hàng GB RAM
        ibs_df = _deterministic_sample(
            eval_df,
            max_rows=max_ibs_rows,
            random_state=random_state,
        )

        out["ibs_rows"] = int(
            len(ibs_df)
        )

        y_ibs = make_y(
            ibs_df
        )

        # -----------------------------------------------
        # Time grid an toàn
        # -----------------------------------------------

        lower = max(
            float(
                ibs_df[
                    "duration_days"
                ].min()
            ) + 1e-6,

            float(
                ibs_df[
                    "duration_days"
                ].quantile(0.10)
            ),
        )

        upper = min(
            float(
                ibs_df[
                    "duration_days"
                ].max()
            ) - 1e-6,

            max_train - 1e-6,

            float(
                ibs_df[
                    "duration_days"
                ].quantile(0.90)
            ),
        )

        if upper <= lower:
            raise ValueError(
                "Không có time grid hợp lệ cho IBS."
            )

        times = np.linspace(
            lower,
            upper,
            30,
        )


        # -----------------------------------------------
        # Model survival probabilities
        # -----------------------------------------------

        survival_probabilities = (
            _survival_probabilities_at_times(
                artifact=artifact,
                df=ibs_df,
                times=times,
                batch_size=int(
                    survival_batch_size
                ),
            )
        )


        # -----------------------------------------------
        # Model IBS
        # -----------------------------------------------

        out["ibs"] = float(
            integrated_brier_score(
                y_train,
                y_ibs,
                survival_probabilities,
                times,
            )
        )

        del survival_probabilities


        # -----------------------------------------------
        # Kaplan-Meier baseline IBS
        # -----------------------------------------------

        km_values = (
            _km_survival_at_times(
                y_train,
                times,
            )
        )

        km_matrix = np.tile(
            km_values.reshape(
                1,
                -1,
            ),
            (
                len(ibs_df),
                1,
            ),
        )

        out["km_ibs"] = float(
            integrated_brier_score(
                y_train,
                y_ibs,
                km_matrix,
                times,
            )
        )

        del km_matrix


        # -----------------------------------------------
        # Positive = model tốt hơn KM baseline
        # -----------------------------------------------

        out[
            "ibs_gain_vs_km"
        ] = float(
            out["km_ibs"]
            - out["ibs"]
        )


    except Exception as e:

        warnings.warn(
            f"IBS failed: {e!r}"
        )

        out["ibs"] = np.nan
        out["km_ibs"] = np.nan

        out[
            "ibs_gain_vs_km"
        ] = np.nan

        out["ibs_rows"] = 0


    return out