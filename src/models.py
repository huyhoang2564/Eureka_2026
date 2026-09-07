from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv


def make_y(df: pd.DataFrame):
    return Surv.from_arrays(
        event=df["event"].astype(bool).to_numpy(),
        time=df["duration_days"].astype(float).to_numpy(),
    )


def _make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # tương thích scikit-learn cũ hơn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(numeric_cols: List[str], categorical_cols: List[str]):
    num_pipe = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    cat_pipe = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_ohe()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def fit_cox(
    train_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    alpha: float = 0.01,
):
    preprocessor = make_preprocessor(numeric_cols, categorical_cols)
    X = preprocessor.fit_transform(train_df)
    y = make_y(train_df)

    model = CoxPHSurvivalAnalysis(alpha=alpha)
    model.fit(X, y)

    feature_names = list(preprocessor.get_feature_names_out())

    return {
        "kind": "cox",
        "preprocessor": preprocessor,
        "model": model,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_names": feature_names,
    }


def fit_rsf(
    train_df: pd.DataFrame,
    numeric_cols: List[str],
    categorical_cols: List[str],
    **params,
):
    preprocessor = make_preprocessor(numeric_cols, categorical_cols)
    X = preprocessor.fit_transform(train_df)
    y = make_y(train_df)

    defaults = dict(
        n_estimators=200,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    defaults.update(params)

    model = RandomSurvivalForest(**defaults)
    model.fit(X, y)

    feature_names = list(preprocessor.get_feature_names_out())

    return {
        "kind": "rsf",
        "preprocessor": preprocessor,
        "model": model,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "feature_names": feature_names,
    }


def transform_X(artifact: Dict, df: pd.DataFrame):
    return artifact["preprocessor"].transform(df)


def predict_risk(artifact: Dict, df: pd.DataFrame):
    X = transform_X(artifact, df)
    return artifact["model"].predict(X)


def predict_survival_functions(artifact: Dict, df: pd.DataFrame):
    X = transform_X(artifact, df)
    return artifact["model"].predict_survival_function(X)


def save_artifact(artifact: Dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)


def load_artifact(path: str | Path):
    return joblib.load(path)
