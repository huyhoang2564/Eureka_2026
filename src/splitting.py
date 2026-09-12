from __future__ import annotations

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def add_project_uid(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "repository" in out.columns:
        out["project_uid"] = out["repository"].astype(str) + "::" + out["project"].astype(str)
    else:
        out["project_uid"] = out["project"].astype(str)
    return out


def make_project_split(
    data: pd.DataFrame,
    random_state: int = 42,
    temp_fraction: float = 0.30,
    test_fraction_of_temp: float = 0.50,
):
    data = add_project_uid(data)
    groups = data["project_uid"]

    gss1 = GroupShuffleSplit(n_splits=1, test_size=temp_fraction, random_state=random_state)
    train_idx, temp_idx = next(gss1.split(data, groups=groups))
    train = data.iloc[train_idx].copy()
    temp = data.iloc[temp_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_fraction_of_temp, random_state=random_state + 1)
    val_rel, test_rel = next(gss2.split(temp, groups=temp["project_uid"]))
    val = temp.iloc[val_rel].copy()
    test = temp.iloc[test_rel].copy()

    tr, va, te = set(train["project_uid"]), set(val["project_uid"]), set(test["project_uid"])
    assert tr.isdisjoint(va)
    assert tr.isdisjoint(te)
    assert va.isdisjoint(te)

    return train, val, test
