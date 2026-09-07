from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import load_artifact, predict_survival_functions
from src.temporal import conditional_unresolved_probability, risk_band


st.set_page_config(page_title="Euréka 2026 - Jira Early Warning", layout="wide")
st.title("Jira Issue Early-Warning Prototype")
st.caption(
    "Proof-of-concept: ước lượng xác suất issue vẫn chưa được resolve trong một khoảng thời gian tiếp theo."
)

model_choice = st.sidebar.selectbox("Model", ["RSF", "Cox PH"])
horizon = st.sidebar.selectbox("Warning horizon (days)", [30, 60, 90], index=0)

model_file = (
    ROOT / "results" / "models" / "rsf_day0.joblib"
    if model_choice == "RSF"
    else ROOT / "results" / "models" / "cox_day0.joblib"
)
data_file = ROOT / "data" / "processed" / "model_day0.parquet"

if not model_file.exists():
    st.error(f"Chưa có model: {model_file}. Hãy chạy notebook 06/07 trước.")
    st.stop()

if not data_file.exists():
    st.error(f"Chưa có dữ liệu: {data_file}. Hãy chạy notebook 04 trước.")
    st.stop()

artifact = load_artifact(model_file)
df = pd.read_parquet(data_file)

# Demo trên censored issues trong dataset.
active = df[~df["event"].astype(bool)].copy()
if active.empty:
    st.warning("Không tìm thấy censored/unresolved issue để demo.")
    st.stop()

# Giới hạn để dashboard phản hồi nhanh.
demo = active.sample(n=min(500, len(active)), random_state=42).copy()
surv_fns = predict_survival_functions(artifact, demo)

demo["current_age_days"] = demo["duration_days"].astype(float)
demo["p_unresolved_next_horizon"] = [
    conditional_unresolved_probability(fn, age, horizon)
    for fn, age in zip(surv_fns, demo["current_age_days"])
]
demo["risk_band"] = risk_band(demo["p_unresolved_next_horizon"])

c1, c2, c3 = st.columns(3)
c1.metric("Issues in demo", len(demo))
c2.metric("High risk", int((demo["risk_band"] == "High").sum()))
c3.metric("Median age (days)", f"{demo['current_age_days'].median():.1f}")

st.subheader("Risk ranking")
display_cols = [
    "repository",
    "project",
    "issue_key",
    "current_age_days",
    "p_unresolved_next_horizon",
    "risk_band",
]
st.dataframe(
    demo.sort_values("p_unresolved_next_horizon", ascending=False)[display_cols],
    use_container_width=True,
)

st.subheader("Issue detail")
choices = demo.sort_values("p_unresolved_next_horizon", ascending=False)["issue_key"].astype(str).tolist()
issue_key = st.selectbox("Issue", choices)

row_pos = demo.reset_index(drop=True).index[
    demo.reset_index(drop=True)["issue_key"].astype(str) == issue_key
][0]
row = demo.reset_index(drop=True).iloc[[row_pos]]
fn = predict_survival_functions(artifact, row)[0]

age = float(row["duration_days"].iloc[0])
risk = conditional_unresolved_probability(fn, age, horizon)

st.write(
    {
        "repository": row["repository"].iloc[0],
        "project": row["project"].iloc[0],
        "issue_key": issue_key,
        "current_age_days": round(age, 2),
        f"P(unresolved +{horizon}d | unresolved now)": round(risk, 4),
    }
)

times = np.asarray(fn.x, dtype=float)
plot_df = pd.DataFrame(
    {
        "days": times,
        "S(t) = P(unresolved after t)": [float(fn(t)) for t in times],
    }
).set_index("days")

st.line_chart(plot_df)
st.info(
    "Cảnh báo là công cụ hỗ trợ ưu tiên review; không thay thế business priority, severity, dependency hay quyết định của PM."
)
