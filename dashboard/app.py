from __future__ import annotations

from pathlib import Path
import json
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.explain import create_explainer, explain_rows, top_contributions

st.set_page_config(page_title="Predictive Maintenance", layout="wide")
st.title("Predictive Maintenance Command Center")
st.caption("Machine health monitoring and 24-hour failure-risk prioritization")

score_path = Path("data/processed/scored_machines.csv")
feature_path = Path("data/processed/features.parquet")
model_path = Path("models/lightgbm_predictive_maintenance.joblib")
shap_global_path = Path("artifacts/shap_global_importance.csv")
metrics_path = Path("artifacts/test_metrics.json")

@st.cache_resource
def load_explanation_resources():
    bundle = joblib.load(model_path)
    return bundle, create_explainer(bundle["model"])

if not score_path.exists() or not feature_path.exists():
    st.warning("Run data generation, training, and batch scoring first. See README Quick Start.")
    st.stop()

scores = pd.read_csv(score_path)
features = pd.read_parquet(feature_path)
scores["failure_risk"] = pd.to_numeric(scores["failure_risk"])

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Machines", scores["machine_id"].nunique())
c2.metric("Alerts", int(scores["alert"].sum()))
c3.metric("High Risk", int((scores["risk_band"] == "HIGH").sum()))
c4.metric("Mean Risk", f"{scores['failure_risk'].mean():.1%}")
c5.metric("Max Risk", f"{scores['failure_risk'].max():.1%}")

st.subheader("Current machine risk ranking")
fig = px.bar(scores.head(20), x="machine_id", y="failure_risk", color="risk_band", hover_data=["temperature_c", "vibration_mm_s", "hours_since_maintenance"])
st.plotly_chart(fig, use_container_width=True)
st.dataframe(scores, use_container_width=True, hide_index=True)

st.subheader("Machine telemetry drill-down")
machine = st.selectbox("Machine", sorted(features["machine_id"].unique()))
metric = st.selectbox("Metric", ["temperature_c", "vibration_mm_s", "pressure_bar", "rpm", "power_kw"])
m = features[features["machine_id"] == machine].sort_values("timestamp").tail(7 * 24)
line = px.line(m, x="timestamp", y=metric, title=f"{machine}: last 7 days")
st.plotly_chart(line, use_container_width=True)

st.subheader("Why the model assigned this risk")
if model_path.exists():
    bundle, explainer = load_explanation_resources()
    current = m.tail(1)[bundle["features"]]
    explanation = explain_rows(explainer, current)
    contributions = pd.DataFrame(top_contributions(explanation, limit=10))
    contributions["feature_label"] = contributions["feature"].str.replace("_", " ").str.title()
    local_chart = px.bar(
        contributions.sort_values("shap_value"),
        x="shap_value", y="feature_label", color="direction", orientation="h",
        hover_data=["feature_value"],
        labels={"shap_value": "Contribution to raw model score", "feature_label": "Feature"},
    )
    st.plotly_chart(local_chart, use_container_width=True)
    st.caption("Positive SHAP values increase modeled failure risk; negative values reduce it. SHAP explains model behavior, not physical root cause.")
else:
    st.info("Train the model to view local SHAP explanations.")

if shap_global_path.exists():
    st.subheader("Global model drivers")
    global_importance = pd.read_csv(shap_global_path).head(15).sort_values("mean_abs_shap")
    global_importance["feature_label"] = global_importance["feature"].str.replace("_", " ").str.title()
    global_chart = px.bar(
        global_importance, x="mean_abs_shap", y="feature_label", orientation="h",
        labels={"mean_abs_shap": "Mean absolute SHAP value", "feature_label": "Feature"},
    )
    st.plotly_chart(global_chart, use_container_width=True)

if metrics_path.exists():
    st.subheader("Operational evaluation")
    evaluation = json.loads(metrics_path.read_text(encoding="utf-8"))
    operational = evaluation.get("operational_metrics")
    if operational:
        o1, o2, o3, o4 = st.columns(4)
        o1.metric("Failure events detected", f"{operational['event_detection_rate']:.1%}")
        o2.metric("Median lead time", f"{operational['median_lead_time_hours']:.1f} h" if operational["median_lead_time_hours"] is not None else "N/A")
        o3.metric("Alert episodes", f"{operational['alert_episodes']:,}")
        o4.metric("Episodes / machine-week", f"{operational['alert_episodes_per_machine_week']:.2f}")
        st.write(
            f"False alert episodes: **{operational['false_alert_episodes']:,}** "
            f"({operational['false_alert_episode_rate']:.1%}); "
            f"mean alert persistence: **{operational['mean_alert_persistence_hours']:.1f} hours**; "
            f"mean failure-window coverage: **{operational['mean_failure_window_coverage']:.1%}**."
        )
        st.caption(
            "Consecutive hourly alerts are grouped into one episode. Failure events without a complete "
            "24-hour evaluation window at the test boundary are excluded from event-level rates."
        )
