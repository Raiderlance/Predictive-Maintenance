from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.explain import create_explainer, explain_rows, top_contributions
from src.scoring import risk_band


FEATURE_PATH = Path("data/processed/ims_features.parquet")
MODEL_PATH = Path("models/ims_lightgbm.joblib")
METRICS_PATH = Path("artifacts/ims_test_metrics.json")
SHAP_GLOBAL_PATH = Path("artifacts/ims_shap_global_importance.csv")
RUN_LABELS = {
    "1st_test": "Set 1 — training",
    "2nd_test": "Set 2 — validation",
    "4th_test": "Set 3 — exploratory evaluation",
}


st.set_page_config(page_title="NASA IMS Bearing Monitor", page_icon="⚙️", layout="wide")
st.title("NASA IMS Bearing Condition Monitor")
st.caption("Run-to-failure vibration analysis, 24-hour risk prediction, and SHAP explanations")


@st.cache_resource
def load_model_resources():
    bundle = joblib.load(MODEL_PATH)
    return bundle, create_explainer(bundle["model"])


@st.cache_data
def load_scored_features(model_modified_time: float) -> pd.DataFrame:
    del model_modified_time  # cache key invalidates predictions when the model changes
    data = pd.read_parquet(FEATURE_PATH)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    bundle = joblib.load(MODEL_PATH)
    threshold = float(bundle.get("threshold", 0.5))
    data["failure_risk"] = bundle["model"].predict_proba(data[bundle["features"]])[:, 1]
    data["alert"] = data["failure_risk"] >= threshold
    data["risk_band"] = data["failure_risk"].map(lambda risk: risk_band(risk, threshold))
    return data


required = [FEATURE_PATH, MODEL_PATH]
missing = [str(path) for path in required if not path.exists()]
if missing:
    st.error("NASA IMS dashboard assets are missing: " + ", ".join(missing))
    st.code("python -m src.ims_download\npython -m src.ims_features\npython -m src.ims_train")
    st.stop()

bundle, explainer = load_model_resources()
threshold = float(bundle.get("threshold", 0.5))
data = load_scored_features(MODEL_PATH.stat().st_mtime)

with st.sidebar:
    st.header("Experiment controls")
    run_id = st.selectbox(
        "Dataset split",
        options=list(RUN_LABELS),
        index=2,
        format_func=lambda value: RUN_LABELS[value],
    )
    run = data[data["run_id"] == run_id].copy()
    bearing_id = st.selectbox("Bearing", sorted(run["bearing_id"].unique()))
    st.metric("Alert threshold", f"{threshold:.2f}")
    st.caption("Threshold selected on Set 2 validation data only.")
    st.divider()
    st.warning(
        "NASA records which bearings had failed by the end of each test. "
        "Run-end time is used as the observed failure time."
    )

latest = run.sort_values("timestamp").groupby("bearing_id", as_index=False).tail(1)
failed_bearings = sorted(run.loc[run["failed_at_run_end"], "bearing_id"].unique())
run_start, run_end = run["timestamp"].min(), run["timestamp"].max()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Run", RUN_LABELS[run_id].split(" — ")[0])
c2.metric("Recordings", f"{run['snapshot_index'].nunique():,}")
c3.metric("Duration", f"{(run_end - run_start).total_seconds() / 86400:.1f} days")
c4.metric("Documented failures", ", ".join(f"B{value}" for value in failed_bearings))
c5.metric("Latest active alerts", int(latest["alert"].sum()))

overview_tab, bearing_tab, explain_tab, evaluation_tab = st.tabs(
    ["Fleet overview", "Bearing drill-down", "Explainability", "Model evaluation"]
)

with overview_tab:
    st.subheader("Latest bearing risk ranking")
    ranking = latest[["bearing_id", "timestamp", "failure_risk", "risk_band", "alert", "failed_at_run_end"]].copy()
    ranking["bearing"] = ranking["bearing_id"].map(lambda value: f"Bearing {value}")
    ranking = ranking.sort_values("failure_risk", ascending=False)
    ranking_chart = px.bar(
        ranking,
        x="bearing",
        y="failure_risk",
        color="risk_band",
        category_orders={"risk_band": ["LOW", "MEDIUM", "HIGH"]},
        color_discrete_map={"LOW": "#2ca02c", "MEDIUM": "#ffbf00", "HIGH": "#d62728"},
        hover_data=["alert", "failed_at_run_end", "timestamp"],
        labels={"failure_risk": "Predicted 24-hour failure risk", "bearing": "Bearing"},
    )
    ranking_chart.add_hline(y=threshold, line_dash="dash", annotation_text="Alert threshold")
    st.plotly_chart(ranking_chart, use_container_width=True)
    st.dataframe(
        ranking[["bearing", "timestamp", "failure_risk", "risk_band", "alert", "failed_at_run_end"]],
        use_container_width=True,
        hide_index=True,
        column_config={"failure_risk": st.column_config.ProgressColumn("Failure risk", min_value=0, max_value=1, format="%.3f")},
    )

    st.subheader("Cross-bearing fault-localization signals")
    final_window = run[run["timestamp"] >= run_end - pd.Timedelta(hours=24)].copy()
    localization = final_window.groupby("bearing_id", as_index=False).agg(
        mean_risk=("failure_risk", "mean"),
        mean_rms_peer_ratio=("rms_mean_peer_ratio", "mean"),
        peak_rms_peer_ratio=("rms_mean_peer_ratio", "max"),
        alerts=("alert", "sum"),
        documented_failure=("failed_at_run_end", "first"),
    )
    localization["bearing"] = localization["bearing_id"].map(lambda value: f"Bearing {value}")
    localization_chart = px.bar(
        localization,
        x="bearing",
        y="mean_rms_peer_ratio",
        color="documented_failure",
        hover_data=["mean_risk", "peak_rms_peer_ratio", "alerts"],
        color_discrete_map={True: "#d62728", False: "#4c78a8"},
        labels={"mean_rms_peer_ratio": "Mean RMS / peer median", "bearing": "Bearing"},
    )
    localization_chart.add_hline(y=1.0, line_dash="dash", annotation_text="Peer median")
    st.plotly_chart(localization_chart, use_container_width=True)
    suspected = localization.sort_values(
        ["mean_rms_peer_ratio", "mean_risk"], ascending=False
    ).iloc[0]
    st.info(
        f"Bearing {int(suspected['bearing_id'])} has the strongest final-window relative RMS signal. "
        "This is a localization indicator, not a confirmed physical diagnosis."
    )

with bearing_tab:
    bearing = run[run["bearing_id"] == bearing_id].sort_values("timestamp").copy()
    documented_failure = bool(bearing["failed_at_run_end"].iloc[0])
    st.subheader(f"Bearing {bearing_id} risk trajectory")
    risk_chart = go.Figure()
    risk_chart.add_trace(go.Scatter(
        x=bearing["timestamp"], y=bearing["failure_risk"], mode="lines",
        name="Failure risk", line={"color": "#d62728"},
    ))
    risk_chart.add_hline(y=threshold, line_dash="dash", annotation_text="Alert threshold")
    if documented_failure:
        horizon_start = run_end - pd.Timedelta(hours=24)
        risk_chart.add_vrect(
            x0=horizon_start, x1=run_end, fillcolor="orange", opacity=0.15,
            line_width=0, annotation_text="Final 24-hour failure window",
        )
    risk_chart.update_layout(
        xaxis_title="Timestamp", yaxis_title="Predicted failure risk", yaxis_range=[0, 1],
        hovermode="x unified",
    )
    st.plotly_chart(risk_chart, use_container_width=True)

    feature_options = [
        "rms_mean", "std_mean", "kurtosis_mean", "crest_factor_mean",
        "spectral_centroid_hz_mean", "dominant_frequency_hz_mean",
        "bandpower_0_1khz_mean", "bandpower_1_5khz_mean", "bandpower_5_10khz_mean",
        "rms_mean_peer_ratio", "std_mean_peer_ratio", "peak_mean_peer_ratio",
        "kurtosis_mean_peer_ratio", "spectral_centroid_hz_mean_peer_ratio",
    ]
    selected_feature = st.selectbox(
        "Waveform-derived feature",
        feature_options,
        format_func=lambda value: value.replace("_", " ").title(),
    )
    feature_chart = px.line(
        bearing, x="timestamp", y=selected_feature,
        labels={selected_feature: selected_feature.replace("_", " ").title(), "timestamp": "Timestamp"},
    )
    st.plotly_chart(feature_chart, use_container_width=True)

    if documented_failure:
        first_alert = bearing[(bearing["failure_within_horizon"] == 1) & bearing["alert"]].head(1)
        if len(first_alert):
            warning_hours = float(first_alert.iloc[0]["hours_to_run_end"])
            st.success(f"First alert inside the final 24-hour window occurred {warning_hours:.1f} hours before run end.")
        else:
            st.error("No alert was raised inside the final 24-hour failure window.")
    else:
        false_alerts = int(bearing["alert"].sum())
        st.info(
            f"This bearing has no documented run-end failure. Sensor alerts across the run: {false_alerts:,}. "
            "They may reflect transmitted vibration from another bearing, not necessarily a local defect."
        )

with explain_tab:
    st.subheader(f"Why Bearing {bearing_id} received its latest score")
    selected = run[run["bearing_id"] == bearing_id].sort_values("timestamp").tail(1)
    explanation = explain_rows(explainer, selected[bundle["features"]])
    contributions = pd.DataFrame(top_contributions(explanation, limit=10))
    contributions["feature_label"] = contributions["feature"].str.replace("_", " ").str.title()
    local_chart = px.bar(
        contributions.sort_values("shap_value"),
        x="shap_value", y="feature_label", color="direction", orientation="h",
        hover_data=["feature_value"],
        color_discrete_map={"increases_risk": "#d62728", "decreases_risk": "#2ca02c"},
        labels={"shap_value": "Contribution to raw model score", "feature_label": "Feature"},
    )
    st.plotly_chart(local_chart, use_container_width=True)
    st.caption(
        "Positive values push the model toward higher risk and negative values push it toward lower risk. "
        "SHAP explains the model's calculation; it does not diagnose the physical failure mode."
    )

    if SHAP_GLOBAL_PATH.exists():
        st.subheader("Global IMS model drivers")
        global_importance = pd.read_csv(SHAP_GLOBAL_PATH).head(15).sort_values("mean_abs_shap")
        global_importance["feature_label"] = global_importance["feature"].str.replace("_", " ").str.title()
        global_chart = px.bar(
            global_importance, x="mean_abs_shap", y="feature_label", orientation="h",
            labels={"mean_abs_shap": "Mean absolute SHAP value", "feature_label": "Feature"},
        )
        st.plotly_chart(global_chart, use_container_width=True)

with evaluation_tab:
    st.subheader("Set 3 exploratory re-evaluation")
    st.caption(
        "Set 3 informed the peer-relative feature design and is no longer a pristine final test set."
    )
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
        m2.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
        m3.metric("Precision", f"{metrics['precision']:.3f}")
        m4.metric("Recall", f"{metrics['recall']:.3f}")
        m5.metric("F1", f"{metrics['f1']:.3f}")

        confusion = metrics["confusion_matrix"]
        confusion_frame = pd.DataFrame(
            confusion,
            index=["Actual healthy", "Actual within horizon"],
            columns=["Predicted healthy", "Predicted alert"],
        )
        st.dataframe(confusion_frame, use_container_width=True)
        alert_rate = metrics.get("non_failed_bearing_alert_rate", metrics.get("healthy_control_false_alert_rate", 0))
        alert_count = metrics.get("non_failed_bearing_alerts", metrics.get("healthy_control_alerts", 0))
        st.write(
            f"Non-failed-bearing alert rate: **{alert_rate:.3%}** "
            f"({alert_count:,} sensor alerts)."
        )
        st.caption(
            "These are false positives under component-level labels, but some may be genuine responses "
            "to vibration transmitted from the documented failed bearing through the shared rig."
        )
        operational = metrics.get("operational_metrics")
        if operational:
            st.subheader("Operational alert metrics")
            o1, o2, o3, o4 = st.columns(4)
            o1.metric("Failure events detected", f"{operational['event_detection_rate']:.1%}")
            o2.metric("Median lead time", f"{operational['median_lead_time_hours']:.1f} h")
            o3.metric("Alert episodes", f"{operational['alert_episodes']:,}")
            o4.metric("Episodes / bearing-week", f"{operational['alert_episodes_per_machine_week']:.2f}")
            st.write(
                f"False alert episodes: **{operational['false_alert_episodes']:,}**; "
                f"mean persistence: **{operational['mean_alert_persistence_hours']:.1f} hours**; "
                f"failure-window coverage: **{operational['mean_failure_window_coverage']:.1%}**."
            )
        detections = pd.DataFrame(metrics.get("failed_bearing_detections", []))
        if not detections.empty:
            st.subheader("Failure warning summary")
            st.dataframe(detections, use_container_width=True, hide_index=True)
    else:
        st.info("Run `python -m src.ims_train` to create evaluation metrics.")

st.divider()
st.caption(
    "Research benchmark using the NASA IMS accelerated bearing-life experiments. "
    "Not validated for autonomous maintenance or safety-critical decisions."
)
