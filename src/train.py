from __future__ import annotations

import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score

from src.config import load_config
from src.explain import generate_shap_artifacts
from src.features import build_features, model_features


def calculate_operational_metrics(
    scored: pd.DataFrame,
    entity_col: str,
    event_col: str,
    horizon_hours: float = 24,
) -> dict:
    """Evaluate failures and alert episodes rather than treating every row independently."""
    required = {entity_col, "timestamp", "alert", event_col}
    missing = required - set(scored.columns)
    if missing:
        raise ValueError(f"Missing operational metric columns: {sorted(missing)}")

    data = scored.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values([entity_col, "timestamp"]).reset_index(drop=True)
    horizon = pd.Timedelta(hours=horizon_hours)
    deltas = data.groupby(entity_col)["timestamp"].diff().dropna()
    cadence = deltas[deltas > pd.Timedelta(0)].median()
    if pd.isna(cadence):
        cadence = pd.Timedelta(hours=1)

    event_rows = []
    total_events = int(data[event_col].astype(bool).sum())
    for entity, group in data.groupby(entity_col, sort=False):
        start = group["timestamp"].min()
        for event_time in group.loc[group[event_col].astype(bool), "timestamp"]:
            if event_time - start < horizon:
                continue
            window = group[(group["timestamp"] >= event_time - horizon) & (group["timestamp"] < event_time)]
            alerts = window[window["alert"].astype(bool)]
            detected = not alerts.empty
            event_rows.append({
                "entity_id": str(entity),
                "failure_timestamp": event_time.isoformat(),
                "detected": detected,
                "first_alert_timestamp": alerts.iloc[0]["timestamp"].isoformat() if detected else None,
                "lead_time_hours": float((event_time - alerts.iloc[0]["timestamp"]).total_seconds() / 3600) if detected else None,
                "failure_window_coverage": float(window["alert"].mean()) if len(window) else 0.0,
            })

    previous_alert = data.groupby(entity_col)["alert"].shift(fill_value=False).astype(bool)
    gaps = data.groupby(entity_col)["timestamp"].diff()
    episode_start = data["alert"].astype(bool) & (~previous_alert | gaps.gt(cadence * 1.5))
    data["episode_id"] = episode_start.groupby(data[entity_col]).cumsum()
    episodes = []
    for (entity, episode_id), episode in data[data["alert"].astype(bool)].groupby([entity_col, "episode_id"]):
        episode_begin = episode["timestamp"].min()
        episode_end = episode["timestamp"].max()
        entity_events = data.loc[(data[entity_col] == entity) & data[event_col].astype(bool), "timestamp"]
        associated = ((entity_events > episode_begin) & (entity_events <= episode_end + horizon)).any()
        episodes.append({
            "entity_id": str(entity),
            "episode_id": int(episode_id),
            "start_timestamp": episode_begin.isoformat(),
            "end_timestamp": episode_end.isoformat(),
            "observations": int(len(episode)),
            "persistence_hours": float(((episode_end - episode_begin) + cadence).total_seconds() / 3600),
            "followed_by_failure": bool(associated),
        })

    detected_events = sum(row["detected"] for row in event_rows)
    eligible_events = len(event_rows)
    lead_times = [row["lead_time_hours"] for row in event_rows if row["lead_time_hours"] is not None]
    coverages = [row["failure_window_coverage"] for row in event_rows]
    false_episodes = sum(not row["followed_by_failure"] for row in episodes)
    exposure_hours = 0.0
    for _, group in data.groupby(entity_col):
        exposure_hours += ((group["timestamp"].max() - group["timestamp"].min()) + cadence).total_seconds() / 3600
    persistence = [row["persistence_hours"] for row in episodes]

    return {
        "failure_events": total_events,
        "eligible_failure_events": eligible_events,
        "boundary_failure_events_excluded": total_events - eligible_events,
        "detected_failure_events": int(detected_events),
        "missed_failure_events": int(eligible_events - detected_events),
        "event_detection_rate": float(detected_events / eligible_events) if eligible_events else 0.0,
        "mean_lead_time_hours": float(pd.Series(lead_times).mean()) if lead_times else None,
        "median_lead_time_hours": float(pd.Series(lead_times).median()) if lead_times else None,
        "alert_episodes": len(episodes),
        "false_alert_episodes": int(false_episodes),
        "false_alert_episode_rate": float(false_episodes / len(episodes)) if episodes else 0.0,
        "alert_episodes_per_machine_week": float(len(episodes) / (exposure_hours / 168)) if exposure_hours else 0.0,
        "mean_alert_persistence_hours": float(pd.Series(persistence).mean()) if persistence else 0.0,
        "max_alert_persistence_hours": float(max(persistence)) if persistence else 0.0,
        "mean_failure_window_coverage": float(pd.Series(coverages).mean()) if coverages else 0.0,
        "failure_event_details": event_rows,
    }


def temporal_split(
    df: pd.DataFrame,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    purge_hours: int = 24,
):
    """Chronologically split data and purge labels that cross a boundary."""
    timestamps = df["timestamp"].sort_values().unique()
    if len(timestamps) < 3:
        raise ValueError("At least three unique timestamps are required")
    train_cut = timestamps[int(len(timestamps) * train_frac)]
    val_cut = timestamps[int(len(timestamps) * (train_frac + val_frac))]
    purge = pd.Timedelta(hours=purge_hours)
    train = df[df["timestamp"] < pd.Timestamp(train_cut) - purge]
    val = df[
        (df["timestamp"] >= train_cut)
        & (df["timestamp"] < pd.Timestamp(val_cut) - purge)
    ]
    test = df[df["timestamp"] >= val_cut]
    return train, val, test


def select_threshold(model, X, y) -> tuple[float, pd.DataFrame]:
    """Choose an operating threshold on validation data only."""
    prob = model.predict_proba(X)[:, 1]
    rows = []
    for threshold in [i / 100 for i in range(5, 96, 5)]:
        pred = (prob >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        rows.append({"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "alerts": int(pred.sum())})
    sweep = pd.DataFrame(rows)
    best = sweep.sort_values(["f1", "recall", "threshold"], ascending=[False, False, True]).iloc[0]
    return float(best["threshold"]), sweep


def evaluate(model, X, y, threshold: float = 0.5):
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {
        "roc_auc": float(roc_auc_score(y, prob)) if y.nunique() > 1 else 0.0,
        "pr_auc": float(average_precision_score(y, prob)) if y.nunique() > 1 else 0.0,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "classification_report": classification_report(y, pred, output_dict=True, zero_division=0),
    }


def main() -> None:
    config = load_config()
    model_config = config.get("model", {})
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/telemetry.csv")
    parser.add_argument("--processed", default="data/processed/features.parquet")
    parser.add_argument("--model-out", default="models/lightgbm_predictive_maintenance.joblib")
    parser.add_argument("--metrics-out", default="artifacts/test_metrics.json")
    parser.add_argument("--importance-out", default="artifacts/feature_importance.csv")
    parser.add_argument("--threshold-out", default="artifacts/threshold_sweep.csv")
    parser.add_argument("--shap-sample-size", type=int, default=1000)
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--horizon", type=int, default=config.get("failure_horizon_hours", 24))
    args = parser.parse_args()

    raw = pd.read_csv(args.input)
    data = build_features(raw, failure_horizon_hours=args.horizon)
    Path(args.processed).parent.mkdir(parents=True, exist_ok=True)
    if Path(args.processed).suffix.lower() == ".csv":
        data.to_csv(args.processed, index=False)
    else:
        data.to_parquet(args.processed, index=False)

    features = model_features(data)
    train, val, test = temporal_split(data, purge_hours=args.horizon)
    X_train, y_train = train[features], train["failure_within_horizon"]
    X_val, y_val = val[features], val["failure_within_horizon"]
    X_test, y_test = test[features], test["failure_within_horizon"]

    model = LGBMClassifier(
            n_estimators=model_config.get("n_estimators", 300),
            learning_rate=model_config.get("learning_rate", 0.05),
            num_leaves=model_config.get("num_leaves", 31),
            class_weight=model_config.get("class_weight", "balanced"),
            random_state=config.get("random_seed", 42),
            verbosity=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
    threshold, threshold_sweep = select_threshold(model, X_val, y_val)
    metrics = evaluate(model, X_test, y_test, threshold=threshold)
    metrics["threshold"] = threshold
    operational = test[["machine_id", "timestamp", "failure"]].copy()
    operational["alert"] = model.predict_proba(X_test)[:, 1] >= threshold
    metrics["operational_metrics"] = calculate_operational_metrics(
        operational, entity_col="machine_id", event_col="failure", horizon_hours=args.horizon
    )

    try:
        import mlflow
        import mlflow.lightgbm
    except ImportError:
        mlflow = None

    if mlflow is not None:
        mlflow.set_experiment("predictive-maintenance")
        with mlflow.start_run():
            for key in ["roc_auc", "pr_auc", "precision", "recall", "f1", "threshold"]:
                mlflow.log_metric(key, metrics[key])
            mlflow.log_params({"failure_horizon_hours": args.horizon, "n_features": len(features)})
            mlflow.lightgbm.log_model(
                model,
                artifact_path="model",
                input_example=X_train.head(5),
            )
    else:
        print("MLflow not installed; skipping experiment logging.")

    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": features,
        "threshold": threshold,
        "failure_horizon_hours": args.horizon,
        "model_version": "1.1.0",
    }, args.model_out)

    importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    Path(args.importance_out).parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(args.importance_out, index=False)
    Path(args.threshold_out).parent.mkdir(parents=True, exist_ok=True)
    threshold_sweep.to_csv(args.threshold_out, index=False)
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2))
    if not args.skip_shap:
        generate_shap_artifacts(
            model, X_test, output_dir=Path(args.metrics_out).parent,
            prefix="shap", sample_size=args.shap_sample_size,
            random_state=config.get("random_seed", 42),
            metadata=test[["timestamp", "machine_id", "failure_within_horizon"]],
        )
    print(json.dumps({k: metrics[k] for k in ["roc_auc", "pr_auc", "precision", "recall", "f1", "threshold"]}, indent=2))
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
