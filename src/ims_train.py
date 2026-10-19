from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier

from src.config import load_config
from src.explain import generate_shap_artifacts
from src.ims_features import ims_model_features
from src.train import calculate_operational_metrics, evaluate, select_threshold


RUN_SPLIT = {"train": "1st_test", "validation": "2nd_test", "test": "4th_test"}


def split_by_run(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing = set(RUN_SPLIT.values()) - set(data["run_id"].unique())
    if missing:
        raise ValueError(f"Missing IMS runs: {sorted(missing)}")
    return tuple(data[data["run_id"] == RUN_SPLIT[name]].copy() for name in ("train", "validation", "test"))


def operational_metrics(model, test: pd.DataFrame, features: list[str], threshold: float) -> dict:
    """Summarize alert timing and false alerts for the held-out run."""
    scored = test.copy()
    scored["alert"] = model.predict_proba(scored[features])[:, 1] >= threshold
    detections = []
    for bearing_id, bearing in scored[scored["failed_at_run_end"]].groupby("bearing_id"):
        in_horizon = bearing[bearing["failure_within_horizon"] == 1].sort_values("timestamp")
        alerts = in_horizon[in_horizon["alert"]]
        detections.append({
            "bearing_id": int(bearing_id),
            "detected": bool(len(alerts)),
            "first_alert_timestamp": alerts.iloc[0]["timestamp"].isoformat() if len(alerts) else None,
            "warning_hours": float(alerts.iloc[0]["hours_to_run_end"]) if len(alerts) else None,
            "alert_coverage": float(in_horizon["alert"].mean()) if len(in_horizon) else 0.0,
        })
    healthy = scored[~scored["failed_at_run_end"]]
    scored["failure_event"] = (
        scored["failed_at_run_end"]
        & scored["timestamp"].eq(scored.groupby("bearing_id")["timestamp"].transform("max"))
    )
    episode_metrics = calculate_operational_metrics(
        scored, entity_col="bearing_id", event_col="failure_event", horizon_hours=24
    )
    return {
        "failed_bearing_detections": detections,
        "non_failed_bearing_alert_rate": float(healthy["alert"].mean()),
        "non_failed_bearing_alerts": int(healthy["alert"].sum()),
        "non_failed_bearing_note": (
            "These component-level false positives may include vibration transmitted "
            "from the documented failed bearing through the shared test rig."
        ),
        "operational_metrics": episode_metrics,
    }


def main() -> None:
    config = load_config()
    ims_config = config.get("ims_model", {})
    parser = argparse.ArgumentParser(description="Train a run-held-out NASA IMS failure classifier")
    parser.add_argument("--features", default="data/processed/ims_features.parquet")
    parser.add_argument("--model-out", default="models/ims_lightgbm.joblib")
    parser.add_argument("--metrics-out", default="artifacts/ims_test_metrics.json")
    parser.add_argument("--importance-out", default="artifacts/ims_feature_importance.csv")
    parser.add_argument("--threshold-out", default="artifacts/ims_threshold_sweep.csv")
    parser.add_argument("--shap-sample-size", type=int, default=1000)
    parser.add_argument("--skip-shap", action="store_true")
    args = parser.parse_args()

    data = pd.read_parquet(args.features)
    features = ims_model_features(data)
    train, validation, test = split_by_run(data)
    model = LGBMClassifier(
        n_estimators=ims_config.get("n_estimators", 300),
        learning_rate=ims_config.get("learning_rate", 0.05),
        num_leaves=ims_config.get("num_leaves", 31),
        class_weight=ims_config.get("class_weight", "balanced"),
        random_state=config.get("random_seed", 42),
        verbosity=-1,
    )
    model.fit(train[features], train["failure_within_horizon"])
    threshold, sweep = select_threshold(model, validation[features], validation["failure_within_horizon"])
    metrics = evaluate(model, test[features], test["failure_within_horizon"], threshold)
    metrics.update({
        "threshold": threshold,
        "split": RUN_SPLIT,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
    })
    metrics.update(operational_metrics(model, test, features, threshold))

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model, "features": features, "threshold": threshold,
        "model_version": "ims-1.0.0", "run_split": RUN_SPLIT,
    }, model_path)
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    importance = pd.DataFrame({"feature": features, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
    Path(args.importance_out).parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(args.importance_out, index=False)
    Path(args.threshold_out).parent.mkdir(parents=True, exist_ok=True)
    sweep.to_csv(args.threshold_out, index=False)
    if not args.skip_shap:
        generate_shap_artifacts(
            model, test[features], output_dir=metrics_path.parent,
            prefix="ims_shap", sample_size=args.shap_sample_size,
            random_state=config.get("random_seed", 42),
            metadata=test[[
                "run_id", "timestamp", "bearing_id", "failed_at_run_end",
                "hours_to_run_end", "failure_within_horizon",
            ]],
        )
    print(json.dumps({key: metrics[key] for key in ("roc_auc", "pr_auc", "precision", "recall", "f1", "threshold")}, indent=2))


if __name__ == "__main__":
    main()
