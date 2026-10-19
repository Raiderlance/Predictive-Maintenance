from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd

from src.scoring import risk_band


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--model", default="models/lightgbm_predictive_maintenance.joblib")
    parser.add_argument("--output", default="data/processed/scored_machines.csv")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model, features = bundle["model"], bundle["features"]
    threshold = float(bundle.get("threshold", 0.5))
    df = (pd.read_csv(args.features) if Path(args.features).suffix.lower() == ".csv" else pd.read_parquet(args.features)).sort_values("timestamp")
    latest = df.groupby("machine_id", as_index=False).tail(1).copy()
    latest["failure_risk"] = model.predict_proba(latest[features])[:, 1]
    latest["alert"] = latest["failure_risk"] >= threshold
    latest["risk_band"] = latest["failure_risk"].map(lambda risk: risk_band(risk, threshold))
    latest["alert_threshold"] = threshold
    out_cols = ["timestamp", "machine_id", "failure_risk", "risk_band", "alert", "alert_threshold", "temperature_c", "vibration_mm_s", "hours_since_maintenance"]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    latest[out_cols].sort_values("failure_risk", ascending=False).to_csv(args.output, index=False)
    print(latest[out_cols].sort_values("failure_risk", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
