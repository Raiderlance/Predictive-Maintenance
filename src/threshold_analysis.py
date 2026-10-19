from __future__ import annotations

import argparse
from pathlib import Path
import joblib
import pandas as pd

from src.train import select_threshold, temporal_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features.parquet")
    parser.add_argument("--model", default="models/lightgbm_predictive_maintenance.joblib")
    parser.add_argument("--output", default="artifacts/threshold_sweep.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.features) if Path(args.features).suffix.lower() == ".csv" else pd.read_parquet(args.features)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    bundle = joblib.load(args.model)
    horizon = int(bundle.get("failure_horizon_hours", 24))
    _, validation, _ = temporal_split(df, purge_hours=horizon)
    best, out = select_threshold(
        bundle["model"], validation[bundle["features"]], validation["failure_within_horizon"].astype(int)
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Best validation threshold: {best:.2f}")
    print(out.sort_values("f1", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
