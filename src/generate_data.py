from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import load_config


def generate_telemetry(n_machines: int = 50, n_days: int = 120, seed: int = 42) -> pd.DataFrame:
    """Generate hourly synthetic machine telemetry with degradation-driven failures."""
    rng = np.random.default_rng(seed)
    rows = []
    start = pd.Timestamp("2026-01-01 00:00:00")
    hours = n_days * 24

    machine_profiles = {}
    for machine_id in range(1, n_machines + 1):
        machine_profiles[machine_id] = {
            "base_temp": rng.normal(70, 3),
            "base_vibration": max(0.8, rng.normal(2.2, 0.35)),
            "base_pressure": rng.normal(100, 4),
            "base_rpm": rng.normal(1500, 80),
            "age_years": rng.uniform(0.5, 8.0),
        }

    for machine_id, p in machine_profiles.items():
        degradation = rng.uniform(0.0, 0.15)
        hours_since_maintenance = int(rng.integers(0, 300))

        for h in range(hours):
            ts = start + pd.Timedelta(hours=h)
            load = np.clip(rng.normal(0.65, 0.18), 0.15, 1.0)
            degradation += 0.00045 * load + rng.normal(0, 0.0008)
            degradation = max(0.0, degradation)
            hours_since_maintenance += 1

            temperature = p["base_temp"] + 12 * load + 17 * degradation + rng.normal(0, 1.5)
            vibration = p["base_vibration"] + 2.2 * load + 5.2 * degradation + rng.normal(0, 0.25)
            pressure = p["base_pressure"] - 7 * degradation + rng.normal(0, 2.0)
            rpm = p["base_rpm"] + 180 * load - 90 * degradation + rng.normal(0, 35)
            power_kw = 18 + 20 * load + 8 * degradation + rng.normal(0, 1.2)

            risk_logit = (
                -9.0
                + 5.2 * degradation
                + 0.09 * max(temperature - 85, 0)
                + 0.55 * max(vibration - 5.5, 0)
                + 0.0025 * max(hours_since_maintenance - 500, 0)
                + 0.10 * p["age_years"]
            )
            failure_prob = 1 / (1 + np.exp(-risk_logit))
            failure = int(rng.random() < failure_prob)

            rows.append({
                "timestamp": ts,
                "machine_id": f"M{machine_id:03d}",
                "temperature_c": round(temperature, 3),
                "vibration_mm_s": round(vibration, 3),
                "pressure_bar": round(pressure, 3),
                "rpm": round(rpm, 2),
                "power_kw": round(power_kw, 3),
                "load_pct": round(load * 100, 2),
                "machine_age_years": round(p["age_years"], 2),
                "hours_since_maintenance": hours_since_maintenance,
                "failure": failure,
            })

            if failure:
                degradation = rng.uniform(0.0, 0.08)
                hours_since_maintenance = 0

    return pd.DataFrame(rows)


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--machines", type=int, default=config.get("n_machines", 50))
    parser.add_argument("--days", type=int, default=config.get("n_days", 120))
    parser.add_argument("--seed", type=int, default=config.get("random_seed", 42))
    parser.add_argument("--output", type=str, default="data/raw/telemetry.csv")
    args = parser.parse_args()

    df = generate_telemetry(args.machines, args.days, args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} rows to {out}")
    print(f"Observed failures: {df['failure'].sum():,}")


if __name__ == "__main__":
    main()
