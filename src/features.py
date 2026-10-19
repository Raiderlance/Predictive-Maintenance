from __future__ import annotations

import pandas as pd

SENSOR_COLS = ["temperature_c", "vibration_mm_s", "pressure_bar", "rpm", "power_kw", "load_pct"]


def build_features(df: pd.DataFrame, failure_horizon_hours: int = 24) -> pd.DataFrame:
    """Create rolling telemetry features and a future-failure classification target."""
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)

    grouped = data.groupby("machine_id", group_keys=False)
    for col in SENSOR_COLS:
        data[f"{col}_roll6_mean"] = grouped[col].transform(lambda s: s.rolling(6, min_periods=1).mean())
        data[f"{col}_roll24_mean"] = grouped[col].transform(lambda s: s.rolling(24, min_periods=1).mean())
        data[f"{col}_roll24_std"] = grouped[col].transform(lambda s: s.rolling(24, min_periods=2).std()).fillna(0)
        data[f"{col}_delta1"] = grouped[col].diff().fillna(0)

    # target = whether any failure occurs in the next N hours, excluding current row
    def future_failure(s: pd.Series) -> pd.Series:
        rev = s.shift(-1)[::-1]
        target = rev.rolling(failure_horizon_hours, min_periods=1).max()[::-1]
        return target.fillna(0).astype(int)

    data["failure_within_horizon"] = grouped["failure"].transform(future_failure)
    data["hour"] = data["timestamp"].dt.hour
    data["dayofweek"] = data["timestamp"].dt.dayofweek
    return data


def model_features(df: pd.DataFrame) -> list[str]:
    excluded = {"timestamp", "machine_id", "failure", "failure_within_horizon"}
    return [c for c in df.columns if c not in excluded]
