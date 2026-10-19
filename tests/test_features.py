import pandas as pd
from src.features import build_features, model_features


def test_build_features_creates_target_and_rolling_columns():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=30, freq="h"),
        "machine_id": ["M001"] * 30,
        "temperature_c": [70.0] * 30,
        "vibration_mm_s": [2.0] * 30,
        "pressure_bar": [100.0] * 30,
        "rpm": [1500.0] * 30,
        "power_kw": [20.0] * 30,
        "load_pct": [50.0] * 30,
        "machine_age_years": [2.0] * 30,
        "hours_since_maintenance": list(range(30)),
        "failure": [0] * 29 + [1],
    })
    out = build_features(df, failure_horizon_hours=24)
    assert "temperature_c_roll24_mean" in out.columns
    assert "failure_within_horizon" in out.columns
    assert out.loc[5, "failure_within_horizon"] == 1
    assert "machine_id" not in model_features(out)


def test_target_excludes_current_failure_and_respects_horizon():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=6, freq="h"),
        "machine_id": ["M001"] * 6,
        "temperature_c": [70.0] * 6,
        "vibration_mm_s": [2.0] * 6,
        "pressure_bar": [100.0] * 6,
        "rpm": [1500.0] * 6,
        "power_kw": [20.0] * 6,
        "load_pct": [50.0] * 6,
        "machine_age_years": [2.0] * 6,
        "hours_since_maintenance": list(range(6)),
        "failure": [0, 0, 0, 1, 0, 0],
    })
    out = build_features(df, failure_horizon_hours=2)
    assert out["failure_within_horizon"].tolist() == [0, 1, 1, 0, 0, 0]


def test_features_do_not_mix_machines():
    base = pd.DataFrame({
        "timestamp": list(pd.date_range("2026-01-01", periods=2, freq="h")) * 2,
        "machine_id": ["M001", "M001", "M002", "M002"],
        "temperature_c": [10.0, 20.0, 100.0, 120.0],
        "vibration_mm_s": [2.0] * 4,
        "pressure_bar": [100.0] * 4,
        "rpm": [1500.0] * 4,
        "power_kw": [20.0] * 4,
        "load_pct": [50.0] * 4,
        "machine_age_years": [2.0] * 4,
        "hours_since_maintenance": [0, 1, 0, 1],
        "failure": [0, 0, 0, 0],
    })
    out = build_features(base)
    first_m002 = out[out["machine_id"] == "M002"].iloc[0]
    assert first_m002["temperature_c_roll6_mean"] == 100.0
    assert first_m002["temperature_c_delta1"] == 0.0
