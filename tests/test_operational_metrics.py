import pandas as pd

from src.train import calculate_operational_metrics


def test_operational_metrics_group_rows_into_episodes_and_events():
    timestamps = pd.date_range("2026-01-01", periods=9, freq="h")
    scored = pd.DataFrame({
        "machine_id": ["M001"] * 9,
        "timestamp": timestamps,
        "alert": [False, True, True, False, False, True, False, False, False],
        "failure": [False, False, False, False, False, False, False, False, True],
    })

    result = calculate_operational_metrics(
        scored, entity_col="machine_id", event_col="failure", horizon_hours=4
    )

    assert result["eligible_failure_events"] == 1
    assert result["detected_failure_events"] == 1
    assert result["event_detection_rate"] == 1.0
    assert result["mean_lead_time_hours"] == 3.0
    assert result["mean_failure_window_coverage"] == 0.25
    assert result["alert_episodes"] == 2
    assert result["false_alert_episodes"] == 1
    assert result["mean_alert_persistence_hours"] == 1.5


def test_operational_metrics_excludes_incomplete_boundary_window():
    scored = pd.DataFrame({
        "machine_id": ["M001"] * 3,
        "timestamp": pd.date_range("2026-01-01", periods=3, freq="h"),
        "alert": [True, False, False],
        "failure": [False, False, True],
    })

    result = calculate_operational_metrics(
        scored, entity_col="machine_id", event_col="failure", horizon_hours=4
    )

    assert result["failure_events"] == 1
    assert result["eligible_failure_events"] == 0
    assert result["boundary_failure_events_excluded"] == 1
