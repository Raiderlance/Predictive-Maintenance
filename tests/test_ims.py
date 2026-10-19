import numpy as np
import pandas as pd

from src.ims_features import add_cross_bearing_features, bearing_features, build_ims_features, ims_model_features
from src.ims_train import split_by_run


def test_bearing_features_are_finite_for_constant_signal():
    features = bearing_features(np.ones((128, 1)))
    assert features["rms_mean"] == 1.0
    assert features["std_mean"] == 0.0
    assert all(np.isfinite(value) for value in features.values())


def test_build_features_uses_documented_failed_bearings(tmp_path):
    layouts = {"1st_test": 8, "2nd_test": 4, "4th_test": 4}
    for run_id, columns in layouts.items():
        run = tmp_path / run_id
        run.mkdir()
        for timestamp in ("2026.01.01.00.00.00", "2026.01.01.02.00.00"):
            np.savetxt(run / timestamp, np.ones((32, columns)), delimiter="\t")

    data = build_ims_features(tmp_path, failure_horizon_hours=1)
    final = data[data["timestamp"] == pd.Timestamp("2026-01-01 02:00:00")]
    positives = set(zip(
        final.loc[final["failure_within_horizon"] == 1, "run_id"],
        final.loc[final["failure_within_horizon"] == 1, "bearing_id"],
    ))
    assert positives == {("1st_test", 3), ("1st_test", 4), ("2nd_test", 1), ("4th_test", 3)}
    assert "hours_to_run_end" not in ims_model_features(data)
    assert "bearing_id" not in ims_model_features(data)


def test_split_holds_out_complete_runs():
    data = pd.DataFrame({"run_id": ["1st_test", "2nd_test", "4th_test"], "value": [1, 2, 3]})
    train, validation, test = split_by_run(data)
    assert train["run_id"].unique().tolist() == ["1st_test"]
    assert validation["run_id"].unique().tolist() == ["2nd_test"]
    assert test["run_id"].unique().tolist() == ["4th_test"]


def test_cross_bearing_features_isolate_abnormal_bearing():
    data = pd.DataFrame({
        "run_id": ["4th_test"] * 4,
        "timestamp": pd.to_datetime(["2026-01-01"] * 4),
    })
    relative_names = (
        "rms_mean", "std_mean", "peak_mean", "kurtosis_mean", "crest_factor_mean",
        "spectral_centroid_hz_mean", "bandpower_0_1khz_mean", "bandpower_1_5khz_mean",
        "bandpower_5_10khz_mean",
    )
    for name in relative_names:
        data[name] = [1.0, 1.0, 4.0, 1.0]

    output = add_cross_bearing_features(data)

    assert output.loc[2, "rms_mean_peer_ratio"] == 4.0
    assert output.loc[2, "rms_mean_vs_peer_median"] == 3.0
    assert output.loc[0, "rms_mean_peer_ratio"] == 1.0
