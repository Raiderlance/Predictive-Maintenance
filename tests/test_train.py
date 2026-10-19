import pandas as pd

from src.train import temporal_split


def test_temporal_split_purges_labels_crossing_boundaries():
    timestamps = pd.date_range("2026-01-01", periods=100, freq="h")
    df = pd.DataFrame({"timestamp": timestamps})

    train, validation, test = temporal_split(df, purge_hours=4)

    train_cut = timestamps[70]
    validation_cut = timestamps[85]
    assert train["timestamp"].max() < train_cut - pd.Timedelta(hours=4)
    assert validation["timestamp"].min() == train_cut
    assert validation["timestamp"].max() < validation_cut - pd.Timedelta(hours=4)
    assert test["timestamp"].min() == validation_cut


def test_temporal_split_rejects_too_few_timestamps():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01", "2026-01-02"])})
    try:
        temporal_split(df)
    except ValueError as error:
        assert "three unique timestamps" in str(error)
    else:
        raise AssertionError("Expected temporal_split to reject insufficient data")
