from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SAMPLE_RATE_HZ = 20_000
RUN_METADATA = {
    "1st_test": {"failed_bearings": {3, 4}, "channels": ((0, 1), (2, 3), (4, 5), (6, 7))},
    "2nd_test": {"failed_bearings": {1}, "channels": ((0,), (1,), (2,), (3,))},
    # The third archive calls this directory "4th_test" despite being Set No. 3.
    "4th_test": {"failed_bearings": {3}, "channels": ((0,), (1,), (2,), (3,))},
}

RELATIVE_FEATURES = (
    "rms_mean",
    "std_mean",
    "peak_mean",
    "kurtosis_mean",
    "crest_factor_mean",
    "spectral_centroid_hz_mean",
    "bandpower_0_1khz_mean",
    "bandpower_1_5khz_mean",
    "bandpower_5_10khz_mean",
)


def _channel_features(signal: np.ndarray, sample_rate_hz: int = SAMPLE_RATE_HZ) -> dict[str, float]:
    signal = np.asarray(signal, dtype=np.float64)
    centered = signal - signal.mean()
    std = centered.std()
    rms = np.sqrt(np.mean(signal**2))
    abs_mean = np.mean(np.abs(signal))
    peak = np.max(np.abs(signal))
    normalized = centered / std if std > 0 else np.zeros_like(centered)

    spectrum = np.abs(np.fft.rfft(centered)) ** 2
    frequencies = np.fft.rfftfreq(len(centered), d=1 / sample_rate_hz)
    spectrum[0] = 0
    total_power = spectrum.sum()
    spectral_centroid = float(np.dot(frequencies, spectrum) / total_power) if total_power else 0.0
    dominant_frequency = float(frequencies[np.argmax(spectrum)]) if total_power else 0.0

    def band_ratio(low: float, high: float) -> float:
        if not total_power:
            return 0.0
        mask = (frequencies >= low) & (frequencies < high)
        return float(spectrum[mask].sum() / total_power)

    return {
        "rms": float(rms),
        "std": float(std),
        "peak": float(peak),
        "peak_to_peak": float(np.ptp(signal)),
        "skew": float(np.mean(normalized**3)),
        "kurtosis": float(np.mean(normalized**4)),
        "crest_factor": float(peak / rms) if rms else 0.0,
        "impulse_factor": float(peak / abs_mean) if abs_mean else 0.0,
        "spectral_centroid_hz": spectral_centroid,
        "dominant_frequency_hz": dominant_frequency,
        "bandpower_0_1khz": band_ratio(0, 1_000),
        "bandpower_1_5khz": band_ratio(1_000, 5_000),
        "bandpower_5_10khz": band_ratio(5_000, 10_001),
    }


def bearing_features(channels: np.ndarray) -> dict[str, float]:
    """Produce a consistent schema for bearings with one or two accelerometers."""
    if channels.ndim == 1:
        channels = channels[:, None]
    per_channel = [_channel_features(channels[:, index]) for index in range(channels.shape[1])]
    output: dict[str, float] = {}
    for name in per_channel[0]:
        values = np.array([channel[name] for channel in per_channel])
        output[f"{name}_mean"] = float(values.mean())
        output[f"{name}_max"] = float(values.max())
    return output


def parse_snapshot_timestamp(path: Path) -> pd.Timestamp:
    return pd.to_datetime(path.name, format="%Y.%m.%d.%H.%M.%S")


def process_run(run_dir: Path, failure_horizon_hours: float = 24, max_files: int | None = None) -> pd.DataFrame:
    if run_dir.name not in RUN_METADATA:
        raise ValueError(f"Unknown IMS run directory: {run_dir.name}")
    metadata = RUN_METADATA[run_dir.name]
    files = sorted((path for path in run_dir.rglob("*") if path.is_file()), key=parse_snapshot_timestamp)
    run_end = parse_snapshot_timestamp(files[-1]) if files else None
    if max_files is not None and files:
        indices = np.linspace(0, len(files) - 1, min(max_files, len(files)), dtype=int)
        files = [files[index] for index in indices]
    if not files:
        raise ValueError(f"No snapshot files found under {run_dir}")
    assert run_end is not None
    rows = []
    for index, path in enumerate(files):
        snapshot = np.loadtxt(path, delimiter="\t", dtype=np.float64)
        timestamp = parse_snapshot_timestamp(path)
        hours_to_run_end = (run_end - timestamp).total_seconds() / 3600
        for bearing_id, channel_indices in enumerate(metadata["channels"], start=1):
            if max(channel_indices) >= snapshot.shape[1]:
                raise ValueError(f"{path} has {snapshot.shape[1]} channels; expected {max(channel_indices) + 1}")
            row = {
                "run_id": run_dir.name,
                "timestamp": timestamp,
                "snapshot_index": index,
                "bearing_id": bearing_id,
                "failed_at_run_end": bearing_id in metadata["failed_bearings"],
                "hours_to_run_end": hours_to_run_end,
                "failure_within_horizon": int(
                    bearing_id in metadata["failed_bearings"] and hours_to_run_end <= failure_horizon_hours
                ),
            }
            row.update(bearing_features(snapshot[:, channel_indices]))
            rows.append(row)
    return pd.DataFrame(rows)


def build_ims_features(root: str | Path, failure_horizon_hours: float = 24, max_files: int | None = None) -> pd.DataFrame:
    root = Path(root)
    frames = []
    for run_id in RUN_METADATA:
        matches = [path for path in root.rglob(run_id) if path.is_dir()]
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one {run_id} directory below {root}, found {len(matches)}")
        frames.append(process_run(matches[0], failure_horizon_hours, max_files=max_files))
    return add_cross_bearing_features(pd.concat(frames, ignore_index=True))


def add_cross_bearing_features(data: pd.DataFrame) -> pd.DataFrame:
    """Compare each bearing with its peers at the same timestamp.

    Median-relative features suppress vibration shared by the complete rig and
    emphasize the bearing that is unusually severe. Only contemporaneous sensor
    values are used, so these features are available at inference time.
    """
    output = data.copy()
    groups = output.groupby(["run_id", "timestamp"])
    for feature in RELATIVE_FEATURES:
        if feature not in output:
            raise ValueError(f"Missing feature required for cross-bearing comparison: {feature}")
        peer_median = groups[feature].transform("median")
        scale = peer_median.abs().clip(lower=1e-12)
        output[f"{feature}_vs_peer_median"] = output[feature] - peer_median
        output[f"{feature}_peer_ratio"] = output[feature] / scale
    return output


def ims_model_features(data: pd.DataFrame) -> list[str]:
    excluded = {
        "run_id", "timestamp", "snapshot_index", "bearing_id", "failed_at_run_end",
        "hours_to_run_end", "failure_within_horizon",
    }
    return [column for column in data.columns if column not in excluded]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract features from NASA IMS vibration snapshots")
    parser.add_argument("--input", default="data/external/ims/recordings")
    parser.add_argument("--output", default="data/processed/ims_features.parquet")
    parser.add_argument("--horizon", type=float, default=24)
    parser.add_argument("--max-files", type=int, help="Per-run limit for a quick smoke test")
    args = parser.parse_args()
    features = build_ims_features(args.input, args.horizon, args.max_files)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output, index=False)
    print(f"Saved {len(features):,} bearing snapshots to {output}")
    print(features.groupby("run_id")["failure_within_horizon"].agg(["count", "sum"]).to_string())


if __name__ == "__main__":
    main()
