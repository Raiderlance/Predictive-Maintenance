from __future__ import annotations

import argparse
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import py7zr


IMS_URL = "https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip"


def download_and_extract(destination: str | Path) -> Path:
    """Download the official archive and unpack its ZIP -> 7z -> RAR nesting."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    zip_path = destination / "IMS.zip"
    if not zip_path.exists():
        print(f"Downloading about 1 GB from {IMS_URL}")
        urllib.request.urlretrieve(IMS_URL, zip_path)

    seven_zip_path = destination / "4. Bearings" / "IMS.7z"
    if not seven_zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(destination)

    archive_names = ("1st_test.rar", "2nd_test.rar", "3rd_test.rar")
    candidates = (destination / "nested_archives", destination / "extracted")
    nested = next(
        (candidate for candidate in candidates if all((candidate / name).exists() for name in archive_names)),
        candidates[0],
    )
    nested.mkdir(exist_ok=True)
    rar_paths = [nested / name for name in archive_names]
    if not all(path.exists() for path in rar_paths):
        with py7zr.SevenZipFile(seven_zip_path) as archive:
            archive.extractall(nested)

    tar = shutil.which("tar")
    if tar is None:
        raise RuntimeError("A bsdtar-compatible 'tar' executable is required to unpack NASA's RAR files")
    recordings = destination / "recordings"
    recordings.mkdir(exist_ok=True)
    expected_dirs = (recordings / "1st_test", recordings / "2nd_test", recordings / "4th_test")
    for rar_path, expected_dir in zip(rar_paths, expected_dirs):
        if not expected_dir.exists():
            subprocess.run([tar, "-xf", str(rar_path), "-C", str(recordings)], check=True)
    print(f"IMS recordings available at {recordings}")
    return recordings


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and extract the official NASA IMS Bearings dataset")
    parser.add_argument("--destination", default="data/external/ims")
    args = parser.parse_args()
    download_and_extract(args.destination)


if __name__ == "__main__":
    main()
