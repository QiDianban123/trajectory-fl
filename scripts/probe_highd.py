"""Inspect highD CSV availability without downloading or mutating raw data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"id", "frame", "x", "y"}


def inspect_csv(path: Path) -> dict[str, object]:
    header = pd.read_csv(path, nrows=0)
    columns = set(header.columns)
    missing = sorted(REQUIRED_COLUMNS - columns)
    return {
        "file": path.name,
        "columns": sorted(columns),
        "required_columns_present": not missing,
        "missing_columns": missing,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check highD trajectory CSV headers")
    parser.add_argument("raw_dir", type=Path, nargs="?", default=Path("data/raw"))
    args = parser.parse_args()
    files = sorted(args.raw_dir.glob("*_tracks.csv"))
    if not files:
        print(f"No '*_tracks.csv' file found in {args.raw_dir}. This is expected before data download.")
        return 0
    for file in files:
        report = inspect_csv(file)
        print(f"{report['file']}: required columns={report['required_columns_present']}")
        if report["missing_columns"]:
            print("  missing: " + ", ".join(report["missing_columns"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
