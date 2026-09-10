"""Rebuild the S1-E data-diagnostic figures from a deterministic legal sample."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/s1-e-data-plots/figures/data"),
        help="Directory in which the four deterministic PNG files are written.",
    )
    return parser


def _make_project_importable() -> None:
    """Allow this repository script to run directly from any working directory."""

    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def main(argv: list[str] | None = None) -> int:
    _make_project_importable()
    from src.data.preprocess import TrainingCoordinateScaler
    from src.evaluation.data_diagnostics import generate_data_diagnostic_figures

    args = build_parser().parse_args(argv)

    raw_xy = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.1],
            [2.0, 0.2],
            [3.0, 1.8],
            [4.0, 0.4],
            [5.0, 0.5],
            [6.0, 0.7],
            [7.0, 0.9],
        ],
        dtype=np.float32,
    )
    cleaned_xy = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.1],
            [2.0, 0.2],
            [4.0, 0.4],
            [5.0, 0.5],
            [6.0, 0.7],
            [7.0, 0.9],
        ],
        dtype=np.float32,
    )
    anomaly_counts = {
        "duplicate_frame": 0,
        "missing_required": 0,
        "nonfinite_coordinate": 0,
        "outlier_removed": 1,
        "short_track": 0,
    }

    scaler = TrainingCoordinateScaler().fit(cleaned_xy, split="train")
    normalized = scaler.transform(cleaned_xy)
    restored_meter = scaler.inverse_transform(normalized)
    split_at = 3

    artifacts = generate_data_diagnostic_figures(
        raw_xy=raw_xy,
        cleaned_xy=cleaned_xy,
        anomaly_counts=anomaly_counts,
        history_meter=restored_meter[:split_at],
        future_meter=restored_meter[split_at:],
        original_meter=cleaned_xy,
        restored_meter=restored_meter,
        output_dir=args.output_dir,
        recording_id="demo-01",
        vehicle_id=1,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
