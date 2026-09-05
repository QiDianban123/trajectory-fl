"""Integration checks for B's scaler and E's physical-coordinate plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.preprocess import TrainingCoordinateScaler
from src.evaluation.data_diagnostics import (
    plot_inverse_transform_check,
    plot_truth_trajectory,
)


def test_train_scaler_round_trip_feeds_meter_truth_plots(tmp_path: Path) -> None:
    coordinates_meter = np.array(
        [
            [10.0, 1.0],
            [12.0, 1.5],
            [14.0, 2.0],
            [16.0, 2.5],
            [18.0, 3.0],
        ],
        dtype=np.float32,
    )
    scaler = TrainingCoordinateScaler().fit(coordinates_meter, split="train")
    normalized = scaler.transform(coordinates_meter)
    restored_meter = scaler.inverse_transform(normalized)

    np.testing.assert_allclose(restored_meter, coordinates_meter, rtol=1e-5, atol=1e-5)
    truth_path = plot_truth_trajectory(
        restored_meter[:2],
        restored_meter[2:],
        tmp_path / "figures" / "truth.png",
        recording_id="01",
        vehicle_id=23,
    )
    inverse_path = plot_inverse_transform_check(
        coordinates_meter,
        restored_meter,
        tmp_path / "figures" / "inverse-transform.png",
        recording_id="01",
        vehicle_id=23,
    )

    assert truth_path.is_file()
    assert inverse_path.is_file()
