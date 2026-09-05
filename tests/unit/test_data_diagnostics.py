"""Unit tests for E's S1 data-diagnostic plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

import src.evaluation.data_diagnostics as diagnostics
from src.evaluation.data_diagnostics import (
    diagnostic_file_stem,
    generate_data_diagnostic_figures,
    plot_anomaly_counts,
    plot_inverse_transform_check,
    plot_raw_cleaned_trajectories,
    plot_truth_trajectory,
)


def _trajectory() -> np.ndarray:
    return np.array(
        [[0.0, 0.0], [1.0, 0.2], [2.0, 0.4], [3.0, 0.7]],
        dtype=np.float32,
    )


def test_raw_cleaned_plot_creates_parent_directory(tmp_path: Path) -> None:
    output = plot_raw_cleaned_trajectories(
        _trajectory(),
        _trajectory()[1:],
        tmp_path / "nested" / "raw-cleaned.png",
        recording_id="01",
        vehicle_id=7,
    )
    assert output == tmp_path / "nested" / "raw-cleaned.png"
    assert output.is_file()
    assert output.stat().st_size > 0


def test_anomaly_plot_sorts_labels_and_rejects_invalid_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_close = plt.close
    monkeypatch.setattr(diagnostics.plt, "close", lambda _figure: None)
    output = plot_anomaly_counts(
        {"short_track": 2, "missing_required": 1},
        tmp_path / "counts.png",
    )
    figure = diagnostics.plt.gcf()
    labels = [tick.get_text() for tick in figure.axes[0].get_xticklabels()]
    assert labels == ["missing_required", "short_track"]
    assert output.is_file()
    original_close(figure)

    with pytest.raises(ValueError, match="non-empty mapping"):
        plot_anomaly_counts({}, tmp_path / "empty.png")
    with pytest.raises(ValueError, match="non-negative integers"):
        plot_anomaly_counts({"missing_required": -1}, tmp_path / "negative.png")


def test_truth_plot_uses_meter_axes_and_history_future_legend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_close = plt.close
    monkeypatch.setattr(diagnostics.plt, "close", lambda _figure: None)
    output = plot_truth_trajectory(
        _trajectory()[:2],
        _trajectory()[2:],
        tmp_path / "truth.png",
        recording_id=1,
        vehicle_id=2,
    )
    figure = diagnostics.plt.gcf()
    axis = figure.axes[0]
    assert axis.get_xlabel() == "x (m)"
    assert axis.get_ylabel() == "y (m)"
    assert [line.get_label() for line in axis.get_lines()] == ["history", "future truth"]
    assert axis.get_legend() is not None
    assert output.is_file()
    original_close(figure)


@pytest.mark.parametrize(
    ("values", "error_type", "message"),
    [
        (np.empty((0, 2), dtype=np.float32), ValueError, "shape"),
        (np.zeros((2, 3), dtype=np.float32), ValueError, "shape"),
        (np.array([[0.0, np.nan]], dtype=np.float32), ValueError, "finite"),
        (np.zeros((2, 2), dtype=np.int64), TypeError, "floating-point"),
    ],
)
def test_truth_plot_rejects_invalid_trajectory_arrays(
    values: np.ndarray,
    error_type: type[Exception],
    message: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(error_type, match=message):
        plot_truth_trajectory(values, _trajectory(), tmp_path / "truth.png")


def test_truth_plot_rejects_normalized_coordinates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="physical coordinates in meters"):
        plot_truth_trajectory(
            _trajectory()[:2],
            _trajectory()[2:],
            tmp_path / "truth.png",
            coordinate_unit="normalized",
        )


def test_inverse_transform_plot_requires_matching_shapes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        plot_inverse_transform_check(
            _trajectory(),
            _trajectory()[:2],
            tmp_path / "inverse.png",
        )


def test_complete_diagnostic_set_uses_deterministic_names(tmp_path: Path) -> None:
    values = _trajectory()
    arguments = {
        "raw_xy": values,
        "cleaned_xy": values,
        "anomaly_counts": {"missing_required": 0},
        "history_meter": values[:2],
        "future_meter": values[2:],
        "original_meter": values,
        "restored_meter": values.copy(),
        "output_dir": tmp_path / "figures" / "data",
        "recording_id": "01",
        "vehicle_id": 23,
    }
    first = generate_data_diagnostic_figures(**arguments)
    second = generate_data_diagnostic_figures(**arguments)
    assert first == second
    assert set(first) == {"raw_cleaned", "anomaly_counts", "truth", "inverse_transform"}
    assert {path.name for path in first.values()} == {
        "recording_01_vehicle_23_raw_cleaned.png",
        "recording_01_vehicle_23_anomaly_counts.png",
        "recording_01_vehicle_23_truth.png",
        "recording_01_vehicle_23_inverse_transform.png",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in first.values())


def test_diagnostic_file_stem_is_stable_and_path_safe() -> None:
    assert diagnostic_file_stem(recording_id="01", vehicle_id=23) == (
        "recording_01_vehicle_23"
    )
    assert diagnostic_file_stem(recording_id="../01", vehicle_id="vehicle / 23") == (
        "recording_01_vehicle_vehicle-23"
    )


def test_output_path_requires_a_file_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="filename with an extension"):
        plot_truth_trajectory(
            _trajectory()[:2],
            _trajectory()[2:],
            tmp_path / "no-extension",
        )
