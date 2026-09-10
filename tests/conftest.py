"""Small, dependency-light fixtures shared by Day 2 unit and integration tests.

S1-F-01 extends this module with valid and invalid highD record/track/sample
factories so data-quality tests can construct edge cases without touching real
raw files. All factories return fresh objects and never share mutable state.
"""

from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.data.adapters import TrajectorySample
from src.utils.config import load_yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_trajectory_pair() -> tuple[np.ndarray, np.ndarray]:
    truth = np.array([[[0.0, 0.0], [3.0, 4.0]]])
    prediction = np.array([[[0.0, 0.0], [0.0, 0.0]]])
    return prediction, truth


@pytest.fixture
def tiny_trajectory_triplet() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical-coordinate history, truth, and prediction suitable for plotting."""

    history = np.array([[-1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    future = np.array([[1.0, 0.0], [2.0, 1.0]], dtype=np.float32)
    prediction = np.array([[1.0, 0.2], [2.0, 1.2]], dtype=np.float32)
    return history, future, prediction


@pytest.fixture
def config_bundle() -> dict[str, dict[str, object]]:
    """Independent copies of the repository's validated configuration sources."""

    paths = {
        "data": PROJECT_ROOT / "configs/data.yaml",
        "model": PROJECT_ROOT / "configs/model.yaml",
        "experiment": PROJECT_ROOT / "configs/experiments/smoke.yaml",
    }
    return {name: deepcopy(load_yaml(path)) for name, path in paths.items()}


@pytest.fixture
def temporary_output_root(tmp_path: Path) -> Path:
    """An isolated output root for path, logging, and result-store tests."""

    return tmp_path / "outputs"


# ---------------------------------------------------------------------------
# S1-F-01: valid highD inputs
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_highd_record() -> dict[str, object]:
    """A single highD record with all required fields and finite coordinates."""

    return {
        "id": 101,
        "frame": 0,
        "x": 0.0,
        "y": 0.0,
        "vehicle_id": 101,
        "scenario_id": 1,
    }


@pytest.fixture
def valid_highd_track() -> Callable[..., list[dict[str, object]]]:
    """Factory returning a strictly increasing, float32 highD track.

    The returned track has ``num_frames`` rows with frame ``[0, 1, ...]``, x
    increasing by 1.0 per frame and y increasing by 0.5 per frame. ``id`` is
    fixed to 101 so multiple calls produce independent but comparable tracks.
    """

    def _build(num_frames: int = 250) -> list[dict[str, object]]:
        if num_frames <= 0:
            raise ValueError("num_frames must be a positive integer")
        return [
            {
                "id": 101,
                "frame": frame,
                "x": np.float32(frame),
                "y": np.float32(0.5 * frame),
                "vehicle_id": 101,
                "scenario_id": 1,
            }
            for frame in range(num_frames)
        ]

    return _build


@pytest.fixture
def valid_trajectory_sample() -> TrajectorySample:
    """A TrajectorySample that satisfies the Day 2 data contract."""

    history = np.array([[0.0, 0.0], [1.0, 0.5]], dtype=np.float32)
    future = np.array([[2.0, 1.0], [3.0, 1.5], [4.0, 2.0]], dtype=np.float32)
    meta: dict[str, object] = {
        "dataset_name": "highd",
        "data_version": "sample-v1",
        "recording_id": 1,
        "vehicle_id": 101,
        "history_start_frame": 0,
        "history_end_frame": 1,
        "future_start_frame": 2,
        "future_end_frame": 4,
        "split_id": "highd-split-42",
        "split": "train",
    }
    return TrajectorySample(history=history, future=future, meta=meta)


# ---------------------------------------------------------------------------
# S1-F-01: invalid highD inputs (factories only; no assertions inside)
# ---------------------------------------------------------------------------


@pytest.fixture
def invalid_highd_record(
    valid_highd_record: dict[str, object],
) -> Callable[[str], dict[str, object]]:
    """Factory returning a highD record missing the requested field."""

    def _build(missing_field: str) -> dict[str, object]:
        if missing_field not in valid_highd_record:
            raise KeyError(f"unknown field {missing_field!r}")
        record = dict(valid_highd_record)
        record.pop(missing_field)
        return record

    return _build


@pytest.fixture
def nonfinite_coordinate_record(
    valid_highd_record: dict[str, object],
) -> Callable[[str, float], dict[str, object]]:
    """Factory returning a highD record with a NaN/Inf value on the given axis."""

    def _build(axis: str, value: float) -> dict[str, object]:
        if axis not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        record = dict(valid_highd_record)
        record[axis] = value
        return record

    return _build


@pytest.fixture
def short_trajectory_record() -> Callable[[int], list[dict[str, object]]]:
    """Factory returning a highD track with fewer than the required frames."""

    def _build(num_frames: int) -> list[dict[str, object]]:
        if num_frames <= 0:
            raise ValueError("num_frames must be a positive integer")
        return [
            {
                "id": 101,
                "frame": frame,
                "x": np.float32(frame),
                "y": np.float32(0.5 * frame),
                "vehicle_id": 101,
                "scenario_id": 1,
            }
            for frame in range(num_frames)
        ]

    return _build


@pytest.fixture
def duplicate_frame_track() -> Callable[[str], list[dict[str, object]]]:
    """Factory returning a highD track with duplicate frame values.

    ``pattern="adjacent"`` produces ``[0, 0, 1, 2, ...]`` while
    ``pattern="non_adjacent"`` produces ``[0, 1, 0, 2, 3, ...]``.
    """

    def _build(pattern: str) -> list[dict[str, object]]:
        if pattern == "adjacent":
            frames = [0, 0, 1, 2, 3]
        elif pattern == "non_adjacent":
            frames = [0, 1, 0, 2, 3]
        else:
            raise ValueError("pattern must be 'adjacent' or 'non_adjacent'")
        return [
            {
                "id": 101,
                "frame": frame,
                "x": np.float32(frame),
                "y": np.float32(0.5 * frame),
                "vehicle_id": 101,
                "scenario_id": 1,
            }
            for frame in frames
        ]

    return _build


@pytest.fixture
def out_of_order_frame_track() -> Callable[[str], list[dict[str, object]]]:
    """Factory returning a highD track with out-of-order frame values.

    ``pattern="adjacent"`` produces ``[0, 2, 1, 3, 4, ...]`` while
    ``pattern="global"`` produces ``[4, 3, 2, 1, 0]``.
    """

    def _build(pattern: str) -> list[dict[str, object]]:
        if pattern == "adjacent":
            frames = [0, 2, 1, 3, 4]
        elif pattern == "global":
            frames = [4, 3, 2, 1, 0]
        else:
            raise ValueError("pattern must be 'adjacent' or 'global'")
        return [
            {
                "id": 101,
                "frame": frame,
                "x": np.float32(frame),
                "y": np.float32(0.5 * frame),
                "vehicle_id": 101,
                "scenario_id": 1,
            }
            for frame in frames
        ]

    return _build
