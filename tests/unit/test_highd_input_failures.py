"""Standard-sample and frame-validator tests; not raw highD adapter acceptance."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from src.data.adapters import TrajectorySample
from src.data.dataset import TrajectoryDataset
from src.data.preprocess import WindowSpec, validate_strictly_increasing_frames


def _metadata(**overrides: object) -> dict[str, object]:
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
    meta.update(overrides)
    return meta


def _sample_with_coordinates(history: np.ndarray, future: np.ndarray) -> TrajectorySample:
    return TrajectorySample(history=history, future=future, meta=_metadata())


# ---------------------------------------------------------------------------
# REQ-FAIL-01: missing required fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["recording_id", "vehicle_id", "split", "split_id"])
def test_missing_sample_metadata_is_rejected(
    field: str,
    valid_trajectory_sample: TrajectorySample,
) -> None:
    meta = dict(valid_trajectory_sample.meta)
    del meta[field]
    with pytest.raises(ValueError, match=field):
        TrajectorySample(valid_trajectory_sample.history, valid_trajectory_sample.future, meta)


# ---------------------------------------------------------------------------
# REQ-FAIL-02: NaN / Inf coordinates
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), -float("inf")], ids=["nan", "inf", "-inf"]
)
def test_nonfinite_coordinate_is_rejected(axis: str, value: float) -> None:
    """TrajectorySample must reject NaN/Inf coordinates with an explicit error."""

    history = np.array([[0.0, 0.0], [1.0, 0.5]], dtype=np.float32)
    future = np.array([[2.0, 1.0], [3.0, 1.5], [4.0, 2.0]], dtype=np.float32)
    if axis == "x":
        future[1, 0] = np.float32(value)
    else:
        future[1, 1] = np.float32(value)

    with pytest.raises(ValueError, match="finite"):
        _sample_with_coordinates(history, future)


# ---------------------------------------------------------------------------
# REQ-FAIL-03: short trajectories
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("part", ["history", "future"])
def test_short_sample_window_is_rejected(
    part: str,
    valid_trajectory_sample: TrajectorySample,
) -> None:
    sample = valid_trajectory_sample
    malformed = TrajectorySample(
        sample.history[:-1] if part == "history" else sample.history,
        sample.future[:-1] if part == "future" else sample.future,
        sample.meta,
    )
    with pytest.raises(ValueError, match=f"sample {part} shape"):
        TrajectoryDataset(
            [malformed],
            split="train",
            split_id="highd-split-42",
            window_spec=WindowSpec(history_steps=2, future_steps=3, stride=1),
        )


# ---------------------------------------------------------------------------
# REQ-FAIL-04: duplicate frames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", ["adjacent", "non_adjacent"])
def test_duplicate_frame_is_rejected(
    pattern: str,
    duplicate_frame_track: Callable[[str], list[dict[str, object]]],
) -> None:
    """Duplicate frame values must raise before window construction."""

    track = duplicate_frame_track(pattern)
    frames = [int(record["frame"]) for record in track]
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_strictly_increasing_frames(frames)


# ---------------------------------------------------------------------------
# REQ-FAIL-05: out-of-order frames
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", ["adjacent", "global"])
def test_out_of_order_frame_is_rejected(
    pattern: str,
    out_of_order_frame_track: Callable[[str], list[dict[str, object]]],
) -> None:
    """Out-of-order frame values must raise before window construction."""

    track = out_of_order_frame_track(pattern)
    frames = [int(record["frame"]) for record in track]
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_strictly_increasing_frames(frames)
