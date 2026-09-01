"""Unit tests for the Day 2 data-layer contracts without raw highD files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.adapters import DatasetAdapter, TrajectorySample
from src.data.dataset import TrajectoryDataset
from src.data.preprocess import (
    TrainingCoordinateScaler,
    WindowSpec,
    validate_split_assignments,
    validate_strictly_increasing_frames,
    validate_window_order,
)


def _metadata(**overrides: object) -> dict[str, object]:
    meta: dict[str, object] = {
        "dataset_name": "highd",
        "data_version": "sample-v1",
        "recording_id": 1,
        "vehicle_id": 101,
        "history_start_frame": 10,
        "history_end_frame": 11,
        "future_start_frame": 12,
        "future_end_frame": 14,
        "split_id": "highd-split-42",
        "split": "train",
    }
    meta.update(overrides)
    return meta


def _sample(**metadata_overrides: object) -> TrajectorySample:
    return TrajectorySample(
        history=np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32),
        future=np.array([[2.0, 3.0], [3.0, 4.0], [4.0, 5.0]], dtype=np.float32),
        meta=_metadata(**metadata_overrides),
    )


def test_standard_sample_requires_fixed_float32_coordinates_and_metadata() -> None:
    sample = _sample()
    assert sample.history.shape == (2, 2)
    assert sample.future.shape == (3, 2)
    assert sample.meta["split"] == "train"

    with pytest.raises(TypeError, match="float32"):
        TrajectorySample(
            history=np.array([[0.0, 1.0]], dtype=np.float64),
            future=np.array([[2.0, 3.0]], dtype=np.float32),
            meta=_metadata(),
        )
    with pytest.raises(ValueError, match="ordered history before future"):
        _sample(future_start_frame=11)


def test_adapter_contract_has_required_day2_methods() -> None:
    class DummyAdapter:
        dataset_name = "highd"

        def load_raw(self, source: Path) -> object:
            return source

        def preprocess(self, raw: object, config: dict[str, object]) -> object:
            return raw

        def build_samples(
            self, cleaned: object, config: dict[str, object]
        ) -> list[TrajectorySample]:
            return []

    assert isinstance(DummyAdapter(), DatasetAdapter)


def test_split_assignments_reject_group_leakage_before_windowing() -> None:
    assignments = validate_split_assignments(
        [(101, "train"), (101, "train"), (202, "validation")], group_key="vehicle_id"
    )
    assert assignments == {101: "train", 202: "validation"}

    with pytest.raises(ValueError, match="split groups before building windows"):
        validate_split_assignments(
            [(101, "train"), (101, "test")], group_key="vehicle_id"
        )


def test_time_order_is_validated_before_window_construction() -> None:
    validate_window_order([10, 11], [12, 13, 14])
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_strictly_increasing_frames([10, 10])
    with pytest.raises(ValueError, match="must end before"):
        validate_window_order([10, 12], [12, 13])


def test_scaler_fits_only_train_coordinates_and_is_reversible() -> None:
    coordinates = np.array([[[0.0, 2.0], [2.0, 6.0]]], dtype=np.float32)
    scaler = TrainingCoordinateScaler().fit(coordinates, split="train")
    normalized = scaler.transform(coordinates)
    assert normalized.shape == coordinates.shape
    assert np.allclose(scaler.inverse_transform(normalized), coordinates)

    with pytest.raises(ValueError, match="only be fitted on the train split"):
        TrainingCoordinateScaler().fit(coordinates, split="validation")


def test_dataset_requires_one_split_id_and_fixed_window_shape() -> None:
    dataset = TrajectoryDataset(
        [_sample()],
        split="train",
        split_id="highd-split-42",
        window_spec=WindowSpec(history_steps=2, future_steps=3, stride=1),
    )
    assert len(dataset) == 1

    with pytest.raises(ValueError, match="dataset split_id"):
        TrajectoryDataset(
            [_sample(split_id="different-split")],
            split="train",
            split_id="highd-split-42",
            window_spec=WindowSpec(history_steps=2, future_steps=3, stride=1),
        )
