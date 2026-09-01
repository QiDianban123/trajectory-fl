"""Dataset-neutral trajectory sample and adapter contracts frozen on Day 2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

import numpy as np


SplitName = Literal["train", "validation", "test"]
CoordinateArray = np.ndarray


class TrajectoryMetadata(TypedDict, total=False):
    """Metadata carried by every standard trajectory sample.

    The required keys are validated by :class:`TrajectorySample`. ``client_id``
    is optional until the D4 client partitioning stage.
    """

    dataset_name: str
    data_version: str
    recording_id: str | int
    vehicle_id: str | int
    history_start_frame: int
    history_end_frame: int
    future_start_frame: int
    future_end_frame: int
    split_id: str
    split: SplitName
    client_id: str


_REQUIRED_META_KEYS = (
    "dataset_name",
    "data_version",
    "recording_id",
    "vehicle_id",
    "history_start_frame",
    "history_end_frame",
    "future_start_frame",
    "future_end_frame",
    "split_id",
    "split",
)
_VALID_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class TrajectorySample:
    """One fixed-length, normalized trajectory sample.

    ``history`` and ``future`` must be finite ``float32`` arrays shaped
    ``[T, 2]``. They remain in normalized coordinates during training; physical
    coordinates are recovered by the evaluation layer using the saved scaler.
    """

    history: CoordinateArray
    future: CoordinateArray
    meta: Mapping[str, object]

    def __post_init__(self) -> None:
        _validate_coordinate_array(self.history, "history")
        _validate_coordinate_array(self.future, "future")
        _validate_metadata(self.meta)


@runtime_checkable
class DatasetAdapter(Protocol):
    """Adapter boundary that hides raw public-dataset table formats."""

    dataset_name: str

    def load_raw(self, source: Path) -> Any:
        """Read raw source files without applying project transformations."""
        ...

    def preprocess(self, raw: Any, config: Mapping[str, object]) -> Any:
        """Clean and normalize raw records according to the frozen data config."""
        ...

    def build_samples(
        self, cleaned: Any, config: Mapping[str, object]
    ) -> Sequence[TrajectorySample]:
        """Create split-safe fixed-length standard samples from cleaned records."""
        ...


def _validate_coordinate_array(array: object, name: str) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape [T, 2] with T > 0")
    if array.dtype != np.float32:
        raise TypeError(f"{name} must use float32 coordinates")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite coordinates")


def _validate_metadata(meta: Mapping[str, object]) -> None:
    missing = [key for key in _REQUIRED_META_KEYS if key not in meta]
    if missing:
        raise ValueError(f"meta is missing required keys: {', '.join(missing)}")

    for key in ("dataset_name", "data_version", "split_id"):
        if not isinstance(meta[key], str) or not meta[key].strip():
            raise ValueError(f"meta.{key} must be a non-empty string")
    for key in ("recording_id", "vehicle_id"):
        if isinstance(meta[key], bool) or not isinstance(meta[key], (str, int)):
            raise ValueError(f"meta.{key} must be a string or integer")

    split = meta["split"]
    if split not in _VALID_SPLITS:
        raise ValueError("meta.split must be train, validation, or test")

    frame_keys = (
        "history_start_frame",
        "history_end_frame",
        "future_start_frame",
        "future_end_frame",
    )
    frames = []
    for key in frame_keys:
        value = meta[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"meta.{key} must be an integer")
        frames.append(value)
    if not frames[0] <= frames[1] < frames[2] <= frames[3]:
        raise ValueError("meta frame ranges must be ordered history before future")
