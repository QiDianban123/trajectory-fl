"""Split safety, temporal-order, and normalization contracts for trajectory data."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from src.data.adapters import SplitName

GroupKey = Literal["vehicle_id", "scenario_id"]
_VALID_SPLITS = {"train", "validation", "test"}


@dataclass(frozen=True)
class WindowSpec:
    """Fixed sliding-window dimensions, validated before sample construction."""

    history_steps: int
    future_steps: int
    stride: int
    coordinate_dimension: int = 2

    def __post_init__(self) -> None:
        for name in ("history_steps", "future_steps", "stride", "coordinate_dimension"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.coordinate_dimension != 2:
            raise ValueError("coordinate_dimension must be 2")


def validate_split_assignments(
    assignments: Iterable[tuple[str | int, SplitName]], *, group_key: GroupKey
) -> dict[str | int, SplitName]:
    """Require every split assignment before windowing and reject group leakage.

    Input is deliberately record-like instead of tied to a pandas schema so D3
    can call it after highD field mapping. A repeated vehicle/scenario is valid
    only when every record has the same split assignment.
    """

    if group_key not in ("vehicle_id", "scenario_id"):
        raise ValueError("group_key must be vehicle_id or scenario_id")

    resolved: dict[str | int, SplitName] = {}
    for group_id, split in assignments:
        if isinstance(group_id, bool) or not isinstance(group_id, (str, int)):
            raise ValueError("group identifiers must be strings or integers")
        if split not in _VALID_SPLITS:
            raise ValueError("split must be train, validation, or test")
        previous = resolved.setdefault(group_id, split)
        if previous != split:
            raise ValueError(
                f"{group_key} {group_id!r} occurs in both {previous!r} and {split!r}; "
                "split groups before building windows"
            )
    if not resolved:
        raise ValueError("at least one group split assignment is required")
    return resolved


def validate_strictly_increasing_frames(frames: Sequence[int]) -> None:
    """Reject empty, duplicate, or out-of-order frame sequences."""

    if not frames:
        raise ValueError("frame sequence cannot be empty")
    if any(isinstance(frame, bool) or not isinstance(frame, int) for frame in frames):
        raise ValueError("frames must be integers")
    if any(current <= previous for previous, current in zip(frames, frames[1:])):
        raise ValueError("frames must be strictly increasing")


def validate_window_order(history_frames: Sequence[int], future_frames: Sequence[int]) -> None:
    """Validate temporal order before pairing one history and future window."""

    validate_strictly_increasing_frames(history_frames)
    validate_strictly_increasing_frames(future_frames)
    if history_frames[-1] >= future_frames[0]:
        raise ValueError("history window must end before future window begins")


@dataclass
class TrainingCoordinateScaler:
    """Two-dimensional standard scaler that can only fit the training split."""

    mean_: np.ndarray | None = field(default=None, init=False)
    scale_: np.ndarray | None = field(default=None, init=False)
    fitted_split: SplitName | None = field(default=None, init=False)

    def fit(self, coordinates: np.ndarray, *, split: SplitName) -> "TrainingCoordinateScaler":
        """Fit x/y statistics from training coordinates only."""

        if split != "train":
            raise ValueError("scaler statistics may only be fitted on the train split")
        values = _validate_coordinate_values(coordinates)
        flattened = values.reshape(-1, 2)
        mean = flattened.mean(axis=0)
        scale = flattened.std(axis=0)
        self.mean_ = mean.astype(np.float32)
        self.scale_ = np.where(scale > 0, scale, 1.0).astype(np.float32)
        self.fitted_split = split
        return self

    def transform(self, coordinates: np.ndarray) -> np.ndarray:
        """Normalize valid x/y coordinates using training-only statistics."""

        mean, scale = self._fitted_statistics()
        values = _validate_coordinate_values(coordinates)
        return ((values - mean) / scale).astype(np.float32)

    def inverse_transform(self, normalized_coordinates: np.ndarray) -> np.ndarray:
        """Recover physical x/y coordinates for evaluation."""

        mean, scale = self._fitted_statistics()
        values = _validate_coordinate_values(normalized_coordinates)
        return (values * scale + mean).astype(np.float32)

    def _fitted_statistics(self) -> tuple[np.ndarray, np.ndarray]:
        if self.mean_ is None or self.scale_ is None or self.fitted_split != "train":
            raise RuntimeError("scaler must be fitted on the train split before use")
        return self.mean_, self.scale_


def _validate_coordinate_values(coordinates: object) -> np.ndarray:
    if not isinstance(coordinates, np.ndarray):
        raise TypeError("coordinates must be a numpy.ndarray")
    if coordinates.ndim < 2 or coordinates.shape[-1] != 2 or coordinates.size == 0:
        raise ValueError("coordinates must have non-empty shape [..., 2]")
    if not np.issubdtype(coordinates.dtype, np.floating):
        raise TypeError("coordinates must use a floating-point dtype")
    if not np.isfinite(coordinates).all():
        raise ValueError("coordinates must be finite")
    return coordinates
