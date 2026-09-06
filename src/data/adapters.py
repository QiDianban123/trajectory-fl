"""Dataset-neutral trajectory sample and adapter contracts frozen on Day 2."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

import numpy as np
import pandas as pd

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


class HighDAdapter:
    """Read, clean, split, normalize, and window highD track CSV files."""

    dataset_name = "highd"

    def load_raw(self, source: Path) -> pd.DataFrame:
        """Load one highD CSV or all track CSVs in a directory."""

        source = Path(source)
        files = [source] if source.is_file() else sorted(source.glob("*_tracks.csv"))
        if not files and source.is_dir():
            files = sorted(source.glob("*.csv"))
        if not files:
            raise FileNotFoundError(f"No highD CSV files found at {source}")

        frames = []
        for file in files:
            frame = pd.read_csv(file)
            frame = self._map_columns(frame)
            if "recording_id" not in frame:
                frame["recording_id"] = file.stem.removesuffix("_tracks")
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def preprocess(self, raw: pd.DataFrame, config: Mapping[str, object]) -> dict[str, object]:
        """Clean tracks, assign disjoint splits, and fit a train-only scaler."""

        from src.data.preprocess import TrainingCoordinateScaler, validate_split_assignments

        frame = self._map_columns(raw.copy())
        if "recording_id" not in frame:
            frame["recording_id"] = "inline"
        required = {"id", "frame", "x", "y", "recording_id"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"highD input is missing required columns: {', '.join(missing)}")

        frame["id"] = pd.to_numeric(frame["id"], errors="coerce")
        frame["frame"] = pd.to_numeric(frame["frame"], errors="coerce")
        frame["x"] = pd.to_numeric(frame["x"], errors="coerce")
        frame["y"] = pd.to_numeric(frame["y"], errors="coerce")
        stats = {"input_rows": len(frame), "rejected_tracks": 0, "rejected_rows": 0}
        minimum = int(config["preprocessing"]["minimum_track_frames"])
        valid_tracks: list[pd.DataFrame] = []
        for _, track in frame.groupby(["recording_id", "id"], sort=False, dropna=False):
            track = track.copy()
            if track[["id", "frame", "x", "y"]].isna().any().any():
                stats["rejected_tracks"] += 1
                stats["rejected_rows"] += len(track)
                continue
            if not np.isfinite(track[["frame", "x", "y"]].to_numpy(dtype=float)).all():
                stats["rejected_tracks"] += 1
                stats["rejected_rows"] += len(track)
                continue
            track["id"] = track["id"].astype(np.int64)
            track["frame"] = track["frame"].astype(np.int64)
            if track["frame"].duplicated().any():
                stats["rejected_tracks"] += 1
                stats["rejected_rows"] += len(track)
                continue
            track = track.sort_values("frame")
            if len(track) < minimum:
                stats["rejected_tracks"] += 1
                stats["rejected_rows"] += len(track)
                continue
            valid_tracks.append(track)
        if not valid_tracks:
            raise ValueError("No valid highD tracks remain after preprocessing")

        cleaned = pd.concat(valid_tracks, ignore_index=True)
        split_config = config["split"]
        groups = sorted(
            cleaned[["recording_id", "id"]].drop_duplicates().itertuples(index=False, name=None),
            key=str,
        )
        rng = np.random.default_rng(int(split_config["seed"]))
        order = rng.permutation(len(groups))
        shuffled = [groups[index] for index in order]
        assignments: dict[tuple[object, object], SplitName] = {}
        ratios = (float(split_config["train"]), float(split_config["validation"]))
        train_end = round(len(shuffled) * ratios[0])
        validation_end = train_end + round(len(shuffled) * ratios[1])
        for index, group in enumerate(shuffled):
            split: SplitName = (
                "train" if index < train_end else "validation" if index < validation_end else "test"
            )
            assignments[group] = split
        validate_split_assignments(
            [
                (f"{recording}:{vehicle}", split)
                for (recording, vehicle), split in assignments.items()
            ],
            group_key="vehicle_id",
        )
        keys = list(zip(cleaned["recording_id"], cleaned["id"], strict=True))
        cleaned["split"] = [assignments[key] for key in keys]

        train_coordinates = cleaned.loc[cleaned["split"] == "train", ["x", "y"]].to_numpy(
            dtype=np.float32
        )
        if train_coordinates.size == 0:
            raise ValueError("vehicle split produced no training tracks for scaler fitting")
        scaler = TrainingCoordinateScaler().fit(train_coordinates, split="train")
        data_version = self._data_version(cleaned)
        split_id = self._split_id(assignments, split_config)
        stats.update(
            {
                "valid_tracks": len(valid_tracks),
                "valid_rows": len(cleaned),
                "split_counts": {
                    name: sum(value == name for value in assignments.values())
                    for name in ("train", "validation", "test")
                },
            }
        )
        return {
            "records": cleaned,
            "assignments": assignments,
            "scaler": scaler,
            "stats": stats,
            "data_version": data_version,
            "split_id": split_id,
        }

    def build_samples(
        self, cleaned: Mapping[str, object], config: Mapping[str, object]
    ) -> Sequence[TrajectorySample]:
        """Build normalized fixed-length samples after group-level splitting."""

        from src.data.preprocess import TrainingCoordinateScaler, WindowSpec

        records = cleaned["records"]
        if not isinstance(records, pd.DataFrame):
            raise TypeError("cleaned records must be a pandas DataFrame")
        scaler = cleaned["scaler"]
        if not isinstance(scaler, TrainingCoordinateScaler):
            raise TypeError("cleaned scaler must be a TrainingCoordinateScaler")
        window = WindowSpec(**config["sequence"])
        split_id = str(cleaned["split_id"])
        data_version = str(cleaned["data_version"])
        samples: list[TrajectorySample] = []
        for (recording_id, vehicle_id), track in records.groupby(
            ["recording_id", "id"], sort=False
        ):
            track = track.sort_values("frame")
            split = track["split"].iloc[0]
            coordinates = scaler.transform(track[["x", "y"]].to_numpy(dtype=np.float32))
            frames = track["frame"].to_numpy(dtype=np.int64)
            total = window.history_steps + window.future_steps
            for start in range(0, len(track) - total + 1, window.stride):
                history_frames = frames[start : start + window.history_steps]
                future_frames = frames[start + window.history_steps : start + total]
                meta: TrajectoryMetadata = {
                    "dataset_name": self.dataset_name,
                    "data_version": data_version,
                    "recording_id": self._scalar(recording_id),
                    "vehicle_id": int(vehicle_id),
                    "history_start_frame": int(history_frames[0]),
                    "history_end_frame": int(history_frames[-1]),
                    "future_start_frame": int(future_frames[0]),
                    "future_end_frame": int(future_frames[-1]),
                    "split_id": split_id,
                    "split": split,
                }
                samples.append(
                    TrajectorySample(
                        history=coordinates[start : start + window.history_steps],
                        future=coordinates[start + window.history_steps : start + total],
                        meta=meta,
                    )
                )
        return samples

    def build_datasets(
        self, cleaned: Mapping[str, object], config: Mapping[str, object]
    ) -> dict[SplitName, object]:
        """Group built samples into validated train/validation/test datasets."""

        from src.data.dataset import TrajectoryDataset
        from src.data.preprocess import WindowSpec

        samples = self.build_samples(cleaned, config)
        window_spec = WindowSpec(**config["sequence"])
        split_id = str(cleaned["split_id"])
        return {
            split: TrajectoryDataset(
                [sample for sample in samples if sample.meta["split"] == split],
                split=split,
                split_id=split_id,
                window_spec=window_spec,
            )
            for split in ("train", "validation", "test")
        }

    @staticmethod
    def _map_columns(frame: pd.DataFrame) -> pd.DataFrame:
        aliases = {
            "id": ("id", "track id", "track_id", "vehicle_id"),
            "frame": ("frame", "frame id", "frame_id"),
            "x": ("x", "x position", "x_position"),
            "y": ("y", "y position", "y_position"),
        }
        normalized = {
            str(column).strip().lower().replace("_", " "): column for column in frame.columns
        }
        rename = {}
        for target, choices in aliases.items():
            source = next((normalized[choice] for choice in choices if choice in normalized), None)
            if source is not None:
                rename[source] = target
        return frame.rename(columns=rename)

    @staticmethod
    def _data_version(records: pd.DataFrame) -> str:
        digest = hashlib.sha256(
            pd.util.hash_pandas_object(records, index=False).values.tobytes()
        ).hexdigest()
        return f"highd-{digest[:16]}"

    @staticmethod
    def _split_id(
        assignments: Mapping[tuple[object, object], SplitName], split_config: Mapping[str, object]
    ) -> str:
        payload = repr(sorted((str(key), value) for key, value in assignments.items())) + repr(
            dict(split_config)
        )
        return f"highd-split-{hashlib.sha256(payload.encode()).hexdigest()[:16]}"

    @staticmethod
    def _scalar(value: object) -> str | int:
        return int(value) if isinstance(value, (int, np.integer)) else str(value)


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
