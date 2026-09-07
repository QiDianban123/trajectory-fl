"""Dataset container that enforces the Day 2 standard-sample contract."""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from src.data.adapters import SplitName, TrajectorySample
from src.data.preprocess import TrainingCoordinateScaler, WindowSpec


class TrajectoryDataset(Sequence[TrajectorySample]):
    """An immutable sequence of one split's fixed-length trajectory samples.

    D4 may add a PyTorch-specific wrapper; this dependency-free container keeps
    split, window, and metadata validation available to every data consumer.
    """

    def __init__(
        self,
        samples: Sequence[TrajectorySample],
        *,
        split: SplitName,
        split_id: str,
        window_spec: WindowSpec,
    ) -> None:
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test")
        if not split_id.strip():
            raise ValueError("split_id must be a non-empty string")
        self._samples = tuple(samples)
        self.split = split
        self.split_id = split_id
        self.window_spec = window_spec
        self._validate_samples()

    def __getitem__(self, index: int) -> TrajectorySample:
        return self._samples[index]

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[TrajectorySample]:
        return iter(self._samples)

    def _validate_samples(self) -> None:
        for sample in self._samples:
            if sample.meta["split"] != self.split:
                raise ValueError("all samples must belong to the dataset split")
            if sample.meta["split_id"] != self.split_id:
                raise ValueError("all samples must use the dataset split_id")
            if sample.history.shape != (self.window_spec.history_steps, 2):
                raise ValueError("sample history shape does not match window_spec")
            if sample.future.shape != (self.window_spec.future_steps, 2):
                raise ValueError("sample future shape does not match window_spec")


def save_dataset(
    dataset: TrajectoryDataset,
    directory: str | Path,
    *,
    scaler: TrainingCoordinateScaler,
    stats: dict[str, object],
) -> Path:
    """Persist one split without pickle and return its artifact directory."""

    _validate_fitted_scaler(scaler)
    _ensure_json_serializable(stats, "stats")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    histories = (
        np.stack([sample.history for sample in dataset], axis=0)
        if dataset
        else np.empty((0, dataset.window_spec.history_steps, 2), dtype=np.float32)
    )
    futures = (
        np.stack([sample.future for sample in dataset], axis=0)
        if dataset
        else np.empty((0, dataset.window_spec.future_steps, 2), dtype=np.float32)
    )
    _atomic_savez(destination / "samples.npz", history=histories, future=futures)
    _atomic_savez(destination / "scaler.npz", mean=scaler.mean_, scale=scaler.scale_)
    manifest = {
        "schema_version": 1,
        "split": dataset.split,
        "split_id": dataset.split_id,
        "window_spec": {
            "history_steps": dataset.window_spec.history_steps,
            "future_steps": dataset.window_spec.future_steps,
            "stride": dataset.window_spec.stride,
            "coordinate_dimension": dataset.window_spec.coordinate_dimension,
        },
        "sample_count": len(dataset),
        "metadata": [dict(sample.meta) for sample in dataset],
        "stats": stats,
        "scaler": {"fitted_split": scaler.fitted_split, "artifact": "scaler.npz"},
    }
    _atomic_write_json(destination / "manifest.json", manifest)
    return destination


def load_dataset(
    directory: str | Path,
) -> tuple[TrajectoryDataset, TrainingCoordinateScaler, dict[str, object]]:
    """Rebuild a persisted dataset, scaler, and processing statistics."""

    source = Path(directory)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported dataset manifest schema")
    window_spec = WindowSpec(**manifest["window_spec"])
    with np.load(source / "samples.npz", allow_pickle=False) as arrays:
        histories = arrays["history"]
        futures = arrays["future"]
    metadata = manifest["metadata"]
    if len(metadata) != len(histories) or len(histories) != len(futures):
        raise ValueError("dataset manifest and sample arrays have different lengths")
    samples = [
        TrajectorySample(
            history=history.astype(np.float32), future=future.astype(np.float32), meta=meta
        )
        for history, future, meta in zip(histories, futures, metadata, strict=True)
    ]
    dataset = TrajectoryDataset(
        samples,
        split=manifest["split"],
        split_id=manifest["split_id"],
        window_spec=window_spec,
    )
    with np.load(source / "scaler.npz", allow_pickle=False) as scaler_values:
        mean = scaler_values["mean"]
        scale = scaler_values["scale"]
    scaler = TrainingCoordinateScaler()
    scaler.mean_ = np.asarray(mean, dtype=np.float32)
    scaler.scale_ = np.asarray(scale, dtype=np.float32)
    scaler.fitted_split = "train"
    _validate_fitted_scaler(scaler)
    return dataset, scaler, manifest.get("stats", {})


def save_split_datasets(
    datasets: dict[SplitName, TrajectoryDataset],
    directory: str | Path,
    *,
    scaler: TrainingCoordinateScaler,
    stats: dict[str, object],
    data_version: str,
) -> Path:
    """Persist all splits and write one manifest describing the complete split."""

    _validate_split_datasets(datasets)
    _validate_fitted_scaler(scaler)
    _ensure_json_serializable(stats, "stats")
    if not isinstance(data_version, str) or not data_version.strip():
        raise ValueError("data_version must be a non-empty string")
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        save_dataset(datasets[split], destination / split, scaler=scaler, stats=stats)
    manifest = {
        "schema_version": 1,
        "dataset": "highd",
        "data_version": data_version,
        "split_id": datasets["train"].split_id,
        "splits": {
            split: {"path": split, "sample_count": len(datasets[split])}
            for split in ("train", "validation", "test")
        },
        "scaler": "train/scaler.npz",
        "stats": stats,
    }
    _atomic_write_json(destination / "split_manifest.json", manifest)
    return destination


def _validate_fitted_scaler(scaler: TrainingCoordinateScaler) -> None:
    if scaler.mean_ is None or scaler.scale_ is None or scaler.fitted_split != "train":
        raise ValueError("scaler must be fitted on train before persistence")
    for name, values in (("mean", scaler.mean_), ("scale", scaler.scale_)):
        array = np.asarray(values)
        if array.shape != (2,) or not np.isfinite(array).all():
            raise ValueError(f"scaler {name} must contain two finite coordinates")
    if np.any(np.asarray(scaler.scale_) <= 0):
        raise ValueError("scaler scale must be positive")


def _validate_split_datasets(datasets: dict[SplitName, TrajectoryDataset]) -> None:
    expected = {"train", "validation", "test"}
    if set(datasets) != expected:
        raise ValueError("datasets must contain exactly train, validation, and test")
    train = datasets["train"]
    for split in expected:
        dataset = datasets[split]
        if not isinstance(dataset, TrajectoryDataset):
            raise TypeError(f"datasets[{split!r}] must be a TrajectoryDataset")
        if dataset.split != split:
            raise ValueError(f"datasets[{split!r}] has mismatched split {dataset.split!r}")
        if dataset.split_id != train.split_id:
            raise ValueError("all persisted datasets must share one split_id")
        if dataset.window_spec != train.window_spec:
            raise ValueError("all persisted datasets must share one window_spec")


def _ensure_json_serializable(value: object, name: str) -> None:
    try:
        json.dumps(value, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be JSON serializable: {exc}") from exc


def _atomic_write_json(path: Path, payload: object) -> None:
    _ensure_json_serializable(payload, path.name)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
