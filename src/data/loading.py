"""Strict processed-data reader and deterministic Torch DataLoader factory."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from src.data.adapters import SplitName, TrajectorySample
from src.data.cache import processed_cache_key, semantic_config_digest
from src.data.dataset import TrajectoryDataset, load_dataset
from src.data.preprocess import TrainingCoordinateScaler
from src.models.base import ModelContract, require_torch

_SPLITS: tuple[SplitName, ...] = ("train", "validation", "test")
_ARTIFACT_NAMES = ("manifest.json", "samples.npz", "scaler.npz")


class ProcessedDataError(ValueError):
    """A processed dataset is stale, corrupt, incomplete, or inconsistent."""


@dataclass(frozen=True)
class DataLoaderConfig:
    """Runtime-only loader settings that do not alter dataset semantics."""

    batch_size: int
    num_workers: int
    seed: int
    shuffle_train: bool = True
    drop_last: bool = False
    pin_memory: bool = False
    persistent_workers: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int):
            raise ValueError("batch_size must be a positive integer")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        for name in ("num_workers", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("shuffle_train", "drop_last", "pin_memory", "persistent_workers"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be boolean")
        if self.persistent_workers and self.num_workers == 0:
            raise ValueError("persistent_workers requires num_workers greater than zero")

    @classmethod
    def from_config_bundle(cls, bundle: Mapping[str, Mapping[str, Any]]) -> DataLoaderConfig:
        """Read batch size, worker count, and seed from the frozen config bundle."""

        try:
            training = bundle["model"]["training"]
            execution = bundle["experiment"]["execution"]
            run = bundle["experiment"]["run"]
            return cls(
                batch_size=training["batch_size"],
                num_workers=execution["num_workers"],
                seed=run["seed"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"config bundle is missing DataLoader settings: {exc}") from exc


@dataclass(frozen=True)
class ProcessedDataBundle:
    """Validated splits plus the single train-fitted coordinate scaler."""

    datasets: Mapping[SplitName, TrajectoryDataset]
    scaler: TrainingCoordinateScaler
    stats: Mapping[str, object]
    data_version: str
    split_id: str
    cache_key: str
    source: Path

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Recover physical coordinates without exposing scaler mutation."""

        return self.scaler.inverse_transform(values)


class ProcessedDatasetReader:
    """Read and cache a complete processed split after validating its provenance."""

    def __init__(self) -> None:
        self._cache: dict[
            tuple[Path, str], tuple[tuple[tuple[str, int, int], ...], ProcessedDataBundle]
        ] = {}

    def clear(self) -> None:
        """Discard all in-process cached bundles."""

        self._cache.clear()

    def load(
        self,
        directory: str | Path,
        *,
        data_config: Mapping[str, object],
        expected_data_version: str | None = None,
        expected_split_id: str | None = None,
    ) -> ProcessedDataBundle:
        """Load all non-empty splits, rejecting stale identities and corrupt files."""

        source = Path(directory).resolve()
        if not source.is_dir():
            raise ProcessedDataError(f"processed dataset directory does not exist: {source}")
        manifest = _read_json(source / "split_manifest.json", "split manifest")
        data_version, split_id, cache_key, artifact_paths = _validate_parent_manifest(
            manifest,
            source=source,
            data_config=data_config,
            expected_data_version=expected_data_version,
            expected_split_id=expected_split_id,
        )
        signature = _artifact_signature(source, (source / "split_manifest.json", *artifact_paths))
        cache_slot = (source, cache_key)
        cached = self._cache.get(cache_slot)
        if cached is not None and cached[0] == signature:
            return _copy_bundle(cached[1])

        _validate_artifact_checksums(manifest, source, artifact_paths)
        datasets: dict[SplitName, TrajectoryDataset] = {}
        scalers: list[TrainingCoordinateScaler] = []
        expected_stats = manifest.get("stats")
        for split in _SPLITS:
            split_entry = manifest["splits"][split]
            split_source = _safe_child(source, split_entry["path"], f"{split} path")
            try:
                dataset, scaler, stats = load_dataset(split_source)
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ProcessedDataError(f"cannot load processed {split} split: {exc}") from exc
            _validate_loaded_split(
                dataset,
                scaler,
                stats,
                split=split,
                parent_manifest=manifest,
                data_config=data_config,
                expected_stats=expected_stats,
            )
            datasets[split] = _freeze_dataset(dataset)
            scalers.append(scaler)
        _validate_shared_scaler(scalers)
        bundle = ProcessedDataBundle(
            datasets=MappingProxyType(datasets),
            scaler=_copy_scaler(scalers[0]),
            stats=_freeze_mapping(expected_stats),
            data_version=data_version,
            split_id=split_id,
            cache_key=cache_key,
            source=source,
        )
        self._cache[cache_slot] = (signature, bundle)
        return _copy_bundle(bundle)


def create_dataloaders(
    data: ProcessedDataBundle,
    *,
    contract: ModelContract,
    config: DataLoaderConfig,
) -> Mapping[SplitName, Any]:
    """Create reproducible train and ordered holdout loaders for one data identity."""

    # Import lazily so ``src.training`` and ``src.data`` remain independently importable.
    from src.training.batching import collate_trajectory_samples

    torch = require_torch()
    loaders: dict[SplitName, Any] = {}
    for offset, split in enumerate(_SPLITS):
        generator = torch.Generator()
        generator.manual_seed(config.seed + offset)
        options: dict[str, object] = {
            "batch_size": config.batch_size,
            "shuffle": config.shuffle_train if split == "train" else False,
            "drop_last": config.drop_last if split == "train" else False,
            "num_workers": config.num_workers,
            "pin_memory": config.pin_memory,
            "collate_fn": partial(collate_trajectory_samples, contract=contract),
            "generator": generator,
            "worker_init_fn": _seed_worker,
        }
        if config.num_workers:
            options["persistent_workers"] = config.persistent_workers
        loaders[split] = torch.utils.data.DataLoader(data.datasets[split], **options)
    return MappingProxyType(loaders)


def _seed_worker(worker_id: int) -> None:
    del worker_id
    torch = require_torch()
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _validate_parent_manifest(
    manifest: Mapping[str, object],
    *,
    source: Path,
    data_config: Mapping[str, object],
    expected_data_version: str | None,
    expected_split_id: str | None,
) -> tuple[str, str, str, tuple[Path, ...]]:
    if manifest.get("schema_version") != 1:
        raise ProcessedDataError("unsupported split manifest schema")
    if manifest.get("dataset") != "highd":
        raise ProcessedDataError("split manifest dataset must be highd")
    data_version = _non_empty_string(manifest.get("data_version"), "data_version")
    split_id = _non_empty_string(manifest.get("split_id"), "split_id")
    if expected_data_version is not None and data_version != expected_data_version:
        raise ProcessedDataError(
            f"data_version mismatch: expected {expected_data_version!r}, found {data_version!r}"
        )
    if expected_split_id is not None and split_id != expected_split_id:
        raise ProcessedDataError(
            f"split_id mismatch: expected {expected_split_id!r}, found {split_id!r}"
        )
    identity = manifest.get("cache_identity")
    if not isinstance(identity, Mapping) or identity.get("schema_version") != 1:
        raise ProcessedDataError("processed cache is legacy or missing cache_identity schema 1")
    expected_digest = semantic_config_digest(data_config)
    if identity.get("semantic_config_digest") != expected_digest:
        raise ProcessedDataError(
            "processed cache semantic configuration does not match data config"
        )
    expected_key = processed_cache_key(
        data_version=data_version, split_id=split_id, data_config=data_config
    )
    if identity.get("key") != expected_key:
        raise ProcessedDataError("processed cache key does not match its data identity")

    splits = manifest.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(_SPLITS):
        raise ProcessedDataError("split manifest must contain exactly train, validation, and test")
    artifact_paths: list[Path] = []
    for split in _SPLITS:
        entry = splits[split]
        if not isinstance(entry, Mapping):
            raise ProcessedDataError(f"split manifest {split} entry must be a mapping")
        count = entry.get("sample_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ProcessedDataError(f"processed {split} split must contain at least one sample")
        split_source = _safe_child(source, entry.get("path"), f"{split} path")
        artifact_paths.extend(split_source / name for name in _ARTIFACT_NAMES)
    scaler_path = _safe_child(source, manifest.get("scaler"), "scaler path")
    expected_scaler = _safe_child(source, splits["train"]["path"], "train path") / "scaler.npz"
    if scaler_path != expected_scaler:
        raise ProcessedDataError("split manifest scaler must reference the train scaler")
    if not isinstance(manifest.get("stats"), Mapping):
        raise ProcessedDataError("split manifest stats must be a mapping")
    return data_version, split_id, expected_key, tuple(sorted(artifact_paths))


def _validate_loaded_split(
    dataset: TrajectoryDataset,
    scaler: TrainingCoordinateScaler,
    stats: Mapping[str, object],
    *,
    split: SplitName,
    parent_manifest: Mapping[str, object],
    data_config: Mapping[str, object],
    expected_stats: object,
) -> None:
    if dataset.split != split:
        raise ProcessedDataError(f"loaded {split} cache declares split {dataset.split!r}")
    if dataset.split_id != parent_manifest["split_id"]:
        raise ProcessedDataError(f"loaded {split} cache has a different split_id")
    expected_count = parent_manifest["splits"][split]["sample_count"]
    if len(dataset) != expected_count:
        raise ProcessedDataError(f"loaded {split} sample count does not match split manifest")
    sequence = data_config.get("sequence")
    if not isinstance(sequence, Mapping):
        raise ProcessedDataError("data.sequence must be a mapping")
    for key in ("history_steps", "future_steps", "stride", "coordinate_dimension"):
        if getattr(dataset.window_spec, key) != sequence.get(key):
            raise ProcessedDataError(f"loaded {split} window_spec.{key} does not match data config")
    if stats != expected_stats:
        raise ProcessedDataError(f"loaded {split} statistics do not match split manifest")
    for index, sample in enumerate(dataset):
        if sample.meta.get("data_version") != parent_manifest["data_version"]:
            raise ProcessedDataError(f"loaded {split} sample {index} has a different data_version")
        if sample.meta.get("dataset_name") != parent_manifest["dataset"]:
            raise ProcessedDataError(f"loaded {split} sample {index} has a different dataset_name")
    if scaler.fitted_split != "train":
        raise ProcessedDataError(f"loaded {split} scaler is not fitted on train")


def _validate_shared_scaler(scalers: list[TrainingCoordinateScaler]) -> None:
    first = scalers[0]
    for scaler in scalers[1:]:
        if not np.array_equal(scaler.mean_, first.mean_) or not np.array_equal(
            scaler.scale_, first.scale_
        ):
            raise ProcessedDataError("processed splits do not share the same train scaler")


def _freeze_dataset(dataset: TrajectoryDataset) -> TrajectoryDataset:
    samples: list[TrajectorySample] = []
    for sample in dataset:
        history = sample.history.copy()
        future = sample.future.copy()
        history.setflags(write=False)
        future.setflags(write=False)
        samples.append(
            TrajectorySample(
                history=history,
                future=future,
                meta=deepcopy(dict(sample.meta)),
            )
        )
    return TrajectoryDataset(
        samples,
        split=dataset.split,
        split_id=dataset.split_id,
        window_spec=dataset.window_spec,
    )


def _copy_bundle(bundle: ProcessedDataBundle) -> ProcessedDataBundle:
    datasets = {
        split: TrajectoryDataset(
            [
                TrajectorySample(
                    history=sample.history,
                    future=sample.future,
                    meta=deepcopy(dict(sample.meta)),
                )
                for sample in dataset
            ],
            split=dataset.split,
            split_id=dataset.split_id,
            window_spec=dataset.window_spec,
        )
        for split, dataset in bundle.datasets.items()
    }
    return ProcessedDataBundle(
        datasets=MappingProxyType(datasets),
        scaler=_copy_scaler(bundle.scaler),
        stats=_freeze_mapping(bundle.stats),
        data_version=bundle.data_version,
        split_id=bundle.split_id,
        cache_key=bundle.cache_key,
        source=bundle.source,
    )


def _copy_scaler(scaler: TrainingCoordinateScaler) -> TrainingCoordinateScaler:
    copied = TrainingCoordinateScaler()
    copied.mean_ = scaler.mean_.copy() if scaler.mean_ is not None else None
    copied.scale_ = scaler.scale_.copy() if scaler.scale_ is not None else None
    copied.fitted_split = scaler.fitted_split
    if copied.mean_ is not None:
        copied.mean_.setflags(write=False)
    if copied.scale_ is not None:
        copied.scale_.setflags(write=False)
    return copied


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType({str(key): _freeze_value(item) for key, item in value.items()})


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return deepcopy(value)


def _read_json(path: Path, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ProcessedDataError(f"cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ProcessedDataError(f"{name} must contain a JSON object")
    return value


def _safe_child(root: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ProcessedDataError(f"{name} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute():
        raise ProcessedDataError(f"{name} must be relative to the processed dataset")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ProcessedDataError(f"{name} escapes the processed dataset directory")
    return resolved


def _non_empty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProcessedDataError(f"split manifest {name} must be a non-empty string")
    return value


def _artifact_signature(root: Path, paths: tuple[Path, ...]) -> tuple[tuple[str, int, int], ...]:
    values: list[tuple[str, int, int]] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError as exc:
            raise ProcessedDataError(f"processed cache artifact is missing: {path}") from exc
        if not path.is_file():
            raise ProcessedDataError(f"processed cache artifact is not a file: {path}")
        values.append((path.relative_to(root).as_posix(), stat.st_size, stat.st_mtime_ns))
    return tuple(values)


def _validate_artifact_checksums(
    manifest: Mapping[str, object], root: Path, paths: tuple[Path, ...]
) -> None:
    artifacts = manifest.get("artifacts")
    expected_names = {path.relative_to(root).as_posix() for path in paths}
    if not isinstance(artifacts, Mapping) or set(artifacts) != expected_names:
        raise ProcessedDataError("processed cache artifact index is missing or inconsistent")
    for path in paths:
        name = path.relative_to(root).as_posix()
        entry = artifacts[name]
        if not isinstance(entry, Mapping):
            raise ProcessedDataError(f"processed cache artifact entry is invalid: {name}")
        expected = entry.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ProcessedDataError(f"processed cache artifact checksum is invalid: {name}")
        if _sha256(path) != expected:
            raise ProcessedDataError(f"processed cache artifact checksum mismatch: {name}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
