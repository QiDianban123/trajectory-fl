"""Orchestrate the frozen data pipeline behind the ``prepare-data`` command."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.adapters import HighDAdapter, SplitName, TrajectorySample
from src.data.dataset import TrajectoryDataset, save_split_datasets
from src.data.partition import (
    GroupExtent,
    PartitionConfig,
    PartitionError,
    PartitionManifest,
    check_partition_invariants,
    partition_train_groups,
)
from src.experiments import RunContext
from src.models.base import ModelContract
from src.training import sample_to_tensor


class PrepareDataError(ValueError):
    """A user-correctable data-preparation input or output error."""


@dataclass(frozen=True)
class PreparedData:
    """Paths and counts emitted by one successful data-preparation run."""

    data_version: str
    split_id: str
    processed_dir: Path
    split_manifest_path: Path
    partition_manifest_path: Path
    processed_index_path: Path
    run_manifest_path: Path
    sample_counts: Mapping[str, int]


def prepare_data(
    bundle: Mapping[str, Mapping[str, Any]],
    *,
    project_root: str | Path,
    raw_dir: str | Path | None = None,
    processed_dir: str | Path | None = None,
    output_root: str | Path | None = None,
    run_id: str | None = None,
) -> PreparedData:
    """Run B/C/D/G public interfaces without reimplementing their algorithms."""

    root = Path(project_root).resolve()
    if not root.is_dir():
        raise PrepareDataError(f"project root does not exist: {root}")
    data_config = _mapping(bundle, "data")
    model_config = _mapping(bundle, "model")
    experiment_config = _mapping(bundle, "experiment")
    dataset_config = _mapping(data_config, "dataset")
    run_config = _mapping(experiment_config, "run")

    raw_source = _resolve_project_path(
        root, raw_dir if raw_dir is not None else dataset_config["raw_dir"], "raw directory"
    )
    if not raw_source.is_dir():
        raise PrepareDataError(f"raw directory does not exist: {raw_source}")
    processed_root = _resolve_project_path(
        root,
        processed_dir if processed_dir is not None else dataset_config["processed_dir"],
        "processed directory",
    )
    if processed_root.exists() and not processed_root.is_dir():
        raise PrepareDataError(f"processed directory is not a directory: {processed_root}")
    resolved_output_root = _resolve_project_path(
        root,
        output_root if output_root is not None else run_config["output_root"],
        "output root",
    )

    adapter = HighDAdapter()
    raw_files = _find_raw_files(raw_source)
    try:
        cleaned = adapter.preprocess(adapter.load_raw(raw_source), data_config)
        datasets = adapter.build_datasets(cleaned, data_config)
        groups = _train_group_extents(cleaned, datasets["train"])
        partition = partition_train_groups(
            groups, PartitionConfig.from_mapping(_mapping(data_config, "partition"))
        )
        check_partition_invariants(partition, groups)
        datasets = _with_train_client_ids(datasets, partition)
        _verify_batch_bridge(datasets["train"], _mapping(model_config, "model"))
    except (FileNotFoundError, OSError, PartitionError, TypeError, ValueError) as exc:
        raise PrepareDataError(str(exc)) from exc

    split_id = str(cleaned["split_id"])
    destination = processed_root / split_id
    if destination.exists():
        raise PrepareDataError(
            f"processed split already exists and will not be overwritten: {destination}"
        )
    context: RunContext | None = None
    try:
        context = RunContext(bundle, root, run_id=run_id, output_root=resolved_output_root)
        processed_root.mkdir(parents=True, exist_ok=True)
        save_split_datasets(
            datasets,
            destination,
            scaler=cleaned["scaler"],  # type: ignore[arg-type]
            stats=cleaned["stats"],  # type: ignore[arg-type]
            data_version=str(cleaned["data_version"]),
        )
        partition_manifest_path = destination / "partition_manifest.json"
        _write_json(partition_manifest_path, partition.to_mapping())
        context.set_data_identity(data_version=str(cleaned["data_version"]), split_id=split_id)
        for path in raw_files:
            context.record_data_file(path)
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                context.record_data_file(path)
        split_manifest_path = destination / "split_manifest.json"
        split_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
        context.add_split_manifest("dataset", split_manifest)
        context.add_partition_manifest(partition)
        sample_counts = {split: len(dataset) for split, dataset in datasets.items()}
        processed_index_path = context.output_dir / "processed_index.json"
        _write_json(
            processed_index_path,
            {
                "processed_dir": _relative_to_root(destination, root),
                "split_manifest": _relative_to_root(split_manifest_path, root),
                "partition_manifest": _relative_to_root(partition_manifest_path, root),
                "scaler": _relative_to_root(destination / "train" / "scaler.npz", root),
                "splits": {
                    split: _relative_to_root(destination / split / "samples.npz", root)
                    for split in ("train", "validation", "test")
                },
            },
        )
        context.add_artifact("processed_index", processed_index_path.name)
        context.export_data_profile(
            {
                "dataset": adapter.dataset_name,
                "processed_dir": _relative_to_root(destination, root),
                "stats": cleaned["stats"],
                "sample_counts": sample_counts,
                "partition_clients": partition.num_clients,
            }
        )
        run_manifest_path = context.export_manifest()
    except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as exc:
        _remove_created_directory(destination, processed_root)
        if context is not None:
            _remove_created_directory(context.output_dir, resolved_output_root)
        raise PrepareDataError(str(exc)) from exc

    return PreparedData(
        data_version=str(cleaned["data_version"]),
        split_id=split_id,
        processed_dir=destination,
        split_manifest_path=split_manifest_path,
        partition_manifest_path=partition_manifest_path,
        processed_index_path=processed_index_path,
        run_manifest_path=run_manifest_path,
        sample_counts=sample_counts,
    )


def _mapping(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = value.get(name)
    if not isinstance(section, Mapping):
        raise PrepareDataError(f"{name} must be a mapping")
    return section


def _resolve_project_path(root: Path, value: object, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise PrepareDataError(f"{name} must be a path")
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise PrepareDataError(f"{name} escapes project root: {value}")
    return resolved


def _relative_to_root(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _find_raw_files(source: Path) -> list[Path]:
    files = sorted(source.glob("*_tracks.csv"))
    if not files:
        files = sorted(source.glob("*.csv"))
    if not files:
        raise PrepareDataError(f"no highD CSV files found in raw directory: {source}")
    return files


def _group_id(recording_id: object, vehicle_id: object) -> str:
    return json.dumps([str(recording_id), int(vehicle_id)], separators=(",", ":"))


def _train_group_extents(
    cleaned: Mapping[str, object], train: TrajectoryDataset
) -> list[GroupExtent]:
    records = cleaned.get("records")
    if not isinstance(records, pd.DataFrame):
        raise PrepareDataError("adapter cleaned records must be a pandas DataFrame")
    train_records = records.loc[records["split"] == "train"]
    sample_counts = Counter(
        _group_id(sample.meta["recording_id"], sample.meta["vehicle_id"]) for sample in train
    )
    return [
        GroupExtent(
            group_id=_group_id(recording_id, vehicle_id),
            x_min=float(track["x"].min()),
            x_max=float(track["x"].max()),
            sample_count=sample_counts[_group_id(recording_id, vehicle_id)],
        )
        for (recording_id, vehicle_id), track in train_records.groupby(
            ["recording_id", "id"], sort=False
        )
    ]


def _with_train_client_ids(
    datasets: Mapping[SplitName, TrajectoryDataset], partition: PartitionManifest
) -> dict[SplitName, TrajectoryDataset]:
    train = datasets["train"]
    assignments = partition.assignment
    samples: list[TrajectorySample] = []
    for sample in train:
        group_id = _group_id(sample.meta["recording_id"], sample.meta["vehicle_id"])
        try:
            client_id = assignments[group_id]
        except KeyError as exc:
            raise PrepareDataError(
                f"train sample group has no partition client: {group_id}"
            ) from exc
        samples.append(
            TrajectorySample(
                sample.history, sample.future, {**dict(sample.meta), "client_id": client_id}
            )
        )
    result = dict(datasets)
    result["train"] = TrajectoryDataset(
        samples,
        split="train",
        split_id=train.split_id,
        window_spec=train.window_spec,
    )
    return result


def _verify_batch_bridge(
    train: Sequence[TrajectorySample], model_config: Mapping[str, Any]
) -> None:
    if not train:
        raise PrepareDataError("train split contains no samples for the batch bridge")
    contract = ModelContract.from_model_config(model_config)
    sample_to_tensor(train[0], contract=contract)


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_created_directory(path: Path, expected_parent: Path) -> None:
    resolved = path.resolve()
    parent = expected_parent.resolve()
    if resolved.parent != parent or not resolved.is_dir():
        return
    shutil.rmtree(resolved)
