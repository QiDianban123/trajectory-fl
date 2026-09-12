"""S3 client-local DataLoaders built from an immutable processed split."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from src.data.adapters import SplitName, TrajectorySample
from src.data.dataset import TrajectoryDataset
from src.data.loading import DataLoaderConfig, ProcessedDataBundle
from src.data.partition import PartitionManifest
from src.models.base import ModelContract, require_torch
from src.training.batching import collate_trajectory_samples


class ClientDataError(ValueError):
    """A frozen client partition cannot safely be applied to processed data."""


@dataclass(frozen=True)
class ClientDataProfile:
    """Serializable, non-sensitive per-client data identity and counts."""

    client_id: str
    x_min: float
    x_max: float
    vehicle_count: int
    train_sample_count: int
    validation_sample_count: int
    test_sample_count: int
    train_x_min: float | None
    train_x_max: float | None

    def sample_visits(self, local_epochs: int) -> int:
        if isinstance(local_epochs, bool) or not isinstance(local_epochs, int) or local_epochs <= 0:
            raise ClientDataError("local_epochs must be a positive integer")
        return self.train_sample_count * local_epochs

    def to_dict(self) -> dict[str, object]:
        return {
            "client_id": self.client_id,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "vehicle_count": self.vehicle_count,
            "train_sample_count": self.train_sample_count,
            "validation_sample_count": self.validation_sample_count,
            "test_sample_count": self.test_sample_count,
            "train_x_min": self.train_x_min,
            "train_x_max": self.train_x_max,
        }


@dataclass(frozen=True)
class ClientDataLoaders:
    """One client's isolated datasets/loaders plus stable accounting fields."""

    client_id: str
    datasets: Mapping[SplitName, TrajectoryDataset]
    loaders: Mapping[SplitName, Any]
    profile: ClientDataProfile

    @property
    def is_empty_train(self) -> bool:
        return self.profile.train_sample_count == 0


@dataclass(frozen=True)
class ClientDataBundle:
    """Client-local view of one processed/scaler/partition identity."""

    data_version: str
    split_id: str
    partition_id: str
    clients: Mapping[str, ClientDataLoaders]

    def profiles(self) -> tuple[ClientDataProfile, ...]:
        return tuple(self.clients[client_id].profile for client_id in sorted(self.clients))

    def write_profiles(self, path: str | Path) -> Path:
        """Export stable RSU/sample/coordinate diagnostics without raw trajectories."""

        destination = Path(path)
        payload = {
            "schema_version": 1,
            "data_version": self.data_version,
            "split_id": self.split_id,
            "partition_id": self.partition_id,
            "clients": [profile.to_dict() for profile in self.profiles()],
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return destination

def create_client_dataloaders(
    data: ProcessedDataBundle,
    partition: PartitionManifest,
    *,
    contract: ModelContract,
    config: DataLoaderConfig,
) -> ClientDataBundle:
    """Build deterministic, disjoint train and frozen-boundary holdout loaders.

    Train samples use the persisted ``meta.client_id`` created from the train
    partition.  Validation/test samples deliberately do not reuse vehicle IDs:
    they are projected with the train-fitted scaler onto the already frozen
    post-merge client intervals.
    """

    if config.drop_last:
        raise ClientDataError("client loaders reject drop_last because it changes effective counts")
    intervals = _validate_intervals(partition)
    client_ids = tuple(client_id for client_id, _, _ in intervals)
    train_samples: dict[str, list[TrajectorySample]] = {client_id: [] for client_id in client_ids}
    holdout_samples: dict[str, dict[SplitName, list[TrajectorySample]]] = {
        client_id: {"validation": [], "test": []} for client_id in client_ids
    }
    for sample in data.datasets["train"]:
        client_id = sample.meta.get("client_id")
        if not isinstance(client_id, str) or client_id not in train_samples:
            raise ClientDataError("train sample client_id is missing or not in frozen partition")
        train_samples[client_id].append(sample)
    for split in ("validation", "test"):
        for sample in data.datasets[split]:
            client_id = _holdout_client_id(sample, data, intervals)
            holdout_samples[client_id][split].append(sample)

    _validate_train_counts(train_samples, partition)
    clients: dict[str, ClientDataLoaders] = {}
    for offset, (client_id, low, high) in enumerate(intervals):
        datasets = {
            "train": _dataset_like(data.datasets["train"], train_samples[client_id]),
            "validation": _dataset_like(
                data.datasets["validation"], holdout_samples[client_id]["validation"]
            ),
            "test": _dataset_like(data.datasets["test"], holdout_samples[client_id]["test"]),
        }
        loaders = {
            split: _loader_for_dataset(
                dataset, split=split, contract=contract, config=config, seed_offset=offset * 3
            )
            for split, dataset in datasets.items()
        }
        clients[client_id] = ClientDataLoaders(
            client_id=client_id,
            datasets=MappingProxyType(datasets),
            loaders=MappingProxyType(loaders),
            profile=_profile(client_id, low, high, datasets, data),
        )
    return ClientDataBundle(
        data_version=data.data_version,
        split_id=data.split_id,
        partition_id=_partition_id(partition),
        clients=MappingProxyType(clients),
    )


def _validate_intervals(partition: PartitionManifest) -> tuple[tuple[str, float, float], ...]:
    intervals = tuple(
        (client.client_id, client.x_min, client.x_max) for client in partition.clients
    )
    if not intervals:
        raise ClientDataError("partition must contain at least one client")
    if (
        intervals[0][1] != partition.region_edges[0]
        or intervals[-1][2] != partition.region_edges[-1]
    ):
        raise ClientDataError("client intervals must cover the frozen region edges")
    for index, (_, low, high) in enumerate(intervals):
        if not isfinite(low) or not isfinite(high) or low > high:
            raise ClientDataError("client intervals must be finite and ordered")
        if index and low != intervals[index - 1][2]:
            raise ClientDataError("client intervals must be contiguous without overlap or gaps")
    return intervals


def _holdout_client_id(
    sample: TrajectorySample,
    data: ProcessedDataBundle,
    intervals: tuple[tuple[str, float, float], ...],
) -> str:
    history = sample.history[-1:]
    physical = data.inverse_transform(history)
    if physical.shape != (1, 2) or not np.isfinite(physical).all():
        raise ClientDataError("holdout assignment anchor is invalid")
    anchor = float(physical[0, 0])
    for index, (client_id, low, high) in enumerate(intervals):
        if low <= anchor < high or (index == len(intervals) - 1 and anchor == high):
            return client_id
    raise ClientDataError("holdout assignment anchor is outside frozen client intervals")


def _validate_train_counts(
    samples: Mapping[str, list[TrajectorySample]], partition: PartitionManifest
) -> None:
    expected = {client.client_id: client.sample_count for client in partition.clients}
    actual = {client_id: len(values) for client_id, values in samples.items()}
    if actual != expected:
        raise ClientDataError(
            "train client sample counts do not match frozen partition: "
            f"expected={expected}, actual={actual}"
        )


def _dataset_like(
    reference: TrajectoryDataset, samples: list[TrajectorySample]
) -> TrajectoryDataset:
    return TrajectoryDataset(
        samples,
        split=reference.split,
        split_id=reference.split_id,
        window_spec=reference.window_spec,
    )


def _loader_for_dataset(
    dataset: TrajectoryDataset,
    *,
    split: SplitName,
    contract: ModelContract,
    config: DataLoaderConfig,
    seed_offset: int,
) -> Any:
    torch = require_torch()
    generator = torch.Generator()
    generator.manual_seed(config.seed + seed_offset)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle_train if split == "train" and dataset else False,
        drop_last=config.drop_last if split == "train" and dataset else False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        persistent_workers=config.persistent_workers if config.num_workers else False,
        collate_fn=partial(collate_trajectory_samples, contract=contract),
        generator=generator,
    )


def _profile(
    client_id: str,
    low: float,
    high: float,
    datasets: Mapping[SplitName, TrajectoryDataset],
    data: ProcessedDataBundle,
) -> ClientDataProfile:
    train = datasets["train"]
    anchors = [float(data.inverse_transform(sample.history[-1:])[0, 0]) for sample in train]
    vehicles = {(sample.meta["recording_id"], sample.meta["vehicle_id"]) for sample in train}
    return ClientDataProfile(
        client_id=client_id,
        x_min=low,
        x_max=high,
        vehicle_count=len(vehicles),
        train_sample_count=len(train),
        validation_sample_count=len(datasets["validation"]),
        test_sample_count=len(datasets["test"]),
        train_x_min=min(anchors) if anchors else None,
        train_x_max=max(anchors) if anchors else None,
    )


def _partition_id(partition: PartitionManifest) -> str:
    payload = json.dumps(partition.to_mapping(), sort_keys=True, separators=(",", ":"))
    return "partition-v1-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
