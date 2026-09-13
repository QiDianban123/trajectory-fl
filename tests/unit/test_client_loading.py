"""S3-B client-local loader, frozen-holdout, and accounting tests."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from src.data.adapters import TrajectorySample
from src.data.client_loading import ClientDataError, create_client_dataloaders
from src.data.dataset import TrajectoryDataset, save_split_datasets
from src.data.loading import DataLoaderConfig, ProcessedDataBundle, ProcessedDatasetReader
from src.data.partition import ClientPartition, PartitionManifest, client_group_id
from src.data.preprocess import TrainingCoordinateScaler, WindowSpec
from src.models.base import ModelContract


def _five_client_partition() -> PartitionManifest:
    edges = (10.0, 13.0, 15.0, 17.0, 19.0, 21.0)
    return PartitionManifest(
        region_edges=edges,
        clients=tuple(
            ClientPartition(
                f"rsu_{index:02d}", edges[index - 1], edges[index], 1, (_group_id(index - 1),)
            )
            for index in range(1, 6)
        ),
    )


def _group_id(vehicle_id: int) -> str:
    return client_group_id(1, vehicle_id)


def test_client_group_id_normalizes_numeric_and_string_recordings() -> None:
    assert client_group_id(1, 2) == client_group_id("1", "2") == '["1",2]'


@pytest.fixture
def processed_cache(tmp_path: Path, config_bundle) -> tuple[Path, dict[str, object]]:
    config = deepcopy(config_bundle["data"])
    window = WindowSpec(**config["sequence"])
    datasets = {}
    for split, count, vehicle_offset in (
        ("train", 5, 0),
        ("validation", 3, 10),
        ("test", 2, 20),
    ):
        samples = []
        for index in range(count):
            samples.append(
                TrajectorySample(
                    history=np.full((window.history_steps, 2), index + 0.25, dtype=np.float32),
                    future=np.full((window.future_steps, 2), index + 0.75, dtype=np.float32),
                    meta={
                        "dataset_name": "highd",
                        "data_version": "s3-b-data",
                        "recording_id": 1,
                        "vehicle_id": vehicle_offset + index,
                        "history_start_frame": 0,
                        "history_end_frame": window.history_steps - 1,
                        "future_start_frame": window.history_steps,
                        "future_end_frame": window.history_steps + window.future_steps - 1,
                        "split_id": "s3-b-split",
                        "split": split,
                        "client_id": f"rsu_{index + 1:02d}",
                    },
                )
            )
        datasets[split] = TrajectoryDataset(
            samples, split=split, split_id="s3-b-split", window_spec=window
        )
    scaler = TrainingCoordinateScaler().fit(
        np.array([[10.0, 20.0], [14.0, 28.0]], dtype=np.float32), split="train"
    )
    return (
        save_split_datasets(
            datasets,
            tmp_path / "processed",
            scaler=scaler,
            stats={"input_rows": 10},
            data_version="s3-b-data",
            data_config=config,
        ),
        config,
    )


def _bundle(processed_cache) -> ProcessedDataBundle:
    root, config = processed_cache
    return ProcessedDatasetReader().load(root, data_config=config)


def _loaders(data: ProcessedDataBundle, partition: PartitionManifest):
    return create_client_dataloaders(
        data,
        partition,
        contract=ModelContract(history_steps=75, future_steps=125),
        config=DataLoaderConfig(batch_size=2, num_workers=0, seed=17),
    )


def test_client_loaders_are_disjoint_stable_and_keep_holdout_tail_batches(processed_cache) -> None:
    bundle = _loaders(_bundle(processed_cache), _five_client_partition())

    assert tuple(bundle.clients) == tuple(f"rsu_{index:02d}" for index in range(1, 6))
    train_ids = []
    for client in bundle.clients.values():
        train_ids.extend(sample.meta["vehicle_id"] for sample in client.datasets["train"])
        assert [batch.history.shape[0] for batch in client.loaders["train"]] == [1]
    assert sorted(train_ids) == [0, 1, 2, 3, 4]
    assert len(train_ids) == len(set(train_ids))

    validation_counts = [
        len(bundle.clients[f"rsu_{index:02d}"].datasets["validation"])
        for index in range(1, 6)
    ]
    test_counts = [
        len(bundle.clients[f"rsu_{index:02d}"].datasets["test"])
        for index in range(1, 6)
    ]
    assert validation_counts == [1, 1, 1, 0, 0]
    assert test_counts == [1, 1, 0, 0, 0]
    assert bundle.clients["rsu_01"].profile.sample_visits(3) == 3


def test_holdout_anchor_outside_frozen_intervals_is_rejected(processed_cache) -> None:
    data = _bundle(processed_cache)
    first = data.datasets["validation"][0]
    changed = TrajectorySample(
        history=np.full_like(first.history, 99.0), future=first.future.copy(), meta=first.meta
    )
    validation = TrajectoryDataset(
        [changed, *data.datasets["validation"][1:]],
        split="validation",
        split_id=data.split_id,
        window_spec=data.datasets["validation"].window_spec,
    )
    altered = replace(data, datasets=MappingProxyType({**data.datasets, "validation": validation}))
    with pytest.raises(ClientDataError, match="outside frozen client intervals"):
        _loaders(altered, _five_client_partition())


def test_client_loaders_reject_vehicle_group_leakage_across_splits(processed_cache) -> None:
    data = _bundle(processed_cache)
    leaked = data.datasets["validation"][0]
    train_sample = data.datasets["train"][0]
    duplicate_group = TrajectorySample(
        history=leaked.history.copy(),
        future=leaked.future.copy(),
        meta={**leaked.meta, "vehicle_id": train_sample.meta["vehicle_id"]},
    )
    validation = TrajectoryDataset(
        [duplicate_group, *data.datasets["validation"][1:]],
        split="validation",
        split_id=data.split_id,
        window_spec=data.datasets["validation"].window_spec,
    )
    altered = replace(data, datasets=MappingProxyType({**data.datasets, "validation": validation}))
    with pytest.raises(ClientDataError, match="multiple splits"):
        _loaders(altered, _five_client_partition())


def test_empty_train_client_is_retained_without_padding_or_training_loader_shuffle(
    processed_cache,
) -> None:
    data = _bundle(processed_cache)
    partition = PartitionManifest(
        region_edges=(10.0, 15.0, 21.0),
        clients=(
            ClientPartition("rsu_01", 10.0, 15.0, 5, tuple(_group_id(index) for index in range(5))),
            ClientPartition("rsu_02", 15.0, 21.0, 0, ("reserved",)),
        ),
    )
    train = TrajectoryDataset(
        [
            TrajectorySample(
                history=sample.history.copy(),
                future=sample.future.copy(),
                meta={**sample.meta, "client_id": "rsu_01"},
            )
            for sample in data.datasets["train"]
        ],
        split="train",
        split_id=data.split_id,
        window_spec=data.datasets["train"].window_spec,
    )
    empty_train_data = replace(
        data, datasets=MappingProxyType({**data.datasets, "train": train})
    )
    bundle = _loaders(empty_train_data, partition)
    assert bundle.clients["rsu_02"].is_empty_train
    assert list(bundle.clients["rsu_02"].loaders["train"]) == []
    assert bundle.clients["rsu_02"].profile.train_sample_count == 0


def test_client_loaders_reject_drop_last_and_duplicate_train_samples(processed_cache) -> None:
    data = _bundle(processed_cache)
    with pytest.raises(ClientDataError, match="reject drop_last"):
        create_client_dataloaders(
            data,
            _five_client_partition(),
            contract=ModelContract(history_steps=75, future_steps=125),
            config=DataLoaderConfig(batch_size=2, num_workers=0, seed=17, drop_last=True),
        )
    duplicate_train = TrajectoryDataset(
        [*data.datasets["train"], data.datasets["train"][0]],
        split="train",
        split_id=data.split_id,
        window_spec=data.datasets["train"].window_spec,
    )
    duplicate_data = replace(
        data, datasets=MappingProxyType({**data.datasets, "train": duplicate_train})
    )
    with pytest.raises(ClientDataError, match="duplicate window identity"):
        _loaders(duplicate_data, _five_client_partition())
    swapped = TrajectoryDataset(
        [
            TrajectorySample(
                history=sample.history.copy(),
                future=sample.future.copy(),
                meta={
                    **sample.meta,
                    "client_id": (
                        "rsu_02"
                        if sample.meta["client_id"] == "rsu_01"
                        else sample.meta["client_id"]
                    ),
                },
            )
            for sample in data.datasets["train"]
        ],
        split="train",
        split_id=data.split_id,
        window_spec=data.datasets["train"].window_spec,
    )
    swapped_data = replace(data, datasets=MappingProxyType({**data.datasets, "train": swapped}))
    with pytest.raises(ClientDataError, match="does not match frozen group assignment"):
        _loaders(swapped_data, _five_client_partition())


def test_extreme_noniid_profile_and_export_are_rebuildable(processed_cache, tmp_path: Path) -> None:
    data = _bundle(processed_cache)
    train = TrajectoryDataset(
        [
            TrajectorySample(
                history=sample.history.copy(),
                future=sample.future.copy(),
                meta={**sample.meta, "client_id": "rsu_01"},
            )
            for sample in data.datasets["train"]
        ],
        split="train",
        split_id=data.split_id,
        window_spec=data.datasets["train"].window_spec,
    )
    altered = replace(data, datasets=MappingProxyType({**data.datasets, "train": train}))
    partition = PartitionManifest(
        region_edges=(10.0, 21.0),
        clients=(
            ClientPartition(
                "rsu_01", 10.0, 21.0, 5, tuple(_group_id(index) for index in range(5))
            ),
        ),
    )
    bundle = _loaders(altered, partition)
    profile = bundle.clients["rsu_01"].profile
    assert profile.vehicle_count == profile.train_sample_count == 5
    assert profile.validation_sample_count == 3
    assert profile.test_sample_count == 2
    assert profile.sample_visits(3) == 15
    train_batch_sizes = [
        batch.history.shape[0] for batch in bundle.clients["rsu_01"].loaders["train"]
    ]
    assert train_batch_sizes == [2, 2, 1]
    output = bundle.write_profiles(tmp_path / "client_profiles.json")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["partition_id"] == bundle.partition_id
    assert payload["clients"] == [profile.to_dict()]
    split_manifest = bundle.write_client_split_manifest(tmp_path / "client_split_manifest.json")
    split_payload = json.loads(split_manifest.read_text(encoding="utf-8"))
    assert split_payload["assignment_anchor"] == "history_last_x_meter"
    assert split_payload["scaler_id"] == bundle.scaler_id
    assert split_payload["out_of_range"] == 0
    assert split_payload["trainable_client_ids"] == ["rsu_01"]
