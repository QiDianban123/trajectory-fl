"""S3-B client-local loader, frozen-holdout, and accounting tests."""

from __future__ import annotations

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
from src.data.partition import ClientPartition, PartitionManifest
from src.data.preprocess import TrainingCoordinateScaler, WindowSpec
from src.models.base import ModelContract


def _five_client_partition() -> PartitionManifest:
    edges = (10.0, 13.0, 15.0, 17.0, 19.0, 21.0)
    return PartitionManifest(
        region_edges=edges,
        clients=tuple(
            ClientPartition(f"rsu_{index:02d}", edges[index - 1], edges[index], 1, (index - 1,))
            for index in range(1, 6)
        ),
    )


@pytest.fixture
def processed_cache(tmp_path: Path, config_bundle) -> tuple[Path, dict[str, object]]:
    config = deepcopy(config_bundle["data"])
    window = WindowSpec(**config["sequence"])
    datasets = {}
    for split, count in (("train", 5), ("validation", 3), ("test", 2)):
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
                        "vehicle_id": index,
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
