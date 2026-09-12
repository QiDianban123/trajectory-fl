from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.adapters import TrajectorySample
from src.data.cache import processed_cache_key, semantic_config_digest
from src.data.dataset import TrajectoryDataset, save_split_datasets
from src.data.loading import (
    DataLoaderConfig,
    ProcessedDataError,
    ProcessedDatasetReader,
    create_dataloaders,
)
from src.data.preprocess import TrainingCoordinateScaler, WindowSpec
from src.models.base import ModelContract

DATA_VERSION = "highd-s2-b-data"
SPLIT_ID = "highd-s2-b-split"
SPLIT_COUNTS = {"train": 5, "validation": 3, "test": 2}


def _sample(
    *, split: str, index: int, window: WindowSpec, data_version: str = DATA_VERSION
) -> TrajectorySample:
    return TrajectorySample(
        history=np.full((window.history_steps, 2), index + 0.25, dtype=np.float32),
        future=np.full((window.future_steps, 2), index + 0.75, dtype=np.float32),
        meta={
            "dataset_name": "highd",
            "data_version": data_version,
            "recording_id": 1,
            "vehicle_id": index,
            "history_start_frame": 0,
            "history_end_frame": window.history_steps - 1,
            "future_start_frame": window.history_steps,
            "future_end_frame": window.history_steps + window.future_steps - 1,
            "split_id": SPLIT_ID,
            "split": split,
            "client_id": f"rsu_{index + 1:02d}",
        },
    )


def _write_cache(
    root: Path,
    data_config: dict[str, object],
    *,
    counts: dict[str, int] | None = None,
) -> Path:
    sequence = data_config["sequence"]
    window = WindowSpec(**sequence)
    resolved_counts = SPLIT_COUNTS if counts is None else counts
    datasets = {
        split: TrajectoryDataset(
            [_sample(split=split, index=index, window=window) for index in range(count)],
            split=split,
            split_id=SPLIT_ID,
            window_spec=window,
        )
        for split, count in resolved_counts.items()
    }
    scaler = TrainingCoordinateScaler().fit(
        np.array([[10.0, 20.0], [14.0, 28.0]], dtype=np.float32), split="train"
    )
    return save_split_datasets(
        datasets,
        root,
        scaler=scaler,
        stats={"input_rows": 100, "rejected_rows": 0},
        data_version=DATA_VERSION,
        data_config=data_config,
    )


@pytest.fixture
def processed_cache(tmp_path: Path, config_bundle) -> tuple[Path, dict[str, object]]:
    data_config = deepcopy(config_bundle["data"])
    return _write_cache(tmp_path / "processed", data_config), data_config


def _manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _refresh_checksum(root: Path, artifact: Path) -> None:
    manifest_path = root / "split_manifest.json"
    manifest = _manifest(manifest_path)
    relative = artifact.relative_to(root).as_posix()
    manifest["artifacts"][relative]["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    _write_manifest(manifest_path, manifest)


def test_cache_key_tracks_identity_and_semantic_configuration(config_bundle) -> None:
    config = deepcopy(config_bundle["data"])
    key = processed_cache_key(data_version=DATA_VERSION, split_id=SPLIT_ID, data_config=config)
    assert key == processed_cache_key(
        data_version=DATA_VERSION, split_id=SPLIT_ID, data_config=deepcopy(config)
    )

    path_only = deepcopy(config)
    path_only["dataset"]["raw_dir"] = "somewhere/else"
    assert semantic_config_digest(path_only) == semantic_config_digest(config)

    changed = deepcopy(config)
    changed["partition"]["num_clients"] = 6
    changed["partition"]["region_edges"] = None
    assert semantic_config_digest(changed) != semantic_config_digest(config)
    assert (
        processed_cache_key(data_version=DATA_VERSION, split_id=SPLIT_ID, data_config=changed)
        != key
    )
    assert (
        processed_cache_key(
            data_version=DATA_VERSION + "-new", split_id=SPLIT_ID, data_config=config
        )
        != key
    )
    assert (
        processed_cache_key(
            data_version=DATA_VERSION, split_id=SPLIT_ID + "-new", data_config=config
        )
        != key
    )


def test_reader_loads_complete_cache_and_inverse_transform_hook(processed_cache) -> None:
    root, config = processed_cache
    reader = ProcessedDatasetReader()
    data = reader.load(
        root,
        data_config=config,
        expected_data_version=DATA_VERSION,
        expected_split_id=SPLIT_ID,
    )

    assert {split: len(dataset) for split, dataset in data.datasets.items()} == SPLIT_COUNTS
    assert data.data_version == DATA_VERSION
    assert data.split_id == SPLIT_ID
    assert data.cache_key.startswith("processed-v1-")
    assert not data.datasets["train"][0].history.flags.writeable
    data.datasets["train"][0].meta["vehicle_id"] = 999  # type: ignore[index]
    reloaded = reader.load(root, data_config=config)
    assert reloaded.datasets["train"][0].meta["vehicle_id"] != 999
    with pytest.raises(TypeError):
        data.stats["input_rows"] = 999  # type: ignore[index]
    normalized = np.array([[[0.0, 0.0], [1.0, -1.0]]], dtype=np.float32)
    np.testing.assert_allclose(
        data.inverse_transform(normalized),
        np.array([[[12.0, 24.0], [14.0, 20.0]]], dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("expected_data_version", "old-version", "data_version mismatch"),
        ("expected_split_id", "wrong-split", "split_id mismatch"),
    ],
)
def test_reader_rejects_expected_identity_mismatch(
    processed_cache, argument, value, message
) -> None:
    root, config = processed_cache
    with pytest.raises(ProcessedDataError, match=message):
        ProcessedDatasetReader().load(root, data_config=config, **{argument: value})


def test_reader_rejects_legacy_and_semantically_stale_cache(processed_cache) -> None:
    root, config = processed_cache
    stale_config = deepcopy(config)
    stale_config["sequence"]["stride"] = 2
    with pytest.raises(ProcessedDataError, match="semantic configuration"):
        ProcessedDatasetReader().load(root, data_config=stale_config)

    manifest_path = root / "split_manifest.json"
    manifest = _manifest(manifest_path)
    manifest.pop("cache_identity")
    _write_manifest(manifest_path, manifest)
    with pytest.raises(ProcessedDataError, match="legacy or missing"):
        ProcessedDatasetReader().load(root, data_config=config)


def test_reader_rejects_empty_split(tmp_path, config_bundle) -> None:
    config = deepcopy(config_bundle["data"])
    root = _write_cache(tmp_path / "empty", config, counts={"train": 5, "validation": 0, "test": 2})
    with pytest.raises(ProcessedDataError, match="validation.*at least one sample"):
        ProcessedDatasetReader().load(root, data_config=config)


def test_reader_rejects_corrupt_artifact_and_invalidates_memory_cache(processed_cache) -> None:
    root, config = processed_cache
    reader = ProcessedDatasetReader()
    first = reader.load(root, data_config=config)
    assert reader.load(root, data_config=config) is not first

    with (root / "validation" / "samples.npz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ProcessedDataError, match="checksum mismatch"):
        reader.load(root, data_config=config)


def test_reader_rejects_inconsistent_child_manifest(processed_cache) -> None:
    root, config = processed_cache
    child_path = root / "test" / "manifest.json"
    child = _manifest(child_path)
    child["split_id"] = "another-split"
    _write_manifest(child_path, child)
    _refresh_checksum(root, child_path)

    with pytest.raises(ProcessedDataError, match="cannot load processed test split"):
        ProcessedDatasetReader().load(root, data_config=config)


def test_reader_rejects_wrong_array_dtype_even_with_updated_checksum(processed_cache) -> None:
    root, config = processed_cache
    samples_path = root / "train" / "samples.npz"
    with np.load(samples_path, allow_pickle=False) as values:
        history = values["history"].astype(np.float64)
        future = values["future"]
    np.savez_compressed(samples_path, history=history, future=future)
    _refresh_checksum(root, samples_path)

    with pytest.raises(ProcessedDataError, match="history must use float32"):
        ProcessedDatasetReader().load(root, data_config=config)


def test_reader_rejects_different_scaler_between_splits(processed_cache) -> None:
    root, config = processed_cache
    scaler_path = root / "validation" / "scaler.npz"
    np.savez_compressed(
        scaler_path,
        mean=np.array([99.0, 99.0], dtype=np.float32),
        scale=np.array([1.0, 1.0], dtype=np.float32),
    )
    _refresh_checksum(root, scaler_path)

    with pytest.raises(ProcessedDataError, match="do not share the same train scaler"):
        ProcessedDatasetReader().load(root, data_config=config)


def test_reader_rejects_manifest_path_escape(processed_cache) -> None:
    root, config = processed_cache
    manifest_path = root / "split_manifest.json"
    manifest = _manifest(manifest_path)
    manifest["splits"]["train"]["path"] = "../outside"
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ProcessedDataError, match="escapes"):
        ProcessedDatasetReader().load(root, data_config=config)


def test_dataloaders_are_deterministic_and_preserve_batch_contract(processed_cache) -> None:
    root, config = processed_cache
    data = ProcessedDatasetReader().load(root, data_config=config)
    contract = ModelContract(history_steps=75, future_steps=125)
    loader_config = DataLoaderConfig(batch_size=2, num_workers=0, seed=123)
    first = create_dataloaders(data, contract=contract, config=loader_config)
    second = create_dataloaders(data, contract=contract, config=loader_config)

    first_ids = [meta["vehicle_id"] for batch in first["train"] for meta in batch.meta]
    second_ids = [meta["vehicle_id"] for batch in second["train"] for meta in batch.meta]
    assert first_ids == second_ids
    validation_batches = list(first["validation"])
    assert [batch.history.shape[0] for batch in validation_batches] == [2, 1]
    assert [meta["vehicle_id"] for batch in validation_batches for meta in batch.meta] == [0, 1, 2]
    for split_loader in create_dataloaders(data, contract=contract, config=loader_config).values():
        for batch in split_loader:
            batch.validate(contract)
            assert batch.history.dtype == batch.future.dtype == torch.float32
            assert all(meta["data_version"] == DATA_VERSION for meta in batch.meta)
            assert all(meta["split_id"] == SPLIT_ID for meta in batch.meta)


def test_dataloader_config_reads_frozen_bundle_fields(config_bundle) -> None:
    config = DataLoaderConfig.from_config_bundle(config_bundle)
    assert config.batch_size == 32
    assert config.num_workers == 0
    assert config.seed == 42


@pytest.mark.parametrize(
    "values",
    [
        {"batch_size": 0, "num_workers": 0, "seed": 0},
        {"batch_size": 1, "num_workers": -1, "seed": 0},
        {"batch_size": 1, "num_workers": 0, "seed": -1},
        {
            "batch_size": 1,
            "num_workers": 0,
            "seed": 0,
            "persistent_workers": True,
        },
    ],
)
def test_dataloader_config_rejects_invalid_runtime_values(values) -> None:
    with pytest.raises((TypeError, ValueError)):
        DataLoaderConfig(**values)


def test_holdout_loaders_never_drop_samples(processed_cache) -> None:
    root, config = processed_cache
    data = ProcessedDatasetReader().load(root, data_config=config)
    loaders = create_dataloaders(
        data,
        contract=ModelContract(history_steps=75, future_steps=125),
        config=DataLoaderConfig(batch_size=2, num_workers=0, seed=7, drop_last=True),
    )

    assert sum(batch.history.shape[0] for batch in loaders["train"]) == 4
    assert sum(batch.history.shape[0] for batch in loaders["validation"]) == 3
    assert sum(batch.history.shape[0] for batch in loaders["test"]) == 2
