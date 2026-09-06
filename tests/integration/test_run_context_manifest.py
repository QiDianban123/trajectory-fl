"""Integrate S1-G manifests with S1-D partition output."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.partition import GroupExtent, PartitionConfig, partition_train_groups
from src.experiments import MANIFEST_SCHEMA_VERSION, RunContext


def test_complete_manifest_is_rebuildable_and_deterministically_serialized(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data" / "processed"
    data_dir.mkdir(parents=True)
    samples_path = data_dir / "samples.npz"
    samples_path.write_bytes(b"deterministic sample bytes")

    groups = [
        GroupExtent("vehicle-b", 10.0, 20.0, 3),
        GroupExtent("vehicle-a", 0.0, 8.0, 2),
    ]
    partition = partition_train_groups(groups, PartitionConfig(num_clients=2))
    context = RunContext(
        {"dataset": "highd", "seed": 42},
        tmp_path,
        run_id="s1-g-smoke",
        code_sha="b" * 40,
    )
    context.set_data_identity(data_version="highd-sample-v1", split_id="highd-split-42")
    context.record_data_file(samples_path)
    context.add_split_manifest(
        "train",
        {"group_ids": ["vehicle-a", "vehicle-b"], "sample_count": 5},
    )
    context.add_partition_manifest(partition)
    context.add_artifact("samples", "data/samples.npz")
    context.export_data_profile({"total_samples": 5, "train_samples": 5})

    manifest_path = context.export_manifest()
    first_bytes = manifest_path.read_bytes()
    assert context.export_manifest().read_bytes() == first_bytes
    manifest = json.loads(first_bytes)

    assert manifest["schema_version"] == MANIFEST_SCHEMA_VERSION
    assert manifest["code_sha"] == "b" * 40
    assert manifest["git_sha"] == manifest["code_sha"]
    assert manifest["data_version"] == "highd-sample-v1"
    assert manifest["split_id"] == "highd-split-42"
    assert manifest["data_files"]["data/processed/samples.npz"]
    assert manifest["splits"]["train"]["sample_count"] == 5
    assert manifest["partition"] == partition.to_mapping()
    assert manifest["artifacts"] == {
        "config": "config_snapshot.json",
        "data_profile": "data_profile.json",
        "manifest": "manifest.json",
        "samples": "data/samples.npz",
    }
