"""End-to-end acceptance tests for the S1-A ``prepare-data`` entry point."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from src.cli import main

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _write_tracks(raw_dir: Path) -> None:
    rows: list[dict[str, float | int]] = []
    for vehicle_id in range(1, 7):
        for frame in range(200):
            rows.append(
                {
                    "Track ID": vehicle_id,
                    "Frame ID": frame,
                    "x Position": float(vehicle_id * 500 + frame),
                    "y Position": float(vehicle_id + frame / 10),
                }
            )
    raw_dir.mkdir()
    pd.DataFrame(rows).to_csv(raw_dir / "01_tracks.csv", index=False)


def _write_data_config(tmp_path: Path, config_bundle: dict[str, dict[str, object]]) -> Path:
    raw_dir = tmp_path / "raw"
    _write_tracks(raw_dir)
    config = deepcopy(config_bundle["data"])
    dataset = config["dataset"]
    partition = config["partition"]
    assert isinstance(dataset, dict)
    assert isinstance(partition, dict)
    dataset["raw_dir"] = _relative(raw_dir)
    dataset["processed_dir"] = _relative(tmp_path / "processed")
    partition["num_clients"] = 3
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_prepare_data_cli_builds_split_partition_and_run_manifests(
    tmp_path: Path, config_bundle: dict[str, dict[str, object]], capsys, monkeypatch
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    data_path = _write_data_config(tmp_path, config_bundle)
    output_root = tmp_path / "runs"

    assert (
        main(
            [
                "prepare-data",
                "--data",
                str(data_path),
                "--output-root",
                _relative(output_root),
                "--run-id",
                "prepare-data-smoke",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "Prepared highD data" in output
    assert "Samples:" in output

    processed = tmp_path / "processed"
    split_directory = next(processed.iterdir())
    split_manifest = json.loads((split_directory / "split_manifest.json").read_text())
    partition_manifest = json.loads((split_directory / "partition_manifest.json").read_text())
    run_manifest = json.loads((output_root / "prepare-data-smoke" / "manifest.json").read_text())

    assert split_manifest["splits"]["train"]["sample_count"] > 0
    assert partition_manifest["num_clients"] >= 1
    assert run_manifest["split_id"] == split_manifest["split_id"]
    assert run_manifest["partition"] == partition_manifest
    train_metadata = json.loads((split_directory / "train" / "manifest.json").read_text())[
        "metadata"
    ]
    assert train_metadata and all("client_id" in metadata for metadata in train_metadata)


def test_prepare_data_cli_reports_missing_raw_directory(
    tmp_path: Path, config_bundle: dict[str, dict[str, object]], capsys, monkeypatch
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    config = deepcopy(config_bundle["data"])
    dataset = config["dataset"]
    assert isinstance(dataset, dict)
    dataset["raw_dir"] = _relative(tmp_path / "missing-raw")
    dataset["processed_dir"] = _relative(tmp_path / "processed")
    data_path = tmp_path / "missing-raw.yaml"
    data_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    assert main(["prepare-data", "--data", str(data_path)]) == 2
    assert "raw directory does not exist" in capsys.readouterr().err
