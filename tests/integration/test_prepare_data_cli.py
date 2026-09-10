"""End-to-end acceptance tests for the S1-A ``prepare-data`` entry point."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import yaml

from src.cli import main
from src.experiments import RunContext

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
    processed_index = json.loads(
        (output_root / "prepare-data-smoke" / "processed_index.json").read_text()
    )

    assert split_manifest["splits"]["train"]["sample_count"] > 0
    assert partition_manifest["num_clients"] >= 1
    assert run_manifest["split_id"] == split_manifest["split_id"]
    assert run_manifest["partition"] == partition_manifest
    assert run_manifest["artifacts"]["processed_index"] == "processed_index.json"
    assert processed_index["split_manifest"].endswith("/split_manifest.json")
    assert processed_index["partition_manifest"].endswith("/partition_manifest.json")
    assert processed_index["scaler"].endswith("/train/scaler.npz")
    assert set(processed_index["splits"]) == {"train", "validation", "test"}
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


def test_prepare_data_cli_checks_duplicate_run_before_writing_processed_data(
    tmp_path: Path, config_bundle: dict[str, dict[str, object]], capsys, monkeypatch
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    data_path = _write_data_config(tmp_path, config_bundle)
    output_root = tmp_path / "runs"
    blocked_run = output_root / "blocked-run"
    blocked_run.mkdir(parents=True)
    marker = blocked_run / "keep.txt"
    marker.write_text("existing run", encoding="utf-8")

    assert (
        main(
            [
                "prepare-data",
                "--data",
                str(data_path),
                "--output-root",
                _relative(output_root),
                "--run-id",
                "blocked-run",
            ]
        )
        == 2
    )
    assert "already exists" in capsys.readouterr().err
    assert marker.read_text(encoding="utf-8") == "existing run"
    assert not (tmp_path / "processed").exists()


def test_prepare_data_rolls_back_new_run_and_processed_split_on_manifest_failure(
    tmp_path: Path, config_bundle: dict[str, dict[str, object]], capsys, monkeypatch
) -> None:
    monkeypatch.chdir(PROJECT_ROOT)
    data_path = _write_data_config(tmp_path, config_bundle)
    output_root = tmp_path / "runs"

    def fail_manifest(self: RunContext, split_name: str, manifest: object) -> None:
        raise ValueError("injected manifest failure")

    monkeypatch.setattr(RunContext, "add_split_manifest", fail_manifest)
    assert (
        main(
            [
                "prepare-data",
                "--data",
                str(data_path),
                "--output-root",
                _relative(output_root),
                "--run-id",
                "rollback-run",
            ]
        )
        == 2
    )
    assert "injected manifest failure" in capsys.readouterr().err
    assert not (output_root / "rollback-run").exists()
    processed_root = tmp_path / "processed"
    assert not processed_root.exists() or not any(processed_root.iterdir())
