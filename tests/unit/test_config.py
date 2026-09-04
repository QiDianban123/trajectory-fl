"""Unit tests for the Day 2 YAML configuration contracts."""

from pathlib import Path

import pytest

from src.utils.config import (
    ConfigError,
    load_and_validate,
    load_yaml,
    validate_config,
    validate_config_bundle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_default_bundle_is_valid() -> None:
    bundle = validate_config_bundle(
        PROJECT_ROOT / "configs/data.yaml",
        PROJECT_ROOT / "configs/model.yaml",
        PROJECT_ROOT / "configs/experiments/smoke.yaml",
    )
    assert bundle["data"]["sequence"]["future_steps"] == 125
    assert bundle["model"]["model"]["output_size"] == 2
    assert bundle["experiment"]["run"]["mode"] == "smoke"


def test_empty_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="root must be a mapping"):
        load_and_validate(path, "data")


def test_training_only_normalization_is_required(tmp_path: Path) -> None:
    path = tmp_path / "data.yaml"
    path.write_text(
        """
schema_version: 1
dataset:
  name: highd
  raw_dir: data/raw
  processed_dir: data/processed
  sample_dir: data/sample
  frame_rate_hz: 25
  coordinate_system: highd_road_local
  coordinate_unit: meter
  required_columns: [id, frame, x, y]
sequence:
  history_steps: 75
  future_steps: 125
  stride: 1
  coordinate_dimension: 2
split:
  strategy: group_then_window
  group_by: vehicle_id
  require_group_disjointness: true
  train: 0.7
  validation: 0.15
  test: 0.15
  seed: 42
partition:
  num_clients: 5
  axis: x
  client_id_prefix: rsu_
  region_edges: null
  min_samples_per_client: 1
normalization:
  method: standard
  fit_split: test
  per_axis: true
  statistics_artifact: scaler.npz
preprocessing:
  time_order: strict_increasing
  missing_required_policy: reject_sample
  nonfinite_coordinate_policy: reject_sample
  duplicate_frame_policy: reject_track
  minimum_track_frames: 200
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="fit_split must be train"):
        load_and_validate(path, "data")


def test_data_config_requires_partition_section() -> None:
    config = load_yaml(PROJECT_ROOT / "configs/data.yaml")
    del config["partition"]
    with pytest.raises(ConfigError, match="partition is missing required keys"):
        validate_config(config, "data")


def test_partition_region_edges_schema_is_validated() -> None:
    base = load_yaml(PROJECT_ROOT / "configs/data.yaml")
    wrong_length = dict(base)
    wrong_length["partition"] = dict(base["partition"], region_edges=[0.0, 10.0])
    with pytest.raises(ConfigError, match="num_clients \\+ 1"):
        validate_config(wrong_length, "data")

    unordered = dict(base)
    unordered["partition"] = dict(base["partition"], region_edges=[0.0, 30.0, 20.0, 40.0, 50.0, 60.0])
    with pytest.raises(ConfigError, match="strictly increasing"):
        validate_config(unordered, "data")

    negative_min = dict(base)
    negative_min["partition"] = dict(base["partition"], min_samples_per_client=0)
    with pytest.raises(ConfigError, match="min_samples_per_client"):
        validate_config(negative_min, "data")
