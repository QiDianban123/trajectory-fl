"""Unit tests for the Day 2 YAML configuration contracts."""

from pathlib import Path

import pytest

from src.utils.config import ConfigError, load_and_validate, validate_config_bundle


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
  required_columns: [id, frame, x, y]
sequence:
  history_steps: 75
  future_steps: 125
  stride: 1
  coordinate_dimension: 2
split:
  group_by: vehicle_id
  train: 0.7
  validation: 0.15
  test: 0.15
  seed: 42
normalization:
  method: standard
  fit_split: test
  per_axis: true
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="fit_split must be train"):
        load_and_validate(path, "data")
