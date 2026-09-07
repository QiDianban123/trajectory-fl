"""Tests for F's reproducibility, logging, output-path, and fixture contracts."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.logging import configure_run_logger
from src.utils.paths import PathSafetyError, create_run_paths, resolve_within
from src.utils.seed import set_global_seed


def test_global_seed_reproduces_python_numpy_and_torch_sequences() -> None:
    first_report = set_global_seed(123)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    second_report = set_global_seed(123)
    second = (random.random(), np.random.rand(), torch.rand(1).item())
    assert first == second
    assert first_report.seed == 123
    assert second_report.torch_available


def test_global_seed_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="seed must be an integer"):
        set_global_seed(-1)


def test_run_paths_create_fixed_layout_without_overwrite(temporary_output_root: Path) -> None:
    paths = create_run_paths(temporary_output_root, "run-20260901-seed42")
    assert paths.run_dir.is_dir()
    assert paths.checkpoint_dir.is_dir()
    assert paths.figures_dir.is_dir()
    assert paths.log_path.parent == paths.run_dir
    with pytest.raises(FileExistsError, match="already exists"):
        create_run_paths(temporary_output_root, "run-20260901-seed42")


def test_output_paths_reject_traversal_and_absolute_artifacts(temporary_output_root: Path) -> None:
    temporary_output_root.mkdir()
    with pytest.raises(PathSafetyError, match="run_id"):
        create_run_paths(temporary_output_root, "../escape")
    with pytest.raises(PathSafetyError, match="escapes"):
        resolve_within(temporary_output_root, "../outside.json")
    with pytest.raises(PathSafetyError, match="relative"):
        resolve_within(temporary_output_root, temporary_output_root.resolve() / "outside.json")


def test_run_logger_writes_structured_json_lines(temporary_output_root: Path) -> None:
    paths = create_run_paths(temporary_output_root, "run-log")
    logger = configure_run_logger("run-log", paths.log_path, level=logging.INFO)
    logger.info("training configured", extra={"split_id": "split-42"})
    for handler in logger.handlers:
        handler.flush()

    payload = json.loads(paths.log_path.read_text(encoding="utf-8"))
    assert payload["message"] == "training configured"
    assert payload["run_id"] == "run-log"
    assert payload["split_id"] == "split-42"


def test_config_fixture_is_independent_and_has_all_day2_sections(
    config_bundle: dict[str, dict[str, object]]
) -> None:
    config_bundle["data"]["schema_version"] = 99
    assert config_bundle["model"]["schema_version"] == 1
    assert "preprocessing" in config_bundle["data"]
    assert "checkpoint" in config_bundle["model"]["training"]
