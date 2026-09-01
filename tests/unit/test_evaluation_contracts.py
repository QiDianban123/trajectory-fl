"""Tests for E's result schema and plotting interfaces."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.result_store import CSV_FIELDS, ResultRecord, write_csv, write_json
from src.evaluation.visualization import (
    plot_convergence,
    plot_mode_comparison,
    plot_trajectory,
)


def _record(mode: str = "centralized") -> ResultRecord:
    return ResultRecord(
        run_id=f"run-{mode}",
        code_sha="abc123",
        seed=42,
        split_id="highd-split-42",
        mode=mode,  # type: ignore[arg-type]
        sample_count=4,
        ade=1.25,
        fde=2.5,
        total_seconds=3.0,
        artifact_paths={"trajectory": "figures/trajectory.png"},
    )


def test_result_record_contains_canonical_metrics_and_artifact_fields() -> None:
    payload = _record().to_dict()
    assert payload["run_id"] == "run-centralized"
    assert payload["code_sha"] == "abc123"
    assert payload["coordinate_unit"] == "meter"
    assert payload["metrics"] == {"ade": 1.25, "fde": 2.5}
    assert payload["artifacts"] == {"trajectory": "figures/trajectory.png"}


def test_result_serialization_writes_json_and_flat_csv(tmp_path: Path) -> None:
    json_path = write_json(_record(), tmp_path / "nested" / "metrics.json")
    csv_path = write_csv([_record(), _record("federated")], tmp_path / "nested" / "metrics.csv")
    assert json.loads(json_path.read_text(encoding="utf-8"))["metrics"]["ade"] == 1.25
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert tuple(rows[0]) == CSV_FIELDS
    assert rows[1]["mode"] == "federated"
    assert "trajectory.png" in rows[0]["artifact_paths"]


def test_result_record_rejects_non_meter_or_invalid_mode() -> None:
    with pytest.raises(ValueError, match="physical coordinates in meters"):
        ResultRecord(**{**_record().__dict__, "coordinate_unit": "normalized"})
    with pytest.raises(ValueError, match="mode must be one"):
        ResultRecord(**{**_record().__dict__, "mode": "smoke"})


def test_visualizations_create_parent_directories_and_files(tmp_path: Path) -> None:
    history = np.array([[0.0, 0.0], [1.0, 0.0]])
    future = np.array([[2.0, 0.0], [3.0, 1.0]])
    prediction = np.array([[2.0, 0.5], [3.0, 1.5]])
    trajectory_path = plot_trajectory(
        history, future, prediction, tmp_path / "figures/trajectory.png"
    )
    convergence_path = plot_convergence([0, 1], [1.0, 0.5], tmp_path / "figures/convergence.png")
    comparison_path = plot_mode_comparison(
        [_record(), _record("federated")], tmp_path / "figures/comparison.png"
    )
    assert trajectory_path.is_file()
    assert convergence_path.is_file()
    assert comparison_path.is_file()


def test_visualizations_reject_shape_and_output_path_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="shapes must match"):
        plot_trajectory(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            np.zeros((3, 2)),
            tmp_path / "trajectory.png",
        )
    with pytest.raises(ValueError, match="filename with an extension"):
        plot_convergence([0], [1.0], tmp_path / "no-extension")
