"""Validated schema-v2 manifest comparison tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.comparison import compare_run_manifests
from src.evaluation.result_store import ResultRecord

IDENTITY = {
    "data_version": "data",
    "split_id": "split",
    "partition_id": "partition",
    "scaler_id": "scaler",
    "model_config_digest": "model",
    "seed": 7,
    "initial_state_id": "initial",
    "metric_schema": "ade_fde_meter_v1",
}


def _manifest(root: Path, mode: str, *, comparable: bool = True) -> Path:
    run = root / mode
    run.mkdir(parents=True)
    budget = {
        "sample_visits": 30,
        "local_epochs": 1,
        "rounds": 1,
        "selected_clients": ["rsu_01", "rsu_02"],
    }
    summary = ResultRecord(
        run_id=f"run-{mode}",
        code_sha="sha",
        seed=7,
        split_id="split",
        mode=mode,  # type: ignore[arg-type]
        sample_count=10,
        ade=1.0,
        fde=2.0,
        total_seconds=3.0,
    ).to_dict()
    payload = {
        "schema_version": 2,
        "run_id": f"run-{mode}",
        "mode": mode,
        "status": "completed",
        "identity": IDENTITY,
        "fairness": {
            **IDENTITY,
            "planned_budget": budget,
            "actual_budget": budget,
            "comparable": comparable,
            "reason": None if comparable else "budget mismatch",
        },
        "summary": summary,
    }
    path = run / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_compare_three_canonical_manifests_creates_figure(tmp_path: Path) -> None:
    paths = [_manifest(tmp_path, mode) for mode in ("centralized", "local_only", "federated")]
    output = compare_run_manifests(paths, tmp_path / "figures" / "comparison.png")
    assert [record.mode for record in output.records] == [
        "centralized",
        "local_only",
        "federated",
    ]
    assert output.figure_path.is_file()


def test_compare_rejects_incomplete_or_unfair_matrix(tmp_path: Path) -> None:
    paths = [_manifest(tmp_path, mode) for mode in ("centralized", "local_only")]
    with pytest.raises(ValueError, match="exactly three"):
        compare_run_manifests(paths, tmp_path / "few.png")

    paths.append(_manifest(tmp_path, "federated", comparable=False))
    with pytest.raises(ValueError, match="fairness must explicitly be comparable"):
        compare_run_manifests(paths, tmp_path / "unfair.png")


def test_compare_accepts_historical_flat_summary_for_compatibility(tmp_path: Path) -> None:
    paths = [_manifest(tmp_path, mode) for mode in ("centralized", "local_only", "federated")]
    payload = json.loads(paths[1].read_text(encoding="utf-8"))
    record = payload["summary"]
    payload["summary"] = {
        **{
            key: value
            for key, value in record.items()
            if key not in {"metrics", "timing_seconds", "artifacts"}
        },
        "ade": record["metrics"]["ade"],
        "fde": record["metrics"]["fde"],
        "total_seconds": record["timing_seconds"]["total"],
        "artifact_paths": record["artifacts"],
    }
    paths[1].write_text(json.dumps(payload), encoding="utf-8")
    assert compare_run_manifests(paths, tmp_path / "legacy.png").figure_path.is_file()
