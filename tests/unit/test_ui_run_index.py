"""Read-only saved-run index tests."""

from __future__ import annotations

import json
from pathlib import Path

from src.ui.run_index import discover_runs


def test_run_index_reads_result_facts_and_ignores_escape_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "run-42"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-42",
                "split_id": "split-42",
                "data_version": "v1",
                "artifacts": {"metrics": "metrics.json", "unsafe": "../outside.json"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "mode": "centralized",
                "seed": 42,
                "sample_count": 4,
                "metrics": {"ade": 1.2, "fde": 2.3},
                "timing_seconds": {"total": 3.4},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "training_history.json").write_text('{"best_epoch": 2}', encoding="utf-8")

    runs = discover_runs(tmp_path)

    assert len(runs) == 1
    assert runs[0].ade == 1.2
    assert runs[0].best_epoch == 2
    assert set(runs[0].artifacts) == {"metrics"}


def test_run_index_reads_s3_v2_summary_and_structured_facts(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "fed-42"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "fed-42",
                "status": "completed",
                "identity": {"split_id": "split", "data_version": "data"},
                "fairness": {"comparable": True},
                "clients": [{"client_id": "rsu_01", "status": "completed"}],
                "rounds": [{"round_index": 0, "status": "completed"}],
                "summary": {
                    "status": "completed",
                    "mode": "federated",
                    "seed": 7,
                    "sample_count": 2,
                    "metrics": {"ade": 1.0, "fde": 2.0},
                    "timing_seconds": {"total": 3.0},
                },
                "artifacts": {"results": "results.csv"},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "results.csv").write_text("client_id\nrsu_01\n", encoding="utf-8")
    run = discover_runs(tmp_path)[0]
    assert (run.mode, run.ade, run.fairness, len(run.clients), len(run.rounds)) == (
        "federated",
        1.0,
        {"comparable": True},
        1,
        1,
    )


def test_v2_summary_overrides_intermediate_metrics_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "local-42"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "local-42",
                "identity": {"split_id": "s"},
                "summary": {
                    "status": "completed",
                    "mode": "local_only",
                    "seed": 1,
                    "sample_count": 1,
                    "ade": 2.0,
                    "fde": 3.0,
                    "total_seconds": 4.0,
                },
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text('{"status":"failed"}', encoding="utf-8")
    run = discover_runs(tmp_path)[0]
    assert (run.status, run.ade, run.fde, run.total_seconds) == ("completed", 2.0, 3.0, 4.0)
