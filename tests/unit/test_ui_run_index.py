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
