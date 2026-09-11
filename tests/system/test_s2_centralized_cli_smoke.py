"""One-command anonymous centralized CLI smoke."""

from __future__ import annotations

import shutil
import subprocess
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_one_command_centralized_smoke() -> None:
    relative_workspace = Path("outputs") / f"pytest-s2-a-{uuid.uuid4().hex}"
    workspace = PROJECT_ROOT / relative_workspace
    try:
        completed = subprocess.run(
            [
                sys.executable,
                "scripts/run_centralized_smoke.py",
                "--workspace",
                str(relative_workspace),
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert completed.returncode == 0, completed.stderr
        assert "Centralized smoke loss:" in completed.stdout
        assert "Centralized smoke passed:" in completed.stdout
        run_dir = workspace / "centralized-runs" / "centralized-smoke"
        for relative_path in (
            "config_snapshot.json",
            "metadata.json",
            "train.log",
            "checkpoints/best.pt",
            "training_history.json",
            "predictions.npz",
            "metrics.json",
            "metrics.csv",
            "figures/loss_curve.png",
            "figures/prediction_trajectory.png",
        ):
            artifact = run_dir / relative_path
            assert artifact.is_file() and artifact.stat().st_size > 0
    finally:
        shutil.rmtree(workspace, ignore_errors=True)
