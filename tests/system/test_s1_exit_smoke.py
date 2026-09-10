"""System acceptance for the S1 one-command data pipeline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_s1_smoke_command_builds_five_rsu_artifacts(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_s1_smoke.py"),
            "--workspace",
            str(tmp_path / "s1-smoke"),
        ],
        cwd=project_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert "S1 smoke passed" in completed.stdout
    assert "rsu_01, rsu_02, rsu_03, rsu_04, rsu_05" in completed.stdout
    assert "Split groups: disjoint" in completed.stdout
    assert (tmp_path / "s1-smoke" / "runs" / "s1-smoke" / "manifest.json").is_file()
