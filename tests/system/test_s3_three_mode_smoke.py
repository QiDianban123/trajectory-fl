"""CI regression for the production, no-argument S3 three-mode smoke."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_real_three_mode_smoke_writes_comparable_manifests() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/run_three_mode_smoke.py"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr + completed.stdout
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["status"] == "completed"
    manifests = {name: PROJECT_ROOT / path for name, path in payload["manifests"].items()}
    for path in manifests.values():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        assert manifest["status"] == "completed"
        assert manifest["fairness"]["comparable"] is True
        assert manifest["summary"] is not None
