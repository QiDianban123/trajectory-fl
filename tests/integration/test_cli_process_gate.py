"""Exercise the available config CLI in a real process, not the D2 placeholder.

This is not prepare-data acceptance; its missing integration is tracked in the
S1_F review report until A/B publish the production pipeline.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("bad_content", [None, "", "- invalid-root"])
def test_config_cli_reports_bad_path_or_yaml(tmp_path: Path, bad_content: str | None) -> None:
    config_path = tmp_path / "bad-data.yaml"
    if bad_content is not None:
        config_path.write_text(bad_content, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-m", "src.cli", "validate-config", "--data", str(config_path)],
        cwd=Path(__file__).resolve().parents[2],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2, result.stderr
    assert "Configuration error" in result.stderr
    assert str(config_path) in result.stderr
    assert "Traceback" not in result.stderr
