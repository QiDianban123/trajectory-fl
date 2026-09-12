"""UI runner must use argument arrays and report subprocess outcomes."""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.ui.capabilities import CommandSpec
from src.ui.command_runner import CommandRunner


def test_runner_uses_shell_false_and_captures_nonzero(monkeypatch, tmp_path: Path) -> None:
    received: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        received.update(kwargs)
        received["args"] = args[0]
        return subprocess.CompletedProcess(args[0], 2, stdout="out", stderr="err")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = CommandRunner(tmp_path).run(CommandSpec("test", ("python", "-m", "src.cli")))

    assert received["args"] == ["python", "-m", "src.cli"]
    assert received["shell"] is False
    assert result.exit_code == 2
    assert result.stdout == "outerr"
