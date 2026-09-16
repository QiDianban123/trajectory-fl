"""UI runner must use argument arrays and report subprocess outcomes."""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.ui.capabilities import CommandSpec
from src.ui.command_runner import CommandRunner, UiRunState


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
    assert result.state == "failed"


def test_runner_records_timeout_and_session_state_never_reexecutes(
    monkeypatch, tmp_path: Path
) -> None:
    calls = 0

    def timeout(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"], output="partial", stderr="late")

    monkeypatch.setattr(subprocess, "run", timeout)
    result = CommandRunner(tmp_path).run(CommandSpec("test", ("python", "-m", "src.cli"), "run-1"))
    state = UiRunState("run-1", result.state, result)

    assert result.exit_code == 124 and result.state == "timed_out"
    assert "timed out" in result.stdout
    assert state.result is result and calls == 1
