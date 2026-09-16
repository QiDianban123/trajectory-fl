"""No-shell subprocess runner for UI command specifications."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Literal

from src.ui.capabilities import CommandSpec, UiCommandError


@dataclass(frozen=True)
class CommandResult:
    """Captured execution state for one allowed UI command."""

    spec: CommandSpec
    stdout: str
    exit_code: int
    started_at: str
    finished_at: str
    state: Literal["succeeded", "failed", "cancelled", "timed_out"] = "succeeded"

    @property
    def succeeded(self) -> bool:
        return self.state == "succeeded" and self.exit_code == 0


@dataclass(frozen=True)
class UiRunState:
    """Session-persisted UI lifecycle; refresh reads this state and never reruns a command."""

    run_id: str | None
    status: Literal["idle", "running", "succeeded", "failed", "cancelled", "timed_out"]
    result: CommandResult | None = None


class CommandRunner:
    """Serialize UI runs and execute only prebuilt argument arrays."""

    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()
        self._active_run_ids: set[str] = set()
        self._lock = Lock()

    def run(self, spec: CommandSpec) -> CommandResult:
        """Execute without a shell and return combined stdout/stderr."""

        self._claim(spec)
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            completed = subprocess.run(
                list(spec.argv),
                cwd=self.project_root,
                shell=False,
                check=False,
                capture_output=True,
                text=True,
                timeout=180,
            )
            output = completed.stdout + completed.stderr
            return CommandResult(
                spec=spec,
                stdout=output,
                exit_code=completed.returncode,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc).isoformat(),
                state="succeeded" if completed.returncode == 0 else "failed",
            )
        except subprocess.TimeoutExpired as exc:
            output = (exc.stdout or "") + (exc.stderr or "")
            return CommandResult(
                spec=spec,
                stdout=output + "\nUI command timed out after 180 seconds.",
                exit_code=124,
                started_at=started_at,
                finished_at=datetime.now(timezone.utc).isoformat(),
                state="timed_out",
            )
        finally:
            self._release(spec)

    def _claim(self, spec: CommandSpec) -> None:
        if spec.run_id is None:
            return
        with self._lock:
            if spec.run_id in self._active_run_ids:
                raise UiCommandError("the same run_id is already running")
            self._active_run_ids.add(spec.run_id)

    def _release(self, spec: CommandSpec) -> None:
        if spec.run_id is None:
            return
        with self._lock:
            self._active_run_ids.discard(spec.run_id)
