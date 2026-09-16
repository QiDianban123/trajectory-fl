"""Formal experiment Git provenance must identify an exact clean source tree."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.experiments.mode_runner import _git_sha, _resolve_code_sha
from src.experiments.three_mode import ExperimentInputError

COMMIT_SHA = "a" * 40


def _completed(returncode: int = 0, stdout: str = "") -> SimpleNamespace:
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


def test_clean_formal_worktree_returns_exact_head(monkeypatch, tmp_path: Path) -> None:
    responses = iter((_completed(stdout=f"{COMMIT_SHA}\n"), _completed(stdout="")))
    monkeypatch.setattr(
        "src.experiments.mode_runner.subprocess.run", lambda *args, **kwargs: next(responses)
    )

    assert _git_sha(tmp_path, require_clean=True) == COMMIT_SHA


def test_dirty_formal_worktree_is_rejected_with_changed_paths(monkeypatch, tmp_path: Path) -> None:
    responses = iter(
        (
            _completed(stdout=f"{COMMIT_SHA}\n"),
            _completed(stdout=" M src/data/adapters.py\n?? configs/local.yaml\n"),
        )
    )
    monkeypatch.setattr(
        "src.experiments.mode_runner.subprocess.run", lambda *args, **kwargs: next(responses)
    )

    with pytest.raises(ExperimentInputError, match="clean Git worktree") as error:
        _git_sha(tmp_path, require_clean=True)
    assert "src/data/adapters.py" in str(error.value)
    assert "configs/local.yaml" in str(error.value)


def test_formal_supplied_sha_must_match_clean_head(monkeypatch, tmp_path: Path) -> None:
    responses = iter((_completed(stdout=f"{COMMIT_SHA}\n"), _completed(stdout="")))
    monkeypatch.setattr(
        "src.experiments.mode_runner.subprocess.run", lambda *args, **kwargs: next(responses)
    )
    bundle = {"experiment": {"provenance": {"require_clean_git": True}}}

    with pytest.raises(ExperimentInputError, match="does not match"):
        _resolve_code_sha(bundle, tmp_path, "b" * 40)


def test_smoke_run_preserves_legacy_unknown_sha_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "src.experiments.mode_runner.subprocess.run",
        lambda *args, **kwargs: _completed(returncode=1),
    )

    assert _git_sha(tmp_path) == "unknown"
