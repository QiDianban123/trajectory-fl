"""Unit tests for S1-G run context and manifest invariants."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src.experiments import RunContext, file_checksum

CODE_SHA = "a" * 40


def _context(tmp_path: Path, *, config: dict[str, object] | None = None) -> RunContext:
    return RunContext(
        config={} if config is None else config,
        project_root=tmp_path,
        run_id="test-run",
        code_sha=CODE_SHA,
    )


def test_context_snapshots_config_before_caller_mutation(tmp_path: Path) -> None:
    config = {"seed": 42, "nested": {"dataset": "highd"}}
    context = _context(tmp_path, config=config)
    config["nested"]["dataset"] = "changed"  # type: ignore[index]

    snapshot = json.loads(context.config_path.read_text(encoding="utf-8"))
    manifest = context.get_manifest()
    assert snapshot == {"nested": {"dataset": "highd"}, "seed": 42}
    assert manifest["config"] == snapshot
    assert context.git_sha == CODE_SHA


@pytest.mark.parametrize("run_id", ["", "a/b", "a..b", "x" * 81, "bad\n"])
def test_invalid_run_id_is_rejected_without_creating_outputs(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        RunContext({}, tmp_path, run_id=run_id, code_sha=CODE_SHA)
    assert not (tmp_path / "outputs").exists()


def test_duplicate_run_is_rejected(tmp_path: Path) -> None:
    _context(tmp_path)
    with pytest.raises(FileExistsError):
        _context(tmp_path)


def test_context_removes_new_run_directory_when_snapshot_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_write(path: Path, payload: object) -> None:
        raise OSError("injected snapshot failure")

    monkeypatch.setattr(RunContext, "_write_json", staticmethod(fail_write))
    with pytest.raises(OSError, match="injected snapshot failure"):
        RunContext({}, tmp_path, run_id="failed-run", code_sha=CODE_SHA)
    assert not (tmp_path / "outputs" / "failed-run").exists()


@pytest.mark.parametrize("config", [{"callback": lambda: None}, {"value": float("nan")}, []])
def test_non_json_object_config_is_rejected_before_output_creation(
    tmp_path: Path, config: object
) -> None:
    with pytest.raises(ValueError, match="config"):
        RunContext(config, tmp_path, run_id="bad-config", code_sha=CODE_SHA)  # type: ignore[arg-type]
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("code_sha", ["", "abc", "z" * 40, "a" * 39])
def test_invalid_code_sha_is_rejected(tmp_path: Path, code_sha: str) -> None:
    with pytest.raises(ValueError, match="code_sha"):
        RunContext({}, tmp_path, run_id="bad-sha", code_sha=code_sha)


def test_missing_git_command_is_reported_before_output_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _missing(*args: object, **kwargs: object) -> object:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", _missing)
    with pytest.raises(RuntimeError, match="cannot determine Git SHA"):
        RunContext({}, tmp_path, run_id="missing-git")
    assert not (tmp_path / "outputs").exists()


def test_file_checksum_rejects_missing_paths_and_directories(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        file_checksum(tmp_path / "missing.csv")
    with pytest.raises(ValueError, match="regular file"):
        file_checksum(tmp_path)


def test_data_files_are_sorted_and_changes_are_detected(tmp_path: Path) -> None:
    context = _context(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    first = data_dir / "a.csv"
    second = data_dir / "b.csv"
    first.write_text("first", encoding="utf-8")
    second.write_text("second", encoding="utf-8")

    context.record_data_file(second)
    context.record_data_file(first)
    assert list(context.get_manifest()["data_files"]) == ["data/a.csv", "data/b.csv"]

    first.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after it was recorded"):
        context.record_data_file(first)


def test_data_file_paths_and_checksums_are_validated(tmp_path: Path) -> None:
    context = _context(tmp_path)
    outside = tmp_path.parent / "outside.csv"
    outside.write_text("outside", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes project root"):
        context.record_data_file(outside)
    with pytest.raises(ValueError, match="project-relative"):
        context.add_data_file(tmp_path / "absolute.csv", "a" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        context.add_data_file("data/file.csv", "not-a-checksum")


def test_manifest_requires_immutable_data_identity(tmp_path: Path) -> None:
    context = _context(tmp_path)
    with pytest.raises(ValueError, match="data_version and split_id"):
        context.export_manifest()
    context.set_data_identity(data_version="highd-sample-v1", split_id="split-42")
    context.set_data_identity(data_version="highd-sample-v1", split_id="split-42")
    with pytest.raises(ValueError, match="cannot be changed"):
        context.set_data_identity(data_version="highd-sample-v2", split_id="split-42")
    assert context.export_manifest().is_file()


def test_returned_manifest_cannot_mutate_context(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.add_split_manifest("train", {"sample_count": 3})
    returned = context.get_manifest()
    returned["splits"]["train"]["sample_count"] = 999
    assert context.get_manifest()["splits"]["train"]["sample_count"] == 3


def test_profile_and_artifact_paths_are_validated(tmp_path: Path) -> None:
    context = _context(tmp_path)
    profile_path = context.export_data_profile({"samples": 3})
    assert json.loads(profile_path.read_text(encoding="utf-8")) == {"samples": 3}
    assert context.get_manifest()["artifacts"]["data_profile"] == "data_profile.json"
    with pytest.raises(ValueError, match="data profile"):
        context.export_data_profile({"bad": float("nan")})
    with pytest.raises(ValueError, match="escapes"):
        context.add_artifact("outside", "../outside.json")
