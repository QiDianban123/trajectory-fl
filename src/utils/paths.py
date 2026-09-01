"""Safe, run-scoped output paths that cannot escape the configured output root."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


class PathSafetyError(ValueError):
    """A requested run or artifact path is unsafe or outside its output root."""


@dataclass(frozen=True)
class RunPaths:
    """The fixed artifact layout for a unique, non-overwriting experiment run."""

    run_dir: Path
    config_path: Path
    metadata_path: Path
    log_path: Path
    metrics_json_path: Path
    metrics_csv_path: Path
    checkpoint_dir: Path
    figures_dir: Path


def create_run_paths(output_root: str | Path, run_id: str) -> RunPaths:
    """Create one new run directory and its artifact subdirectories.

    Existing run IDs are rejected to prevent overwriting a prior result.
    """

    validate_run_id(run_id)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_dir = resolve_within(root, run_id)
    if run_dir.exists():
        raise FileExistsError(f"run output already exists: {run_dir}")
    run_dir.mkdir()
    checkpoint_dir = resolve_within(run_dir, "checkpoints")
    figures_dir = resolve_within(run_dir, "figures")
    checkpoint_dir.mkdir()
    figures_dir.mkdir()
    return RunPaths(
        run_dir=run_dir,
        config_path=resolve_within(run_dir, "config.yaml"),
        metadata_path=resolve_within(run_dir, "metadata.json"),
        log_path=resolve_within(run_dir, "train.log"),
        metrics_json_path=resolve_within(run_dir, "metrics.json"),
        metrics_csv_path=resolve_within(run_dir, "metrics.csv"),
        checkpoint_dir=checkpoint_dir,
        figures_dir=figures_dir,
    )


def resolve_within(root: str | Path, relative_path: str | Path) -> Path:
    """Resolve a relative artifact path and reject directory traversal or absolutes."""

    root_path = Path(root).resolve()
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise PathSafetyError("artifact paths must be relative to the output root")
    resolved = (root_path / candidate).resolve()
    if not resolved.is_relative_to(root_path):
        raise PathSafetyError("artifact path escapes the configured output root")
    return resolved


def validate_run_id(run_id: str) -> None:
    """Allow stable IDs while rejecting separators, traversal, and empty names."""

    if not isinstance(run_id, str) or not _RUN_ID_PATTERN.fullmatch(run_id):
        raise PathSafetyError("run_id must contain only letters, numbers, underscores, or hyphens")
