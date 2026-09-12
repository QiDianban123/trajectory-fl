"""S2 capability registry and validated command construction."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from src.utils.paths import resolve_within, validate_run_id


class UiCommandError(ValueError):
    """A UI command violates the phase, path, or run safety boundary."""


@dataclass(frozen=True)
class CommandSpec:
    """One reviewable argument-array command permitted by the S2 console."""

    action: str
    argv: tuple[str, ...]
    run_id: str | None = None
    output_root: Path | None = None

    @property
    def preview(self) -> str:
        return " ".join(self.argv)


@dataclass(frozen=True)
class Capability:
    """A UI action that is either available or explicitly disabled for S2."""

    key: str
    label: str
    enabled: bool
    reason: str


def s2_capabilities() -> tuple[Capability, ...]:
    """Expose only MS3-approved centralized capabilities."""

    return (
        Capability("centralized_smoke", "集中式 smoke", True, "使用匿名小样例和生产 CLI"),
        Capability("centralized_train", "集中式训练", True, "使用已验证 processed 数据"),
        Capability("local_only", "Local-only", False, "S3 功能尚未验收"),
        Capability("federated", "Federated", False, "S3 功能尚未验收"),
        Capability("compare", "三模式比较", False, "S3/S4 功能尚未验收"),
    )


def build_smoke_command(project_root: str | Path, workspace: str) -> CommandSpec:
    """Build the sole anonymous-data smoke command allowed by the S2 page."""

    root = Path(project_root).resolve()
    workspace_path = _safe_relative(root, workspace, "smoke workspace")
    if workspace_path.relative_to(root).parts[0] != "outputs":
        raise UiCommandError("smoke workspace must be under outputs")
    if workspace_path.exists():
        raise UiCommandError("smoke workspace already exists and will not be overwritten")
    return CommandSpec(
        action="centralized_smoke",
        argv=(
            sys.executable,
            "scripts/run_centralized_smoke.py",
            "--workspace",
            workspace_path.relative_to(root).as_posix(),
        ),
        output_root=workspace_path,
    )


def build_centralized_train_command(
    project_root: str | Path,
    *,
    data_config: str,
    model_config: str,
    experiment_config: str,
    processed_dir: str,
    output_root: str,
    run_id: str,
    seed: int,
    epochs: int,
    batch_size: int,
) -> CommandSpec:
    """Build a validated production centralized-train argument array."""

    root = Path(project_root).resolve()
    try:
        validate_run_id(run_id)
    except ValueError as exc:
        raise UiCommandError(str(exc)) from exc
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (epochs, batch_size)
    ):
        raise UiCommandError("epochs and batch_size must be positive integers")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise UiCommandError("seed must be a non-negative integer")
    data_path = _safe_config(root, data_config)
    model_path = _safe_config(root, model_config)
    experiment_path = _safe_config(root, experiment_config)
    processed_path = _safe_relative(root, processed_dir, "processed directory")
    if not processed_path.is_dir():
        raise UiCommandError("processed directory does not exist")
    output_path = _safe_relative(root, output_root, "output root")
    if output_path.relative_to(root).parts[0] != "outputs":
        raise UiCommandError("output root must be under outputs")
    if (output_path / run_id).exists():
        raise UiCommandError("run output already exists and will not be overwritten")
    return CommandSpec(
        action="centralized_train",
        argv=(
            sys.executable,
            "-m",
            "src.cli",
            "train",
            "--mode",
            "centralized",
            "--data",
            data_path.relative_to(root).as_posix(),
            "--model",
            model_path.relative_to(root).as_posix(),
            "--experiment",
            experiment_path.relative_to(root).as_posix(),
            "--processed-dir",
            processed_path.relative_to(root).as_posix(),
            "--output-root",
            output_path.relative_to(root).as_posix(),
            "--run-id",
            run_id,
            "--seed",
            str(seed),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
        ),
        run_id=run_id,
        output_root=output_path,
    )


def _safe_config(root: Path, value: str) -> Path:
    path = _safe_relative(root, value, "config path")
    if not path.is_file() or not path.is_relative_to(root / "configs"):
        raise UiCommandError("config path must be an existing file under configs")
    return path


def _safe_relative(root: Path, value: str, label: str) -> Path:
    try:
        return resolve_within(root, value)
    except ValueError as exc:
        raise UiCommandError(f"{label} is unsafe: {exc}") from exc
