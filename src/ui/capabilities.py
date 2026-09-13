"""S2 capability registry and validated command construction."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

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
    mode: Literal["centralized", "local_only", "federated"] | None = None
    resume: bool = False

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


class CapabilityRegistry:
    """S3 UI allow-list; every visible action has a fixed production entry point."""

    def capabilities(self) -> tuple[Capability, ...]:
        return (
            Capability("centralized_smoke", "集中式 smoke", True, "匿名样例与生产 CLI"),
            Capability("centralized_train", "Centralized", True, "生产三模式入口"),
            Capability("local_only_train", "Local-only", True, "生产三模式入口"),
            Capability("federated_train", "Federated", True, "生产三模式入口"),
            Capability("three_mode_smoke", "三模式 smoke", True, "共享数据、初态与公平性 guard"),
            Capability("resume", "恢复失败运行", True, "仅正式 recovery.json 边界"),
            Capability("compare", "三模式比较", False, "S3 UI-2 只读比较视图尚未交付"),
        )


def s2_capabilities() -> tuple[Capability, ...]:
    """Expose only MS3-approved centralized capabilities."""

    return (
        Capability("centralized_smoke", "集中式 smoke", True, "使用匿名小样例和生产 CLI"),
        Capability("centralized_train", "集中式训练", True, "使用已验证 processed 数据"),
        Capability("local_only", "Local-only", False, "S3 功能尚未验收"),
        Capability("federated", "Federated", False, "S3 功能尚未验收"),
        Capability("compare", "三模式比较", False, "S3/S4 功能尚未验收"),
    )


def s3_capabilities() -> tuple[Capability, ...]:
    """Compatibility-friendly S3 registry accessor for the Streamlit page."""

    return CapabilityRegistry().capabilities()


@dataclass(frozen=True)
class FairnessPreflight:
    allowed: bool
    reason: str
    fields: dict[str, object]


def preflight_s3_fairness(
    project_root: str | Path, *, mode: str, experiment_config: str
) -> FairnessPreflight:
    """Read only frozen config facts; production runner repeats identity/budget validation."""

    root = Path(project_root).resolve()
    if mode not in ("centralized", "local_only", "federated"):
        return FairnessPreflight(False, "未知训练模式", {})
    try:
        experiment = yaml.safe_load(
            _safe_config(root, experiment_config).read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, yaml.YAMLError, UiCommandError) as exc:
        return FairnessPreflight(False, f"无法读取实验配置：{exc}", {})
    if not isinstance(experiment, dict):
        return FairnessPreflight(False, "实验配置必须是对象", {})
    run = experiment.get("run")
    three = experiment.get("three_mode")
    if not isinstance(run, dict) or run.get("mode") != mode:
        return FairnessPreflight(False, "命令模式与实验配置不一致", {})
    if not isinstance(three, dict):
        return FairnessPreflight(False, "缺少 three_mode 公平性配置", {})
    matrix = three.get("mode_matrix")
    required = {"rounds", "local_epochs", "clients_per_round", "metric_schema", "recovery"}
    if matrix != ["centralized", "local_only", "federated"] or not required <= set(three):
        return FairnessPreflight(False, "三模式公平性配置不完整", {})
    if three["metric_schema"] != "ade_fde_meter_v1" or three["clients_per_round"] != 5:
        return FairnessPreflight(False, "指标 schema 或 5-RSU 预算不符合冻结契约", {})
    return FairnessPreflight(
        True,
        "配置预检通过；生产 runner 将再次校验数据、初态与实际预算",
        {
            "mode": mode,
            "seed": run.get("seed"),
            "rounds": three["rounds"],
            "local_epochs": three["local_epochs"],
            "clients_per_round": three["clients_per_round"],
            "metric_schema": three["metric_schema"],
        },
    )


def build_s3_train_command(
    project_root: str | Path,
    *,
    mode: Literal["centralized", "local_only", "federated"],
    processed_dir: str,
    run_id: str,
    resume: bool = False,
) -> CommandSpec:
    """Build one S3 allow-listed train command after read-only fairness preflight."""

    root = Path(project_root).resolve()
    experiment = f"configs/experiments/s3_{mode}_smoke.yaml"
    preflight = preflight_s3_fairness(root, mode=mode, experiment_config=experiment)
    if not preflight.allowed:
        raise UiCommandError(preflight.reason)
    validate_run_id(run_id)
    processed = _safe_relative(root, processed_dir, "processed directory")
    if not processed.is_dir() or not any(
        processed.is_relative_to(candidate)
        for candidate in (root / "data" / "processed", root / "outputs")
    ):
        raise UiCommandError("processed directory must be an existing allowed processed split")
    output = resolve_within(root, "outputs")
    run_dir = resolve_within(output, run_id)
    if resume:
        recovery = resolve_within(run_dir, "checkpoints/recovery.json")
        if not recovery.is_file():
            raise UiCommandError("only an existing run recovery.json can be resumed")
    elif run_dir.exists():
        raise UiCommandError("run output already exists and will not be overwritten")
    argv = [
        sys.executable,
        "-m",
        "src.cli",
        "train",
        "--mode",
        mode,
        "--experiment",
        experiment,
        "--processed-dir",
        processed.relative_to(root).as_posix(),
        "--output-root",
        "outputs",
        "--run-id",
        run_id,
    ]
    if resume:
        argv.extend(("--resume-checkpoint", "checkpoints/recovery.json"))
    return CommandSpec(
        f"{mode}_{'resume' if resume else 'train'}", tuple(argv), run_id, output, mode, resume
    )


def build_three_mode_smoke_command(project_root: str | Path) -> CommandSpec:
    root = Path(project_root).resolve()
    if not (root / "scripts/run_three_mode_smoke.py").is_file():
        raise UiCommandError("three-mode smoke entry is unavailable")
    return CommandSpec("three_mode_smoke", (sys.executable, "scripts/run_three_mode_smoke.py"))


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
