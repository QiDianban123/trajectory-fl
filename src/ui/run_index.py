"""Read-only discovery of saved runs, final archives, and comparable results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.archive import ArchiveIntegrityError, verify_final_archive
from src.utils.paths import resolve_within


@dataclass(frozen=True)
class RunSummary:
    """Manifest and ResultRecord facts displayed by the UI without recomputation."""

    run_dir: Path
    run_id: str
    status: str
    mode: str
    seed: int
    split_id: str
    data_version: str | None
    best_epoch: int | None
    sample_count: int | None
    ade: float | None
    fde: float | None
    total_seconds: float | None
    artifacts: dict[str, Path]
    fairness: dict[str, object] | None = None
    clients: tuple[dict[str, object], ...] = ()
    rounds: tuple[dict[str, object], ...] = ()
    identity: dict[str, object] | None = None
    code_sha: str | None = None
    source: str = "outputs"
    is_final: bool = False
    provenance_verified: bool | None = None


@dataclass(frozen=True)
class FinalArchiveSummary:
    """The authoritative retained result set displayed ahead of transient runs."""

    archive_dir: Path
    manifest_path: Path
    status: str
    retained_file_count: int
    provenance_verified: bool
    runs: tuple[RunSummary, ...]
    comparison_table: Path | None
    comparison_figure: Path | None


class ArtifactResolver:
    """Resolve only manifest-relative artifacts that remain inside the run root."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir).resolve()

    def resolve(self, relative_path: object) -> Path | None:
        if not isinstance(relative_path, str):
            return None
        try:
            return resolve_within(self.run_dir, relative_path)
        except ValueError:
            return None


def discover_runs(project_root: str | Path, output_root: str = "outputs") -> list[RunSummary]:
    """Read manifests under the safe output root; never start training or mutate files."""

    root = Path(project_root).resolve()
    outputs = resolve_within(root, output_root)
    if not outputs.is_dir():
        return []
    summaries: list[RunSummary] = []
    s3_manifests = sorted(outputs.rglob("s3_manifest.json"), key=lambda path: path.as_posix())
    preferred_directories = {path.parent.resolve() for path in s3_manifests}
    manifests = s3_manifests + [
        path
        for path in sorted(outputs.rglob("manifest.json"), key=lambda path: path.as_posix())
        if path.parent.resolve() not in preferred_directories
    ]
    for manifest_path in manifests:
        summary = _read_run(manifest_path)
        if summary is not None:
            summaries.append(summary)
    return sorted(summaries, key=lambda item: item.run_id, reverse=True)


def discover_final_archive(
    project_root: str | Path,
    archive_path: str = "artifacts/three_mode_training_20260916/FINAL_MANIFEST.json",
) -> FinalArchiveSummary | None:
    """Read the authoritative final archive without trusting historical artifact paths."""

    root = Path(project_root).resolve()
    try:
        manifest_path = resolve_within(root, archive_path)
    except ValueError:
        return None
    try:
        retained_file_count = verify_final_archive(manifest_path)
    except ArchiveIntegrityError:
        return None
    manifest = _read_json(manifest_path)
    if manifest is None or manifest.get("manifest_type") != "final_release_archive":
        return None
    archive_dir = manifest_path.parent.resolve()
    integrity = manifest.get("integrity")
    retained = integrity.get("files") if isinstance(integrity, dict) else None
    identity = manifest.get("comparison_identity")
    results = manifest.get("results")
    provenance = manifest.get("provenance")
    if (
        not isinstance(retained, dict)
        or not isinstance(identity, dict)
        or not isinstance(results, list)
    ):
        return None
    provenance_verified = (
        provenance.get("verified") is True if isinstance(provenance, dict) else False
    )
    code_sha = _archive_code_sha(archive_dir, provenance, retained)
    runs: list[RunSummary] = []
    for result in results:
        if not isinstance(result, dict):
            return None
        run = _final_run(archive_dir, result, identity, retained, code_sha, provenance_verified)
        if run is None:
            return None
        runs.append(run)
    if {run.mode for run in runs} != {"centralized", "local_only", "federated"}:
        return None
    return FinalArchiveSummary(
        archive_dir=archive_dir,
        manifest_path=manifest_path,
        status=str(manifest.get("status", "unknown")),
        retained_file_count=retained_file_count,
        provenance_verified=provenance_verified,
        runs=tuple(sorted(runs, key=lambda item: _mode_order(item.mode))),
        comparison_table=_retained_path(archive_dir, "comparison.csv", retained),
        comparison_figure=_retained_path(
            archive_dir, "figures/three_mode_comparison.png", retained
        ),
    )


def comparison_error(runs: list[RunSummary] | tuple[RunSummary, ...]) -> str | None:
    """Return why three selected runs cannot be fairly compared, or ``None``."""

    if len(runs) != 3 or {run.mode for run in runs} != {
        "centralized",
        "local_only",
        "federated",
    }:
        return "必须各选择一个 Centralized、Local-only 和 Federated 运行。"
    if any(
        run.status != "completed" or run.ade is None or run.fde is None or run.sample_count is None
        for run in runs
    ):
        return "只有包含完整 ADE/FDE 和样本数的 completed 运行可以比较。"
    if len({run.sample_count for run in runs}) != 1:
        return "三个运行的评价样本数不同。"
    if any(run.identity is None for run in runs):
        return "一个或多个运行缺少比较身份。"
    identity_fields = (
        "data_version",
        "split_id",
        "partition_id",
        "scaler_id",
        "model_config_digest",
        "seed",
        "initial_state_id",
        "metric_schema",
    )
    signatures = {tuple(run.identity.get(key) for key in identity_fields) for run in runs}  # type: ignore[union-attr]
    if len(signatures) != 1 or any(value is None for value in next(iter(signatures))):
        return "三个运行的数据、模型、seed、初始状态或指标口径不同。"
    if any(run.fairness is None or run.fairness.get("comparable") is not True for run in runs):
        return "一个或多个运行未通过公平性预算校验。"
    budgets = [_budget_signature(run.fairness) for run in runs]  # type: ignore[arg-type]
    if any(value is None for value in budgets) or len(set(budgets)) != 1:
        return "三个运行的样本访问预算或客户端覆盖不同。"
    code_shas = {run.code_sha for run in runs}
    if None in code_shas or len(code_shas) != 1:
        return "三个运行的代码版本不同或无法确认。"
    return None


def _read_run(manifest_path: Path) -> RunSummary | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict):
        return None
    run_dir = manifest_path.parent
    run_id = manifest.get("run_id")
    identity = manifest.get("identity")
    split_id = manifest.get("split_id")
    if not isinstance(split_id, str) and isinstance(identity, dict):
        split_id = identity.get("split_id")
    if not isinstance(run_id, str) or not isinstance(split_id, str):
        return None
    metrics = _read_json(run_dir / "metrics.json")
    summary = manifest.get("summary") if isinstance(manifest.get("summary"), dict) else {}
    history = _read_json(run_dir / "training_history.json")
    artifact_paths: dict[str, Path] = {}
    resolver = ArtifactResolver(run_dir)
    artifacts = manifest.get("artifacts", {})
    if isinstance(artifacts, dict):
        for name, relative_path in artifacts.items():
            if isinstance(name, str) and isinstance(relative_path, str):
                resolved = resolver.resolve(relative_path)
                if resolved is not None:
                    artifact_paths[name] = resolved
    candidate_facts = summary if manifest.get("schema_version") == 2 and summary else metrics
    facts: dict[str, object] = candidate_facts if isinstance(candidate_facts, dict) else {}
    metrics_values = facts.get("metrics", {})
    timing = facts.get("timing_seconds", {})
    return RunSummary(
        run_dir=run_dir,
        run_id=run_id,
        status=facts.get("status", manifest.get("status", "unknown")),
        mode=facts.get("mode", "unknown"),
        seed=facts.get("seed", 0),
        split_id=split_id,
        data_version=(
            manifest.get("data_version")
            if isinstance(manifest.get("data_version"), str)
            else identity.get("data_version")
            if isinstance(identity, dict) and isinstance(identity.get("data_version"), str)
            else None
        ),
        best_epoch=history.get("best_epoch") if isinstance(history, dict) else None,
        sample_count=_integer(facts.get("sample_count")),
        ade=_metric_number(metrics_values, "ade", facts),
        fde=_metric_number(metrics_values, "fde", facts),
        total_seconds=_metric_number(timing, "total", facts, fallback="total_seconds"),
        artifacts=artifact_paths,
        fairness=manifest.get("fairness") if isinstance(manifest.get("fairness"), dict) else None,
        clients=tuple(item for item in manifest.get("clients", []) if isinstance(item, dict)),
        rounds=tuple(item for item in manifest.get("rounds", []) if isinstance(item, dict)),
        identity=dict(identity) if isinstance(identity, dict) else None,
        code_sha=facts.get("code_sha") if isinstance(facts.get("code_sha"), str) else None,
    )


def _final_run(
    archive_dir: Path,
    result: dict[str, object],
    identity: dict[str, object],
    retained: dict[str, object],
    code_sha: str | None,
    provenance_verified: bool,
) -> RunSummary | None:
    mode = result.get("mode")
    run_id = result.get("run_id")
    if mode not in ("centralized", "local_only", "federated") or not isinstance(run_id, str):
        return None
    result_path = result.get("result_path")
    historical_path = result.get("historical_manifest_path")
    if not isinstance(result_path, str) or not isinstance(historical_path, str):
        return None
    result_file = _retained_path(archive_dir, result_path, retained)
    historical_file = _retained_path(archive_dir, historical_path, retained)
    if result_file is None or historical_file is None:
        return None
    historical = _read_json(historical_file) or {}
    fairness = historical.get("fairness")
    artifacts: dict[str, Path] = {
        "final_manifest": archive_dir / "FINAL_MANIFEST.json",
        "result": result_file,
        "historical_manifest": historical_file,
    }
    models = result.get("model_paths")
    if not isinstance(models, list) or not models:
        return None
    for index, relative in enumerate(models, start=1):
        if not isinstance(relative, str):
            return None
        model_path = _retained_path(archive_dir, relative, retained)
        if model_path is None:
            return None
        artifacts[f"model_{index:02d}"] = model_path
    for key, relative in (
        ("comparison_table", "comparison.csv"),
        ("comparison_figure", "figures/three_mode_comparison.png"),
    ):
        path = _retained_path(archive_dir, relative, retained)
        if path is not None:
            artifacts[key] = path
    history_path = _retained_path(
        archive_dir, "results/centralized/training_history.json", retained
    )
    history = _read_json(history_path) if history_path is not None else None
    return RunSummary(
        run_dir=archive_dir,
        run_id=run_id,
        status="completed",
        mode=mode,
        seed=_integer(identity.get("seed")) or 0,
        split_id=str(identity.get("split_id", "unknown")),
        data_version=(
            str(identity["data_version"]) if isinstance(identity.get("data_version"), str) else None
        ),
        best_epoch=(
            _integer(history.get("best_epoch"))
            if mode == "centralized" and isinstance(history, dict)
            else None
        ),
        sample_count=_integer(identity.get("evaluation_sample_count")),
        ade=_number(result.get("ade_meter")),
        fde=_number(result.get("fde_meter")),
        total_seconds=_number(result.get("total_seconds")),
        artifacts=artifacts,
        fairness=dict(fairness) if isinstance(fairness, dict) else None,
        clients=tuple(item for item in historical.get("clients", []) if isinstance(item, dict)),
        rounds=tuple(item for item in historical.get("rounds", []) if isinstance(item, dict)),
        identity=dict(identity),
        code_sha=code_sha,
        source="final_archive",
        is_final=True,
        provenance_verified=provenance_verified,
    )


def _archive_code_sha(
    archive_dir: Path, provenance: object, retained: dict[str, object]
) -> str | None:
    if not isinstance(provenance, dict):
        return None
    details = provenance.get("details")
    if not isinstance(details, str):
        return None
    path = _retained_path(archive_dir, details, retained)
    payload = _read_json(path) if path is not None else None
    value = payload.get("recorded_runtime_head") if isinstance(payload, dict) else None
    return value if isinstance(value, str) else None


def _retained_path(archive_dir: Path, relative: str, retained: dict[str, object]) -> Path | None:
    if relative not in retained:
        return None
    resolved = ArtifactResolver(archive_dir).resolve(relative)
    return resolved if resolved is not None and resolved.is_file() else None


def _budget_signature(fairness: dict[str, object]) -> tuple[int, tuple[str, ...]] | None:
    planned = fairness.get("planned_budget")
    actual = fairness.get("actual_budget")
    if not isinstance(planned, dict) or planned != actual:
        return None
    visits = planned.get("sample_visits")
    clients = planned.get("selected_clients")
    if (
        isinstance(visits, bool)
        or not isinstance(visits, int)
        or not isinstance(clients, list)
        or any(not isinstance(item, str) for item in clients)
    ):
        return None
    return visits, tuple(sorted(set(clients)))


def _mode_order(mode: str) -> int:
    return ("centralized", "local_only", "federated").index(mode)


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _integer(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _metric_number(
    nested: object, key: str, facts: dict[str, object], *, fallback: str | None = None
) -> float | None:
    value = nested.get(key) if isinstance(nested, dict) else None
    return _number(value if value is not None else facts.get(fallback or key))
