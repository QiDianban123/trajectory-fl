"""Read-only discovery of saved centralized runs and artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

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
    for manifest_path in sorted(outputs.rglob("manifest.json"), key=lambda path: path.as_posix()):
        summary = _read_run(manifest_path)
        if summary is not None:
            summaries.append(summary)
    return sorted(summaries, key=lambda item: item.run_id, reverse=True)


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
    facts = metrics if isinstance(metrics, dict) else summary
    metrics_values = facts.get("metrics", {}) if isinstance(facts, dict) else {}
    timing = facts.get("timing_seconds", {}) if isinstance(facts, dict) else {}
    return RunSummary(
        run_dir=run_dir,
        run_id=run_id,
        status=facts.get("status", manifest.get("status", "unknown"))
        if isinstance(facts, dict)
        else "unknown",
        mode=facts.get("mode", "unknown") if isinstance(facts, dict) else "unknown",
        seed=facts.get("seed", 0) if isinstance(facts, dict) else 0,
        split_id=split_id,
        data_version=(
            manifest.get("data_version")
            if isinstance(manifest.get("data_version"), str)
            else identity.get("data_version")
            if isinstance(identity, dict) and isinstance(identity.get("data_version"), str)
            else None
        ),
        best_epoch=history.get("best_epoch") if isinstance(history, dict) else None,
        sample_count=facts.get("sample_count") if isinstance(facts, dict) else None,
        ade=metrics_values.get("ade") if isinstance(metrics_values, dict) else None,
        fde=metrics_values.get("fde") if isinstance(metrics_values, dict) else None,
        total_seconds=timing.get("total") if isinstance(timing, dict) else None,
        artifacts=artifact_paths,
        fairness=manifest.get("fairness") if isinstance(manifest.get("fairness"), dict) else None,
        clients=tuple(item for item in manifest.get("clients", []) if isinstance(item, dict)),
        rounds=tuple(item for item in manifest.get("rounds", []) if isinstance(item, dict)),
    )


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None
