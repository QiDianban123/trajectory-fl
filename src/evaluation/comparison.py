"""Validated comparison of completed schema-v2 experiment manifests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.evaluation.federated_results import ComparisonIdentity, compare_modes
from src.evaluation.result_store import ResultRecord, result_record_from_dict

THREE_MODES = ("centralized", "local_only", "federated")
IDENTITY_FIELDS = {
    "data_version",
    "split_id",
    "partition_id",
    "scaler_id",
    "model_config_digest",
    "seed",
    "initial_state_id",
    "metric_schema",
}


@dataclass(frozen=True)
class ComparisonOutput:
    """Canonical records and the generated comparison figure."""

    records: tuple[ResultRecord, ...]
    figure_path: Path


def compare_run_manifests(
    run_paths: Sequence[str | Path], output_path: str | Path
) -> ComparisonOutput:
    """Validate and compare exactly one completed run for each training mode."""

    if len(run_paths) != len(THREE_MODES):
        raise ValueError("compare requires exactly three run directories or manifest files")
    output = Path(output_path)
    if output.exists():
        raise ValueError(f"comparison output already exists: {output}")

    loaded = [_load_manifest(value) for value in run_paths]
    records: list[ResultRecord] = []
    identities: list[ComparisonIdentity] = []
    budget_signatures: list[tuple[int, tuple[str, ...]]] = []
    modes: list[str] = []
    for source, manifest in loaded:
        if manifest.get("schema_version") != 2:
            raise ValueError(f"{source}: comparison requires a schema-v2 manifest")
        if manifest.get("status") != "completed":
            raise ValueError(f"{source}: run status must be completed")
        mode = manifest.get("mode")
        if mode not in THREE_MODES:
            raise ValueError(f"{source}: unsupported or missing mode")
        summary = manifest.get("summary")
        if not isinstance(summary, Mapping):
            raise ValueError(f"{source}: completed run requires a summary")
        record = result_record_from_dict(summary)
        if record.mode != mode:
            raise ValueError(f"{source}: manifest mode and summary mode differ")
        identity, budget_signature = _comparison_identity(manifest, source)
        if record.seed != identity.seed or record.split_id != identity.split_id:
            raise ValueError(f"{source}: summary and comparison identity differ")
        records.append(record)
        identities.append(identity)
        budget_signatures.append(budget_signature)
        modes.append(mode)

    if set(modes) != set(THREE_MODES) or len(set(modes)) != len(THREE_MODES):
        raise ValueError("compare requires centralized, local_only, and federated exactly once")
    if len(set(budget_signatures)) != 1:
        raise ValueError("runs do not share sample-visit budget and client coverage")
    if len({record.sample_count for record in records}) != 1:
        raise ValueError("runs do not share the evaluation sample count")

    figure = compare_modes(records, output, identities=identities)
    ordered = tuple(sorted(records, key=lambda item: THREE_MODES.index(item.mode)))
    return ComparisonOutput(ordered, figure)


def _load_manifest(value: str | Path) -> tuple[Path, Mapping[str, object]]:
    source = Path(value)
    if source.is_dir():
        candidates = (source / "s3_manifest.json", source / "manifest.json")
        source = next((candidate for candidate in candidates if candidate.is_file()), candidates[1])
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read comparison manifest {source}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{source}: manifest root must be a mapping")
    return source, payload


def _comparison_identity(
    manifest: Mapping[str, object], source: Path
) -> tuple[ComparisonIdentity, tuple[int, tuple[str, ...]]]:
    raw_identity = manifest.get("identity")
    if not isinstance(raw_identity, Mapping) or set(raw_identity) != IDENTITY_FIELDS:
        raise ValueError(f"{source}: identity must contain the eight comparison fields")
    fairness = manifest.get("fairness")
    if not isinstance(fairness, Mapping) or fairness.get("comparable") is not True:
        raise ValueError(f"{source}: fairness must explicitly be comparable")
    if any(fairness.get(key) != raw_identity[key] for key in IDENTITY_FIELDS):
        raise ValueError(f"{source}: fairness identity differs from manifest identity")
    planned = fairness.get("planned_budget")
    actual = fairness.get("actual_budget")
    if not isinstance(planned, Mapping) or not isinstance(actual, Mapping) or planned != actual:
        raise ValueError(f"{source}: planned and actual budgets must match")
    sample_visits = planned.get("sample_visits")
    selected = planned.get("selected_clients")
    if (
        isinstance(sample_visits, bool)
        or not isinstance(sample_visits, int)
        or sample_visits <= 0
        or not isinstance(selected, list)
        or not selected
        or any(not isinstance(item, str) or not item.strip() for item in selected)
    ):
        raise ValueError(f"{source}: invalid comparison budget")
    client_coverage = tuple(sorted(set(selected)))
    budget_id = f"sample-visits:{sample_visits};clients:{','.join(client_coverage)}"
    try:
        identity = ComparisonIdentity(**raw_identity, budget_id=budget_id)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: invalid comparison identity: {exc}") from exc
    return identity, (sample_visits, client_coverage)
