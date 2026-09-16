"""Integrity verification for a retained final experiment archive."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


class ArchiveIntegrityError(ValueError):
    """A final archive manifest or one of its retained files is invalid."""


def verify_final_archive(manifest_path: str | Path) -> int:
    """Verify final-manifest structure, safe paths, byte sizes, hashes, and result table."""

    manifest_file = Path(manifest_path).resolve()
    try:
        payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ArchiveIntegrityError(f"cannot read final manifest: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("manifest_type") != "final_release_archive":
        raise ArchiveIntegrityError("unsupported final archive manifest")
    integrity = payload.get("integrity")
    files = integrity.get("files") if isinstance(integrity, Mapping) else None
    if not isinstance(files, Mapping) or not files:
        raise ArchiveIntegrityError("final manifest must list retained files")

    archive_root = manifest_file.parent
    for relative, metadata in files.items():
        if not isinstance(relative, str) or not isinstance(metadata, Mapping):
            raise ArchiveIntegrityError("retained file entries must be path-to-metadata mappings")
        path = _resolve_retained_path(archive_root, relative)
        if not path.is_file():
            raise ArchiveIntegrityError(f"retained file is missing: {relative}")
        expected_size = metadata.get("size_bytes")
        if isinstance(expected_size, bool) or not isinstance(expected_size, int):
            raise ArchiveIntegrityError(f"invalid retained file size: {relative}")
        if path.stat().st_size != expected_size:
            raise ArchiveIntegrityError(f"retained file size differs: {relative}")
        expected_hash = metadata.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ArchiveIntegrityError(f"invalid retained file digest: {relative}")
        if _sha256(path) != expected_hash:
            raise ArchiveIntegrityError(f"retained file digest differs: {relative}")

    results = payload.get("results")
    if not isinstance(results, list) or {
        item.get("mode") for item in results if isinstance(item, Mapping)
    } != {"centralized", "local_only", "federated"}:
        raise ArchiveIntegrityError("final manifest must contain exactly the three result modes")
    if len(results) != 3:
        raise ArchiveIntegrityError("final manifest must contain exactly three result records")
    for item in results:
        if not isinstance(item, Mapping):
            raise ArchiveIntegrityError("final result entries must be mappings")
        referenced = [item.get("result_path"), item.get("historical_manifest_path")]
        models = item.get("model_paths")
        if not isinstance(models, list) or not models:
            raise ArchiveIntegrityError("each final result must reference at least one model")
        referenced.extend(models)
        if any(not isinstance(path, str) or path not in files for path in referenced):
            raise ArchiveIntegrityError("final result references an unretained file")
    _verify_comparison_table(archive_root / "comparison.csv", results, payload)
    return len(files)


def _resolve_retained_path(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute():
        raise ArchiveIntegrityError(f"retained file path must be relative: {relative}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ArchiveIntegrityError(f"retained file path escapes archive: {relative}") from exc
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_comparison_table(
    path: Path, results: list[object], payload: Mapping[str, object]
) -> None:
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            rows = {row["mode"]: row for row in csv.DictReader(stream)}
    except (OSError, UnicodeError, KeyError) as exc:
        raise ArchiveIntegrityError(f"cannot read comparison table: {exc}") from exc
    identity = payload.get("comparison_identity")
    if not isinstance(identity, Mapping):
        raise ArchiveIntegrityError("comparison identity is missing")
    for result in results:
        if not isinstance(result, Mapping):
            raise ArchiveIntegrityError("final result entries must be mappings")
        mode = result["mode"]
        row = rows.get(mode)
        if row is None:
            raise ArchiveIntegrityError(f"comparison table is missing mode: {mode}")
        expected = {
            "ade_m": result.get("ade_meter"),
            "fde_m": result.get("fde_meter"),
            "runtime_seconds": result.get("total_seconds"),
            "test_samples": identity.get("evaluation_sample_count"),
            "sample_visits": identity.get("training_sample_visits"),
        }
        if any(str(value) != row.get(key) for key, value in expected.items()):
            raise ArchiveIntegrityError(f"comparison table differs from final result: {mode}")
        if row.get("status") != "completed":
            raise ArchiveIntegrityError(f"comparison table mode is not completed: {mode}")
