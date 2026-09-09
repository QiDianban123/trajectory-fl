"""Reproducible run context and deterministic data-manifest artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.utils.paths import validate_run_id

MANIFEST_SCHEMA_VERSION = 1
_CHECKSUM_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CODE_SHA_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"


def file_checksum(filepath: str | Path) -> str:
    """Return the SHA-256 digest of one regular file."""

    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"data file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"data path is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RunContext:
    """Create a unique run directory and collect reproducibility metadata."""

    def __init__(
        self,
        config: Mapping[str, Any],
        project_root: str | Path,
        run_id: str | None = None,
        *,
        code_sha: str | None = None,
        output_root: str | Path = "outputs",
    ) -> None:
        self.project_root = Path(project_root).resolve()
        if not self.project_root.is_dir():
            raise ValueError(f"project_root is not a directory: {self.project_root}")

        self.run_id = run_id if run_id is not None else self._generate_run_id()
        validate_run_id(self.run_id)
        self.code_sha = (
            self._validate_code_sha(code_sha) if code_sha is not None else self._get_git_sha()
        )
        self.git_sha = self.code_sha
        self._config = self._json_clone(config, "config", require_mapping=True)

        output_root_path = Path(output_root)
        if output_root_path.is_absolute():
            resolved_output_root = output_root_path.resolve()
        else:
            resolved_output_root = (self.project_root / output_root_path).resolve()
        if not resolved_output_root.is_relative_to(self.project_root):
            raise ValueError(f"output_root escapes project root: {output_root}")
        self.output_dir = (resolved_output_root / self.run_id).resolve()
        if not self.output_dir.is_relative_to(resolved_output_root):
            raise ValueError(f"run output escapes output root: {self.output_dir}")
        if self.output_dir.exists():
            raise FileExistsError(f"run output already exists: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=False)

        self.config_path = self.output_dir / "config_snapshot.json"
        try:
            self._write_json(self.config_path, self._config)
        except (OSError, TypeError, ValueError):
            shutil.rmtree(self.output_dir)
            raise
        self._created_at = datetime.now(timezone.utc).isoformat()
        self._data_version: str | None = None
        self._split_id: str | None = None
        self._data_files: dict[str, str] = {}
        self._splits: dict[str, object] = {}
        self._partition: object | None = None
        self._artifacts: dict[str, str] = {"config": self.config_path.name}

    @staticmethod
    def _generate_run_id() -> str:
        timestamp = datetime.now(timezone.utc).strftime(_TIMESTAMP_FORMAT)
        return f"{timestamp}_{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _validate_code_sha(code_sha: str) -> str:
        if not isinstance(code_sha, str) or not _CODE_SHA_PATTERN.fullmatch(code_sha.lower()):
            raise ValueError("code_sha must be a 40- or 64-character hexadecimal Git SHA")
        return code_sha.lower()

    def _get_git_sha(self) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return self._validate_code_sha(result.stdout.strip())
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as exc:
            raise RuntimeError(f"cannot determine Git SHA from {self.project_root}: {exc}") from exc
        except ValueError as exc:
            raise RuntimeError(f"Git returned an invalid SHA for {self.project_root}") from exc

    def set_data_identity(self, *, data_version: str, split_id: str) -> None:
        """Set the immutable dataset version and split identifier for this run."""

        identity = (
            self._non_empty_string(data_version, "data_version"),
            self._non_empty_string(split_id, "split_id"),
        )
        previous = (self._data_version, self._split_id)
        if previous != (None, None) and previous != identity:
            raise ValueError("data identity is already set and cannot be changed")
        self._data_version, self._split_id = identity

    def add_data_file(self, relative_path: str | Path, checksum: str) -> None:
        """Record a project-relative data file and validated SHA-256 digest."""

        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise ValueError("data file manifest paths must be project-relative")
        normalized, _ = self._project_relative(candidate)
        digest = checksum.lower() if isinstance(checksum, str) else ""
        if not _CHECKSUM_PATTERN.fullmatch(digest):
            raise ValueError("data file checksum must be a 64-character SHA-256 digest")
        existing = self._data_files.get(normalized)
        if existing is not None and existing != digest:
            raise ValueError(f"data file changed after it was recorded: {normalized}")
        self._data_files[normalized] = digest

    def record_data_file(self, filepath: str | Path) -> None:
        """Hash and record one regular file located inside the project root."""

        normalized, resolved = self._project_relative(filepath)
        self.add_data_file(normalized, file_checksum(resolved))

    def add_split_manifest(self, split_name: str, manifest: object) -> None:
        """Record one JSON-serializable split payload under a stable name."""

        name = self._non_empty_string(split_name, "split_name")
        payload = self._json_clone(manifest, f"split manifest {name!r}")
        existing = self._splits.get(name)
        if existing is not None and existing != payload:
            raise ValueError(f"split manifest {name!r} is already recorded")
        self._splits[name] = payload

    def add_partition_manifest(self, manifest: object) -> None:
        """Record D's partition manifest or an equivalent JSON mapping."""

        payload = manifest.to_mapping() if hasattr(manifest, "to_mapping") else manifest
        normalized = self._json_clone(payload, "partition manifest", require_mapping=True)
        if self._partition is not None and self._partition != normalized:
            raise ValueError("partition manifest is already recorded")
        self._partition = normalized

    def add_artifact(self, name: str, relative_path: str | Path) -> None:
        """Record a path relative to this run directory and reject traversal."""

        artifact_name = self._non_empty_string(name, "artifact name")
        candidate = Path(relative_path)
        if candidate.is_absolute():
            raise ValueError("artifact paths must be relative to the run directory")
        resolved = (self.output_dir / candidate).resolve()
        if not resolved.is_relative_to(self.output_dir):
            raise ValueError(f"artifact path escapes the run directory: {relative_path}")
        normalized = resolved.relative_to(self.output_dir).as_posix()
        existing = self._artifacts.get(artifact_name)
        if existing is not None and existing != normalized:
            raise ValueError(f"artifact {artifact_name!r} is already recorded")
        self._artifacts[artifact_name] = normalized

    def export_data_profile(self, profile: Mapping[str, Any]) -> Path:
        """Write a deterministic data-profile JSON artifact."""

        payload = self._json_clone(profile, "data profile", require_mapping=True)
        profile_path = self.output_dir / "data_profile.json"
        self._write_json(profile_path, payload)
        self.add_artifact("data_profile", profile_path.name)
        return profile_path

    def export_manifest(self) -> Path:
        """Validate and atomically write the complete run manifest."""

        if self._data_version is None or self._split_id is None:
            raise ValueError("data_version and split_id must be set before manifest export")
        manifest_path = self.output_dir / "manifest.json"
        self.add_artifact("manifest", manifest_path.name)
        self._write_json(manifest_path, self._manifest_payload())
        return manifest_path

    def get_manifest(self) -> dict[str, Any]:
        """Return a deep copy so callers cannot mutate internal manifest state."""

        return self._json_clone(self._manifest_payload(), "manifest", require_mapping=True)

    def _manifest_payload(self) -> dict[str, Any]:
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "run_id": self.run_id,
            "code_sha": self.code_sha,
            "git_sha": self.code_sha,
            "created_at": self._created_at,
            "output_dir": self.output_dir.relative_to(self.project_root).as_posix(),
            "config": self._config,
            "data_version": self._data_version,
            "split_id": self._split_id,
            "data_files": dict(sorted(self._data_files.items())),
            "splits": {key: self._splits[key] for key in sorted(self._splits)},
            "partition": self._partition,
            "artifacts": dict(sorted(self._artifacts.items())),
        }

    def _project_relative(self, filepath: str | Path) -> tuple[str, Path]:
        candidate = Path(filepath)
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (self.project_root / candidate).resolve()
        )
        if not resolved.is_relative_to(self.project_root):
            raise ValueError(f"data file escapes project root: {filepath}")
        return resolved.relative_to(self.project_root).as_posix(), resolved

    @staticmethod
    def _non_empty_string(value: object, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    @staticmethod
    def _json_clone(value: object, name: str, *, require_mapping: bool = False) -> Any:
        if require_mapping and not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping")
        try:
            serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} is not JSON serializable: {exc}") from exc
        return json.loads(serialized)

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        try:
            temporary_path.write_text(f"{text}\n", encoding="utf-8")
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
