"""Canonical result record and JSON/CSV serialization contracts."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Literal, Protocol

from src.evaluation.metrics import PHYSICAL_COORDINATE_UNIT, SUPPORTED_MODES

ResultMode = Literal["centralized", "local_only", "federated"]
CSV_FIELDS = (
    "schema_version",
    "run_id",
    "status",
    "code_sha",
    "seed",
    "split_id",
    "mode",
    "dataset",
    "model",
    "sample_count",
    "coordinate_unit",
    "ade",
    "fde",
    "total_seconds",
    "artifact_paths",
)


@dataclass(frozen=True)
class ResultRecord:
    """One completed evaluation record shared by all three training modes."""

    run_id: str
    code_sha: str
    seed: int
    split_id: str
    mode: ResultMode
    sample_count: int
    ade: float
    fde: float
    total_seconds: float
    artifact_paths: Mapping[str, str] = field(default_factory=dict)
    status: str = "completed"
    dataset: str = "highd"
    model: str = "lstm_encoder_decoder"
    schema_version: int = 1
    coordinate_unit: str = PHYSICAL_COORDINATE_UNIT

    def __post_init__(self) -> None:
        for name in ("run_id", "code_sha", "split_id", "dataset", "model"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string")
        if self.schema_version != 1:
            raise ValueError("schema_version must be 1")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.mode not in SUPPORTED_MODES:
            raise ValueError(f"mode must be one of {SUPPORTED_MODES}")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count <= 0
        ):
            raise ValueError("sample_count must be a positive integer")
        for name in ("ade", "fde", "total_seconds"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError(f"{name} must be finite")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.coordinate_unit != PHYSICAL_COORDINATE_UNIT:
            raise ValueError("metrics must use physical coordinates in meters")
        if self.status not in ("completed", "failed"):
            raise ValueError("status must be completed or failed")
        if not isinstance(self.artifact_paths, Mapping):
            raise ValueError("artifact_paths must be a mapping")
        if any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            or not value.strip()
            for key, value in self.artifact_paths.items()
        ):
            raise ValueError("artifact_paths must map non-empty names to non-empty paths")

    @property
    def timing_seconds(self) -> dict[str, float]:
        return {"total": float(self.total_seconds)}

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-compatible representation."""

        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "status": self.status,
            "code_sha": self.code_sha,
            "seed": self.seed,
            "split_id": self.split_id,
            "mode": self.mode,
            "dataset": self.dataset,
            "model": self.model,
            "sample_count": self.sample_count,
            "coordinate_unit": self.coordinate_unit,
            "metrics": {"ade": float(self.ade), "fde": float(self.fde)},
            "timing_seconds": self.timing_seconds,
            "artifacts": dict(self.artifact_paths),
        }


def write_json(record: ResultRecord, path: str | Path) -> Path:
    """Write one canonical result record to JSON."""

    destination = _prepare_path(path)
    destination.write_text(
        json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def write_csv(records: Iterable[ResultRecord], path: str | Path) -> Path:
    """Write a flat CSV view derived from the same ResultRecord objects."""

    values = list(records)
    if not values:
        raise ValueError("at least one result record is required")
    destination = _prepare_path(path)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in values:
            writer.writerow(
                {
                    "schema_version": record.schema_version,
                    "run_id": record.run_id,
                    "status": record.status,
                    "code_sha": record.code_sha,
                    "seed": record.seed,
                    "split_id": record.split_id,
                    "mode": record.mode,
                    "dataset": record.dataset,
                    "model": record.model,
                    "sample_count": record.sample_count,
                    "coordinate_unit": record.coordinate_unit,
                    "ade": record.ade,
                    "fde": record.fde,
                    "total_seconds": record.total_seconds,
                    "artifact_paths": json.dumps(
                        dict(record.artifact_paths), ensure_ascii=False, sort_keys=True
                    ),
                }
            )
    return destination


class ResultStore(Protocol):
    """Minimal storage boundary for future experiment runners."""

    def save_json(self, record: ResultRecord, path: str | Path) -> Path:
        ...

    def save_csv(self, records: Iterable[ResultRecord], path: str | Path) -> Path:
        ...


def _prepare_path(path: str | Path) -> Path:
    destination = Path(path)
    if not destination.name or not destination.suffix:
        raise ValueError("result path must include a filename with an extension")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination
