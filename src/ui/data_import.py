"""Safe CSV import and processed-data summaries for the demonstration UI."""

from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Protocol, Sequence

from src.utils.paths import resolve_within

MAX_UPLOAD_BYTES = 1024 * 1024 * 1024
_COLUMN_ALIASES = {
    "id": {"id", "track id", "vehicle id"},
    "frame": {"frame", "frame id"},
    "x": {"x", "x position"},
    "y": {"y", "y position"},
}


class UploadedCsv(Protocol):
    """The small interface supplied by Streamlit's UploadedFile."""

    name: str

    def getvalue(self) -> bytes: ...


@dataclass(frozen=True)
class ImportedData:
    workspace: Path
    raw_dir: Path
    processed_root: Path
    preparation_root: Path
    file_count: int
    total_bytes: int


@dataclass(frozen=True)
class ProcessedDataSummary:
    processed_dir: Path
    data_version: str
    split_id: str
    input_rows: int
    valid_tracks: int
    rejected_tracks: int
    sample_counts: dict[str, int]
    client_counts: dict[str, int]


def save_uploaded_csvs(
    project_root: str | Path, uploads: Sequence[UploadedCsv], *, import_id: str
) -> ImportedData:
    """Validate and atomically retain user-selected highD-compatible CSV files."""

    root = Path(project_root).resolve()
    if not uploads:
        raise ValueError("请至少选择一个 CSV 文件。")
    workspace = resolve_within(root, f"outputs/ui-import-{import_id}")
    if workspace.exists():
        raise ValueError("本次导入目录已经存在，请重新点击导入。")
    raw_dir = workspace / "raw"
    raw_dir.mkdir(parents=True)
    total_bytes = 0
    names: set[str] = set()
    try:
        for upload in uploads:
            name = _safe_csv_name(upload.name)
            if name in names:
                raise ValueError(f"存在同名文件：{name}")
            payload = upload.getvalue()
            if not payload:
                raise ValueError(f"文件为空：{name}")
            total_bytes += len(payload)
            if total_bytes > MAX_UPLOAD_BYTES:
                raise ValueError("上传文件总大小不能超过 1 GiB。")
            _validate_csv_header(name, payload)
            temporary = raw_dir / f".{name}.uploading"
            temporary.write_bytes(payload)
            temporary.replace(raw_dir / name)
            names.add(name)
    except (OSError, UnicodeError, csv.Error, ValueError):
        shutil.rmtree(workspace, ignore_errors=True)
        raise
    return ImportedData(
        workspace=workspace,
        raw_dir=raw_dir,
        processed_root=workspace / "processed",
        preparation_root=workspace / "preparation",
        file_count=len(names),
        total_bytes=total_bytes,
    )


def find_processed_summary(processed_root: str | Path) -> ProcessedDataSummary:
    """Read the single split produced by one UI import."""

    root = Path(processed_root)
    manifests = sorted(root.glob("*/split_manifest.json"))
    if len(manifests) != 1:
        raise ValueError("预处理结果必须且只能包含一个数据划分。")
    manifest_path = manifests[0]
    partition_path = manifest_path.parent / "partition_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        partition = json.loads(partition_path.read_text(encoding="utf-8"))
        splits = manifest["splits"]
        stats = manifest["stats"]
        clients = partition["clients"]
        sample_counts = {
            name: int(splits[name]["sample_count"])
            for name in ("train", "validation", "test")
        }
        client_counts = {str(item["client_id"]): int(item["sample_count"]) for item in clients}
        return ProcessedDataSummary(
            processed_dir=manifest_path.parent,
            data_version=str(manifest["data_version"]),
            split_id=str(manifest["split_id"]),
            input_rows=int(stats["input_rows"]),
            valid_tracks=int(stats["valid_tracks"]),
            rejected_tracks=int(stats["rejected_tracks"]),
            sample_counts=sample_counts,
            client_counts=client_counts,
        )
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取预处理摘要：{exc}") from exc


def _safe_csv_name(value: str) -> str:
    name = Path(value).name
    if name != value or not name or name.startswith(".") or Path(name).suffix.lower() != ".csv":
        raise ValueError(f"仅支持安全的 .csv 文件名：{value}")
    return name


def _validate_csv_header(name: str, payload: bytes) -> None:
    try:
        first_line = payload[: min(len(payload), 64 * 1024)].decode("utf-8-sig").splitlines()[0]
    except (UnicodeDecodeError, IndexError) as exc:
        raise ValueError(f"无法读取 CSV 表头：{name}") from exc
    columns = next(csv.reader(StringIO(first_line)))
    normalized = {column.strip().lower().replace("_", " ") for column in columns}
    missing = [target for target, aliases in _COLUMN_ALIASES.items() if not aliases & normalized]
    if missing:
        raise ValueError(f"{name} 缺少必要字段：{', '.join(missing)}")
