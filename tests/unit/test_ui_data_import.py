"""UI data-import boundary and processed-summary tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.ui.data_import import find_processed_summary, save_uploaded_csvs


@dataclass
class Upload:
    name: str
    payload: bytes

    def getvalue(self) -> bytes:
        return self.payload


def test_csv_import_accepts_highd_aliases_and_stays_under_outputs(tmp_path: Path) -> None:
    uploaded = Upload(
        "01_tracks.csv",
        b"Track ID,Frame ID,x Position,y Position\n1,0,1.0,2.0\n",
    )
    imported = save_uploaded_csvs(tmp_path, [uploaded], import_id="safe")
    assert imported.raw_dir == tmp_path / "outputs/ui-import-safe/raw"
    assert (imported.raw_dir / uploaded.name).read_bytes() == uploaded.payload
    assert imported.file_count == 1


def test_csv_import_rejects_unsafe_name_and_missing_columns(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="文件名"):
        save_uploaded_csvs(
            tmp_path,
            [Upload("../tracks.csv", b"id,frame,x,y\n1,0,0,0\n")],
            import_id="unsafe",
        )
    with pytest.raises(ValueError, match="必要字段"):
        save_uploaded_csvs(
            tmp_path,
            [Upload("tracks.csv", b"id,frame,x\n1,0,0\n")],
            import_id="missing",
        )
    assert not (tmp_path / "outputs/ui-import-missing").exists()


def test_processed_summary_reads_split_and_rsu_counts(tmp_path: Path) -> None:
    split = tmp_path / "processed/split-1"
    split.mkdir(parents=True)
    (split / "split_manifest.json").write_text(
        json.dumps(
            {
                "data_version": "data-1",
                "split_id": "split-1",
                "splits": {
                    "train": {"sample_count": 7},
                    "validation": {"sample_count": 2},
                    "test": {"sample_count": 1},
                },
                "stats": {"input_rows": 1000, "valid_tracks": 5, "rejected_tracks": 1},
            }
        ),
        encoding="utf-8",
    )
    (split / "partition_manifest.json").write_text(
        json.dumps({"clients": [{"client_id": "rsu_01", "sample_count": 7}]}),
        encoding="utf-8",
    )
    summary = find_processed_summary(tmp_path / "processed")
    assert summary.sample_counts == {"train": 7, "validation": 2, "test": 1}
    assert summary.client_counts == {"rsu_01": 7}
