from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.data.adapters import HighDAdapter
from src.data.dataset import (
    TrajectoryDataset,
    load_dataset,
    save_dataset,
    save_split_datasets,
)
from src.data.preprocess import WindowSpec


def _config() -> dict[str, object]:
    return {
        "preprocessing": {"minimum_track_frames": 5},
        "split": {
            "seed": 42,
            "train": 0.7,
            "validation": 0.15,
            "test": 0.15,
        },
        "sequence": {
            "history_steps": 2,
            "future_steps": 2,
            "stride": 1,
            "coordinate_dimension": 2,
        },
    }


def _records(vehicle_count: int = 10) -> pd.DataFrame:
    rows = []
    for vehicle_id in range(1, vehicle_count + 1):
        for frame in range(5):
            rows.append(
                {
                    "Track ID": vehicle_id,
                    "Frame ID": frame,
                    "x Position": float(vehicle_id + frame),
                    "y Position": float(vehicle_id * 2 + frame),
                }
            )
    return pd.DataFrame(rows)


def test_highd_pipeline_maps_fields_splits_before_windowing_and_round_trips(tmp_path) -> None:
    adapter = HighDAdapter()
    cleaned = adapter.preprocess(_records(), _config())
    samples = adapter.build_samples(cleaned, _config())

    assert samples
    assert all(sample.history.shape == (2, 2) for sample in samples)
    assert all(sample.future.shape == (2, 2) for sample in samples)
    groups = {
        (sample.meta["recording_id"], sample.meta["vehicle_id"]): sample.meta["split"]
        for sample in samples
    }
    assert len(groups) == 10
    assert {sample.meta["split"] for sample in samples} == {"train", "validation", "test"}
    for sample in samples:
        assert sample.meta["history_end_frame"] < sample.meta["future_start_frame"]

    train_samples = [sample for sample in samples if sample.meta["split"] == "train"]
    train_dataset = TrajectoryDataset(
        train_samples,
        split="train",
        split_id=str(cleaned["split_id"]),
        window_spec=WindowSpec(**_config()["sequence"]),
    )
    output = save_dataset(
        train_dataset,
        tmp_path / "train",
        scaler=cleaned["scaler"],
        stats=cleaned["stats"],
    )
    restored, scaler, stats = load_dataset(output)
    assert len(restored) == len(train_dataset)
    assert np.array_equal(restored[0].history, train_dataset[0].history)
    assert stats["rejected_tracks"] == 0
    physical = np.array([[1.0, 2.0]], dtype=np.float32)
    assert np.allclose(scaler.inverse_transform(scaler.transform(physical)), physical)

    datasets = adapter.build_datasets(cleaned, _config())
    split_output = save_split_datasets(
        datasets,
        tmp_path / "all-splits",
        scaler=cleaned["scaler"],
        stats=cleaned["stats"],
        data_version=str(cleaned["data_version"]),
    )
    manifest = json.loads((split_output / "split_manifest.json").read_text(encoding="utf-8"))
    assert set(datasets) == {"train", "validation", "test"}
    assert manifest["split_id"] == str(cleaned["split_id"])


def test_highd_pipeline_rejects_bad_tracks_and_reports_statistics() -> None:
    records = _records()
    records = pd.concat(
        [
            records,
            pd.DataFrame(
                [
                    {"Track ID": 99, "Frame ID": 0, "x Position": 0.0, "y Position": 0.0},
                    {"Track ID": 99, "Frame ID": 0, "x Position": 1.0, "y Position": 1.0},
                ]
            ),
        ],
        ignore_index=True,
    )
    cleaned = HighDAdapter().preprocess(records, _config())

    assert cleaned["stats"]["rejected_tracks"] == 1
    assert cleaned["stats"]["rejected_rows"] == 2
