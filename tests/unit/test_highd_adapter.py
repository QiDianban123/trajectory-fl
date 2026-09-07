from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data.adapters import HighDAdapter, TrajectorySample
from src.data.dataset import (
    TrajectoryDataset,
    load_dataset,
    save_dataset,
    save_split_datasets,
)
from src.data.preprocess import TrainingCoordinateScaler, WindowSpec


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


def test_highd_pipeline_rejects_out_of_order_frames_with_a_reason() -> None:
    records = _records()
    records.loc[1, "Frame ID"] = 2
    records.loc[2, "Frame ID"] = 1

    cleaned = HighDAdapter().preprocess(records, _config())

    assert 1 not in set(cleaned["records"]["id"])
    assert cleaned["stats"]["rejection_reasons"] == {"out_of_order_frame": 1}


def test_split_id_changes_when_processed_data_changes() -> None:
    records = _records()
    changed = records.copy()
    changed["x Position"] += 1000.0

    first = HighDAdapter().preprocess(records, _config())
    second = HighDAdapter().preprocess(changed, _config())

    assert first["data_version"] != second["data_version"]
    assert first["split_id"] != second["split_id"]


def test_highd_pipeline_rejects_missing_non_integer_and_short_tracks() -> None:
    adapter = HighDAdapter()
    with pytest.raises(ValueError, match="missing required columns: y"):
        adapter.preprocess(_records().drop(columns="y Position"), _config())

    records = _records()
    records["Frame ID"] = records["Frame ID"].astype(float)
    records.loc[0, "Frame ID"] = 0.5
    short = pd.DataFrame(
        [
            {"Track ID": 99, "Frame ID": frame, "x Position": 1.0, "y Position": 2.0}
            for frame in range(4)
        ]
    )
    cleaned = adapter.preprocess(pd.concat([records, short], ignore_index=True), _config())

    reasons = cleaned["stats"]["rejection_reasons"]
    assert reasons == {"non_integer_id_or_frame": 1, "short_track": 1}


def test_splitter_allocates_every_split_for_the_smallest_legal_group_count() -> None:
    cleaned = HighDAdapter().preprocess(_records(vehicle_count=3), _config())

    assert cleaned["stats"]["split_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert {record["split"] for _, record in cleaned["records"].iterrows()} == {
        "train",
        "validation",
        "test",
    }


def test_persistence_rejects_invalid_scaler_and_inconsistent_splits(tmp_path) -> None:
    cleaned = HighDAdapter().preprocess(_records(), _config())
    datasets = HighDAdapter().build_datasets(cleaned, _config())
    train = datasets["train"]

    with pytest.raises(ValueError, match="scaler must be fitted"):
        save_dataset(
            train, tmp_path / "invalid-scaler", scaler=TrainingCoordinateScaler(), stats={}
        )
    assert not (tmp_path / "invalid-scaler").exists()

    inconsistent = dict(datasets)
    mismatched_samples = [
        TrajectorySample(
            sample.history,
            sample.future,
            {**sample.meta, "split_id": "different-split"},
        )
        for sample in datasets["validation"]
    ]
    inconsistent["validation"] = TrajectoryDataset(
        mismatched_samples,
        split="validation",
        split_id="different-split",
        window_spec=datasets["validation"].window_spec,
    )
    with pytest.raises(ValueError, match="share one split_id"):
        save_split_datasets(
            inconsistent,
            tmp_path / "inconsistent",
            scaler=cleaned["scaler"],
            stats=cleaned["stats"],
            data_version=cleaned["data_version"],
        )
    assert not (tmp_path / "inconsistent").exists()


def test_load_dataset_rejects_corrupt_scaler(tmp_path) -> None:
    cleaned = HighDAdapter().preprocess(_records(), _config())
    dataset = HighDAdapter().build_datasets(cleaned, _config())["train"]
    output = save_dataset(dataset, tmp_path / "train", scaler=cleaned["scaler"], stats={})
    np.savez_compressed(
        output / "scaler.npz",
        mean=np.array([np.nan, 0.0], dtype=np.float32),
        scale=np.array([1.0, 1.0], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="two finite coordinates"):
        load_dataset(output)
