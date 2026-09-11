"""S2-E centralized physical evaluation and artifact tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.data.preprocess import TrainingCoordinateScaler
from src.evaluation.centralized import (
    CentralizedEvaluationRequest,
    LossHistory,
    evaluate_centralized,
    inverse_transform_batch,
)
from src.evaluation.result_store import CSV_FIELDS


def _scaler() -> TrainingCoordinateScaler:
    scaler = TrainingCoordinateScaler()
    scaler.mean_ = np.array([10.0, 20.0], dtype=np.float32)
    scaler.scale_ = np.array([2.0, 4.0], dtype=np.float32)
    scaler.fitted_split = "train"
    return scaler


def _request(tmp_path: Path) -> CentralizedEvaluationRequest:
    prediction = np.zeros((2, 2, 2), dtype=np.float32)
    truth = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ],
        dtype=np.float32,
    )
    return CentralizedEvaluationRequest(
        prediction=prediction,
        truth=truth,
        scaler=_scaler(),
        loss_history=LossHistory(train=(1.0, 0.5), validation=(1.2, 0.6)),
        run_id="20260911T010203Z-centralized-seed42",
        code_sha="abc123",
        seed=42,
        split_id="highd-split-42",
        total_seconds=3.5,
        output_dir=tmp_path,
    )


def test_inverse_transform_batch_restores_all_samples_in_meter_coordinates() -> None:
    prediction = np.zeros((2, 2, 2), dtype=np.float32)
    truth = np.ones((2, 2, 2), dtype=np.float32)

    physical = inverse_transform_batch(prediction, truth, _scaler())

    np.testing.assert_allclose(physical.prediction, [[[10.0, 20.0]] * 2] * 2)
    np.testing.assert_allclose(physical.truth, [[[12.0, 24.0]] * 2] * 2)
    assert physical.sample_count == 2


def test_centralized_evaluation_uses_known_meter_distances_and_batch_count(
    tmp_path: Path,
) -> None:
    output = evaluate_centralized(_request(tmp_path))

    assert output.record.coordinate_unit == "meter"
    assert output.record.sample_count == 2
    assert output.record.ade == pytest.approx(3.0)
    assert output.record.fde == pytest.approx(4.0)


def test_centralized_json_csv_share_one_record_and_declared_artifacts_exist(
    tmp_path: Path,
) -> None:
    output = evaluate_centralized(_request(tmp_path))
    payload = json.loads(output.json_path.read_text(encoding="utf-8"))
    with output.csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert tuple(rows[0]) == CSV_FIELDS
    assert len(rows) == 1
    row = rows[0]
    assert row["run_id"] == payload["run_id"] == output.record.run_id
    assert int(row["sample_count"]) == payload["sample_count"] == 2
    assert float(row["ade"]) == payload["metrics"]["ade"]
    assert float(row["fde"]) == payload["metrics"]["fde"]
    assert row["coordinate_unit"] == payload["coordinate_unit"] == "meter"
    assert json.loads(row["artifact_paths"]) == payload["artifacts"]
    for relative_path in payload["artifacts"].values():
        path = tmp_path / relative_path
        assert path.is_file()
        assert path.stat().st_size > 0


def test_centralized_figures_can_be_rebuilt_from_same_saved_inputs(tmp_path: Path) -> None:
    first = evaluate_centralized(_request(tmp_path / "first"))
    second = evaluate_centralized(_request(tmp_path / "rebuilt"))

    assert first.record.to_dict() == second.record.to_dict()
    assert second.loss_curve_path.is_file()
    assert second.trajectory_path.is_file()


@pytest.mark.parametrize(
    ("prediction", "truth", "error", "match"),
    [
        (
            np.zeros((2, 2), dtype=np.float32),
            np.zeros((2, 2), dtype=np.float32),
            ValueError,
            "B, T, 2",
        ),
        (
            np.zeros((1, 2, 2), dtype=np.float32),
            np.zeros((2, 2, 2), dtype=np.float32),
            ValueError,
            "does not match",
        ),
        (
            np.zeros((1, 2, 2), dtype=np.int64),
            np.zeros((1, 2, 2), dtype=np.float32),
            TypeError,
            "floating-point",
        ),
        (
            np.full((1, 2, 2), np.nan, dtype=np.float32),
            np.zeros((1, 2, 2), dtype=np.float32),
            ValueError,
            "finite",
        ),
        (
            np.zeros((1, 0, 2), dtype=np.float32),
            np.zeros((1, 0, 2), dtype=np.float32),
            ValueError,
            "B, T, 2",
        ),
    ],
)
def test_inverse_transform_batch_rejects_invalid_inputs(
    prediction: np.ndarray,
    truth: np.ndarray,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        inverse_transform_batch(prediction, truth, _scaler())


class _CorruptScaler:
    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return np.full(values.shape, np.inf, dtype=np.float32)


def test_inverse_transform_batch_rejects_corrupt_scaler_output() -> None:
    values = np.zeros((1, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="restored prediction.*finite"):
        inverse_transform_batch(values, values, _CorruptScaler())


def test_loss_history_rejects_invalid_or_mismatched_logs() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        LossHistory(train=())
    with pytest.raises(ValueError, match="finite"):
        LossHistory(train=(float("nan"),))
    with pytest.raises(ValueError, match="match train"):
        LossHistory(train=(1.0, 0.5), validation=(1.0,))
