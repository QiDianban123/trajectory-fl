"""Constant-velocity baseline contract and physical-metric tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.preprocess import TrainingCoordinateScaler
from src.evaluation.centralized import evaluate_prediction_arrays
from src.models.base import ModelContract, ModelContractError
from src.models.constant_velocity import ConstantVelocityBaseline


def _model() -> ConstantVelocityBaseline:
    return ConstantVelocityBaseline(ModelContract(history_steps=3, future_steps=2))


def _scaler() -> TrainingCoordinateScaler:
    scaler = TrainingCoordinateScaler()
    scaler.mean_ = np.array([10.0, 20.0], dtype=np.float32)
    scaler.scale_ = np.array([2.0, 4.0], dtype=np.float32)
    scaler.fitted_split = "train"
    return scaler


def test_stationary_and_constant_velocity_batches_have_frozen_shape() -> None:
    history = torch.tensor(
        [
            [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]],
            [[0.0, 0.0], [1.0, -1.0], [2.0, -2.0]],
        ],
        dtype=torch.float32,
    )

    prediction = _model()(history)

    assert prediction.shape == (2, 2, 2)
    torch.testing.assert_close(prediction[0], torch.tensor([[1.0, 2.0], [1.0, 2.0]]))
    torch.testing.assert_close(
        prediction[1], torch.tensor([[3.0, -3.0], [4.0, -4.0]])
    )


@pytest.mark.parametrize(
    "history",
    [
        torch.zeros((3, 2), dtype=torch.float32),
        torch.zeros((0, 3, 2), dtype=torch.float32),
        torch.zeros((1, 2, 2), dtype=torch.float32),
        torch.zeros((1, 3, 2), dtype=torch.float64),
        torch.full((1, 3, 2), float("nan"), dtype=torch.float32),
    ],
)
def test_baseline_rejects_invalid_standard_history(history: torch.Tensor) -> None:
    with pytest.raises(ModelContractError):
        _model()(history)


def test_baseline_uses_shared_scaler_and_meter_metrics() -> None:
    history = torch.tensor(
        [[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]], dtype=torch.float32
    )
    prediction = _model()(history).detach().numpy()
    truth = prediction.copy()
    truth[:, :, 0] += 1.0

    evaluated = evaluate_prediction_arrays(prediction, truth, _scaler())

    assert evaluated.trajectories.sample_count == 1
    assert evaluated.metrics == {"ade": 2.0, "fde": 2.0}
