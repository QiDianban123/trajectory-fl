"""CPU-only tests for the Day 2 model and shared Trainer contracts."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pytest
import torch

from src.models.base import (
    BaseTrajectoryModel,
    ModelContract,
    ModelContractError,
    validate_checkpoint_payload,
    validate_future_tensor,
)
from src.training.centralized import CentralizedTrainingRequest, run_centralized
from src.training.local_only import LocalTrainingRequest, run_local_only
from src.training.trainer import (
    EpochStats,
    EvaluationResult,
    FitResult,
    TrajectoryBatch,
)


CONTRACT = ModelContract(history_steps=2, future_steps=3)


class ShapeCheckedPredictor(BaseTrajectoryModel):
    """Test double only; it verifies the contract without implementing an LSTM."""

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        self.validate_history(history)
        prediction = torch.zeros(
            (history.shape[0], self.contract.future_steps, 2),
            dtype=history.dtype,
            device=history.device,
        )
        self.validate_prediction(prediction, history)
        return prediction


def _batch() -> TrajectoryBatch:
    return TrajectoryBatch(
        history=torch.zeros((2, 2, 2), dtype=torch.float32),
        future=torch.zeros((2, 3, 2), dtype=torch.float32),
    )


def _checkpoint_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model_state": {},
        "model_config": {"name": "lstm_encoder_decoder"},
        "seed": 42,
        "epoch": 0,
        "split_id": "highd-split-42",
        "metrics": {"loss": 0.0},
    }


def _fit_result() -> FitResult:
    return FitResult(
        epoch_stats=(EpochStats(epoch=0, sample_count=2, train_loss=0.1),),
        best_epoch=0,
        checkpoint_payload=_checkpoint_payload(),
    )


def test_model_contract_enforces_batch_shape_dtype_and_normalized_output() -> None:
    model = ShapeCheckedPredictor(CONTRACT)
    prediction = model(torch.zeros((2, 2, 2), dtype=torch.float32))
    assert prediction.shape == (2, 3, 2)
    assert prediction.dtype == torch.float32

    with pytest.raises(ModelContractError, match="torch.float32"):
        model(torch.zeros((2, 2, 2), dtype=torch.float64))
    with pytest.raises(ModelContractError, match="B, 3, 2"):
        validate_future_tensor(torch.zeros((2, 2, 2), dtype=torch.float32), CONTRACT)


def test_checkpoint_contract_rejects_missing_reproducibility_metadata() -> None:
    payload = _checkpoint_payload()
    validate_checkpoint_payload(payload)
    payload.pop("split_id")
    with pytest.raises(ModelContractError, match="checkpoint is missing keys"):
        validate_checkpoint_payload(payload)


def test_trajectory_batch_requires_same_shape_and_device_contract() -> None:
    _batch().validate(CONTRACT)
    invalid = TrajectoryBatch(
        history=torch.zeros((2, 2, 2), dtype=torch.float32),
        future=torch.zeros((1, 3, 2), dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="batch sizes must match"):
        invalid.validate(CONTRACT)


class RecordingTrainer:
    """Trainer double that proves mode adapters delegate to the common API."""

    def __init__(self) -> None:
        self.calls: list[tuple[object, object, object, Mapping[str, Any]]] = []

    def fit(
        self,
        model: ShapeCheckedPredictor,
        train_batches: Iterable[TrajectoryBatch],
        validation_batches: Iterable[TrajectoryBatch] | None,
        *,
        initial_state: Mapping[str, Any],
    ) -> FitResult:
        self.calls.append((model, train_batches, validation_batches, initial_state))
        return _fit_result()

    def evaluate(
        self, model: ShapeCheckedPredictor, batches: Iterable[TrajectoryBatch]
    ) -> EvaluationResult:
        return EvaluationResult(sample_count=2, loss=0.1)


def test_centralized_and_local_only_reuse_one_trainer_interface() -> None:
    trainer = RecordingTrainer()
    model = ShapeCheckedPredictor(CONTRACT)
    initial_state: Mapping[str, Any] = {"weight": torch.tensor([1.0])}

    centralized = run_centralized(
        trainer,
        CentralizedTrainingRequest(model, [_batch()], None, initial_state),
    )
    local = run_local_only(
        trainer,
        LocalTrainingRequest("rsu-1", model, [_batch()], None, initial_state),
    )

    assert centralized.best_epoch == 0
    assert local.best_epoch == 0
    assert len(trainer.calls) == 2
    assert trainer.calls[0][3] is initial_state
    assert trainer.calls[1][3] is initial_state
