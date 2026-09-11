"""Prediction collection boundary shared by experiment runners."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.models.base import (
    ModelContract,
    TrajectoryPredictor,
    require_torch,
    validate_prediction_tensor,
)
from src.training.trainer import TrajectoryBatch


@dataclass(frozen=True)
class PredictionCollection:
    """Normalized arrays collected in stable batch and row order."""

    history: np.ndarray
    prediction: np.ndarray
    truth: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.prediction.shape[0])


def collect_predictions(
    model: TrajectoryPredictor,
    batches: Iterable[TrajectoryBatch],
    *,
    contract: ModelContract,
    device: str | Any = "cpu",
) -> PredictionCollection:
    """Run finite inference without changing model mode or implementing metrics."""

    torch = require_torch()
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if getattr(model, "contract", None) != contract:
        raise ValueError("model contract does not match prediction contract")
    target_device = torch.device(device)
    module = model.to(target_device)
    was_training = module.training
    histories: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    module.eval()
    try:
        with torch.no_grad():
            for batch in batches:
                if not isinstance(batch, TrajectoryBatch):
                    raise TypeError("batches must contain TrajectoryBatch values")
                batch.validate(contract)
                history = batch.history.to(target_device)
                truth = batch.future.to(target_device)
                prediction = module(history)
                validate_prediction_tensor(prediction, history, contract)
                histories.append(history.detach().cpu().numpy().copy())
                predictions.append(prediction.detach().cpu().numpy().copy())
                truths.append(truth.detach().cpu().numpy().copy())
    finally:
        module.train(was_training)
    if not predictions:
        raise ValueError("prediction batches must contain at least one sample")
    return PredictionCollection(
        history=np.concatenate(histories, axis=0),
        prediction=np.concatenate(predictions, axis=0),
        truth=np.concatenate(truths, axis=0),
    )
