"""Centralized mode adapter; its implementation remains in the shared Trainer."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from src.models.base import TrajectoryPredictor
from src.training.trainer import FitResult, Trainer, TrajectoryBatch


@dataclass(frozen=True)
class CentralizedTrainingRequest:
    """Inputs that centralized orchestration forwards unchanged to one Trainer."""

    model: TrajectoryPredictor
    train_batches: Iterable[TrajectoryBatch]
    validation_batches: Iterable[TrajectoryBatch] | None
    initial_state: Mapping[str, Any]


def run_centralized(trainer: Trainer, request: CentralizedTrainingRequest) -> FitResult:
    """Delegate centralized training to the shared Trainer without a duplicate loop."""

    return trainer.fit(
        request.model,
        request.train_batches,
        request.validation_batches,
        initial_state=request.initial_state,
    )
