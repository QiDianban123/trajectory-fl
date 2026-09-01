"""Local-only mode adapter; each client still uses the same Trainer contract."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from src.models.base import TrajectoryPredictor
from src.training.trainer import FitResult, Trainer, TrajectoryBatch


@dataclass(frozen=True)
class LocalTrainingRequest:
    """One client's isolated training request with a common initial-state snapshot."""

    client_id: str
    model: TrajectoryPredictor
    train_batches: Iterable[TrajectoryBatch]
    validation_batches: Iterable[TrajectoryBatch] | None
    initial_state: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.client_id.strip():
            raise ValueError("client_id must be a non-empty string")


def run_local_only(trainer: Trainer, request: LocalTrainingRequest) -> FitResult:
    """Delegate one local-only client to Trainer; no parameter sharing occurs here."""

    return trainer.fit(
        request.model,
        request.train_batches,
        request.validation_batches,
        initial_state=request.initial_state,
    )
