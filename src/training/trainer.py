"""Common Trainer contract used by centralized, local-only, and federated modes."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol

from src.models.base import (
    ModelContract,
    TrajectoryPredictor,
    TrajectoryTensor,
    require_torch,
    validate_checkpoint_payload,
    validate_future_tensor,
    validate_history_tensor,
)


@dataclass(frozen=True)
class TrajectoryBatch:
    """One normalized batch, with the caller retaining DataLoader ownership."""

    history: TrajectoryTensor
    future: TrajectoryTensor
    meta: tuple[Mapping[str, object], ...] = ()

    def validate(self, contract: ModelContract) -> None:
        torch = require_torch()
        # Check devices before finite-value operations (e.g. on a meta tensor).
        for name in ("history", "future"):
            if not isinstance(getattr(self, name), torch.Tensor):
                raise ValueError(f"{name} must be a torch.Tensor")
        if self.history.device != self.future.device:
            raise ValueError("history and future must be on the same device")
        if self.history.device.type == "meta":
            raise ValueError("trajectory batches require a device with materialized values")
        validate_history_tensor(self.history, contract)
        validate_future_tensor(self.future, contract)
        if self.history.shape[0] != self.future.shape[0]:
            raise ValueError("history and future batch sizes must match")
        if self.meta:
            if len(self.meta) != self.history.shape[0]:
                raise ValueError("meta must contain one mapping per sample")
            if not all(isinstance(item, Mapping) for item in self.meta):
                raise ValueError("meta must contain one mapping per sample")


@dataclass(frozen=True)
class EpochStats:
    """One epoch's scalar statistics, recorded without embedding raw batches."""

    epoch: int
    sample_count: int
    train_loss: float
    validation_loss: float | None = None

    def __post_init__(self) -> None:
        if self.epoch < 0 or self.sample_count <= 0:
            raise ValueError("epoch must be non-negative and sample_count must be positive")
        _finite_number(self.train_loss, "train_loss")
        if self.validation_loss is not None:
            _finite_number(self.validation_loss, "validation_loss")


@dataclass(frozen=True)
class FitResult:
    """The common return value from ``Trainer.fit`` for all training modes."""

    epoch_stats: tuple[EpochStats, ...]
    best_epoch: int
    checkpoint_payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.epoch_stats:
            raise ValueError("fit result must include at least one epoch statistic")
        if self.best_epoch not in {stat.epoch for stat in self.epoch_stats}:
            raise ValueError("best_epoch must identify one returned epoch statistic")
        validate_checkpoint_payload(self.checkpoint_payload)


@dataclass(frozen=True)
class EvaluationResult:
    """Mode-neutral validation result; ADE/FDE are added by the evaluation layer."""

    sample_count: int
    loss: float

    def __post_init__(self) -> None:
        if self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        _finite_number(self.loss, "loss")


class Trainer(Protocol):
    """Only component permitted to own optimizer, device transfer, and train mode."""

    def fit(
        self,
        model: TrajectoryPredictor,
        train_batches: Iterable[TrajectoryBatch],
        validation_batches: Iterable[TrajectoryBatch] | None,
        *,
        initial_state: Mapping[str, Any],
    ) -> FitResult:
        """Train from a caller-supplied initial state and return checkpoint metadata."""
        ...

    def evaluate(
        self, model: TrajectoryPredictor, batches: Iterable[TrajectoryBatch]
    ) -> EvaluationResult:
        """Evaluate without changing model parameters."""
        ...


def _finite_number(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
        raise ValueError(f"{name} must be a finite number")
