"""D1 model contract, to be frozen in the D2 design review."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, TypeAlias

try:
    import torch

    TrajectoryTensor: TypeAlias = torch.Tensor
except ImportError:  # Allows document/interface inspection before dependencies are installed.
    TrajectoryTensor: TypeAlias = object


class TrajectoryPredictor(Protocol):
    """Predict a future two-dimensional trajectory from a history trajectory.

    `history` has shape [batch, history_steps, 2], and the returned tensor has
    shape [batch, future_steps, 2]. Both are normalized coordinates during training.
    """

    def forward(self, history: TrajectoryTensor) -> TrajectoryTensor: ...


class BaseTrajectoryModel(ABC):
    """Optional abstract base for concrete PyTorch models introduced on D5."""

    @abstractmethod
    def forward(self, history: TrajectoryTensor) -> TrajectoryTensor:
        """Return future coordinates with shape [B, T_f, 2]."""
        raise NotImplementedError
