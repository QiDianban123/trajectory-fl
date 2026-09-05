"""Shared training contracts and mode-specific orchestration adapters."""

from src.training.batching import collate_trajectory_samples, sample_to_tensor
from src.training.trainer import EvaluationResult, FitResult, Trainer, TrajectoryBatch

__all__ = [
    "EvaluationResult",
    "FitResult",
    "Trainer",
    "TrajectoryBatch",
    "collate_trajectory_samples",
    "sample_to_tensor",
]
