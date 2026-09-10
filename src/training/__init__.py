"""Shared training contracts and mode-specific orchestration adapters."""

from src.training.batching import collate_trajectory_samples, sample_to_tensor
from src.training.torch_trainer import TorchTrainer, TorchTrainerConfig
from src.training.trainer import EvaluationResult, FitResult, Trainer, TrajectoryBatch

__all__ = [
    "EvaluationResult",
    "FitResult",
    "Trainer",
    "TrajectoryBatch",
    "TorchTrainer",
    "TorchTrainerConfig",
    "collate_trajectory_samples",
    "sample_to_tensor",
]
