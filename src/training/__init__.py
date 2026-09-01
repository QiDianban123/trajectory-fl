"""Shared training contracts and mode-specific orchestration adapters."""

from src.training.trainer import EvaluationResult, FitResult, Trainer, TrajectoryBatch

__all__ = ["EvaluationResult", "FitResult", "Trainer", "TrajectoryBatch"]
