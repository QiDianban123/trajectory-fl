"""Dataset-independent ADE/FDE metrics in physical coordinates."""

from __future__ import annotations

import numpy as np


def _validate_pair(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=float)
    truth = np.asarray(target, dtype=float)
    if pred.shape != truth.shape:
        raise ValueError(f"prediction shape {pred.shape} does not match target shape {truth.shape}")
    if pred.ndim not in (2, 3) or pred.shape[-1] != 2 or pred.shape[-2] == 0:
        raise ValueError("expected [T, 2] or [B, T, 2] with at least one future step")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("prediction and target must contain only finite values")
    return pred, truth


def displacement_errors(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return Euclidean position errors, shaped [T] or [B, T]."""
    pred, truth = _validate_pair(prediction, target)
    return np.linalg.norm(pred - truth, axis=-1)


def ade(prediction: np.ndarray, target: np.ndarray) -> float:
    """Average displacement error across all examples and future steps."""
    return float(displacement_errors(prediction, target).mean())


def fde(prediction: np.ndarray, target: np.ndarray) -> float:
    """Final displacement error, averaged across examples when batched."""
    return float(displacement_errors(prediction, target)[..., -1].mean())
