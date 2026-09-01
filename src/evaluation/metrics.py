"""Dataset-independent ADE/FDE metrics in physical coordinates."""

from __future__ import annotations

import numpy as np

PHYSICAL_COORDINATE_UNIT = "meter"
SUPPORTED_MODES = ("centralized", "local_only", "federated")


def _validate_pair(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    coordinate_unit: str = PHYSICAL_COORDINATE_UNIT,
) -> tuple[np.ndarray, np.ndarray]:
    if coordinate_unit != PHYSICAL_COORDINATE_UNIT:
        raise ValueError(
            "ADE/FDE require physical coordinates in meters; inverse-transform first"
        )
    pred = np.asarray(prediction, dtype=float)
    truth = np.asarray(target, dtype=float)
    if pred.shape != truth.shape:
        raise ValueError(f"prediction shape {pred.shape} does not match target shape {truth.shape}")
    if pred.ndim not in (2, 3) or pred.shape[-1] != 2 or pred.shape[-2] == 0:
        raise ValueError("expected [T, 2] or [B, T, 2] with at least one future step")
    if not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError("prediction and target must contain only finite values")
    return pred, truth


def displacement_errors(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    coordinate_unit: str = PHYSICAL_COORDINATE_UNIT,
) -> np.ndarray:
    """Return Euclidean position errors, shaped [T] or [B, T]."""
    pred, truth = _validate_pair(prediction, target, coordinate_unit=coordinate_unit)
    return np.linalg.norm(pred - truth, axis=-1)


def ade(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    coordinate_unit: str = PHYSICAL_COORDINATE_UNIT,
) -> float:
    """Average displacement error across all examples and future steps."""
    return float(
        displacement_errors(prediction, target, coordinate_unit=coordinate_unit).mean()
    )


def fde(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    coordinate_unit: str = PHYSICAL_COORDINATE_UNIT,
) -> float:
    """Final displacement error, averaged across examples when batched."""
    return float(
        displacement_errors(prediction, target, coordinate_unit=coordinate_unit)[..., -1].mean()
    )


def compute_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    coordinate_unit: str = PHYSICAL_COORDINATE_UNIT,
) -> dict[str, float]:
    """Return the canonical ADE/FDE mapping for one physical-coordinate evaluation."""

    return {
        "ade": ade(prediction, target, coordinate_unit=coordinate_unit),
        "fde": fde(prediction, target, coordinate_unit=coordinate_unit),
    }
