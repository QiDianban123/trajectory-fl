"""Deterministic plotting interfaces for trajectories and experiment summaries."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.metrics import PHYSICAL_COORDINATE_UNIT, ade, fde
from src.evaluation.result_store import ResultRecord


def plot_trajectory(
    history: np.ndarray,
    future: np.ndarray,
    prediction: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Save a history/ground-truth/prediction plot and return its path."""

    history_values = _validate_xy(history, "history")
    future_values, prediction_values = _validate_pair(future, prediction)
    path = _prepare_output_path(output_path)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(*history_values.T, "o-", label="history")
    axis.plot(*future_values.T, "o-", label="future truth")
    axis.plot(*prediction_values.T, "o--", label="prediction")
    axis.set(
        title=f"Trajectory | ADE={ade(prediction_values, future_values):.3f}, "
        f"FDE={fde(prediction_values, future_values):.3f}",
        xlabel="x (m)",
        ylabel="y (m)",
    )
    axis.legend()
    axis.axis("equal")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_convergence(
    rounds: Sequence[int], values: Sequence[float], output_path: str | Path, *, label: str = "loss"
) -> Path:
    """Save a finite round/metric convergence curve."""

    round_values = np.asarray(rounds)
    metric_values = np.asarray(values, dtype=float)
    if round_values.ndim != 1 or metric_values.ndim != 1 or len(round_values) == 0:
        raise ValueError("rounds and values must be non-empty one-dimensional sequences")
    if len(round_values) != len(metric_values):
        raise ValueError("rounds and values must have equal length")
    if not np.isfinite(metric_values).all():
        raise ValueError("convergence values must be finite")
    path = _prepare_output_path(output_path)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot(round_values, metric_values, "o-")
    axis.set(xlabel="round", ylabel=label, title=f"Federated convergence ({label})")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_mode_comparison(records: Sequence[ResultRecord], output_path: str | Path) -> Path:
    """Save ADE/FDE bars for canonical result records."""

    if not records:
        raise ValueError("at least one result record is required")
    modes = [record.mode for record in records]
    ade_values = np.asarray([record.ade for record in records], dtype=float)
    fde_values = np.asarray([record.fde for record in records], dtype=float)
    if len(set(modes)) != len(modes):
        raise ValueError("mode comparison requires at most one record per mode")
    if not np.isfinite(ade_values).all() or not np.isfinite(fde_values).all():
        raise ValueError("comparison metrics must be finite")
    path = _prepare_output_path(output_path)

    positions = np.arange(len(modes))
    width = 0.38
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(positions - width / 2, ade_values, width, label="ADE")
    axis.bar(positions + width / 2, fde_values, width, label="FDE")
    axis.set_xticks(positions, modes)
    axis.set_ylabel(f"error ({PHYSICAL_COORDINATE_UNIT})")
    axis.set_title("Training mode comparison")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _validate_xy(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape [T, 2] with T > 0")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_pair(first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first_values = _validate_xy(first, "future")
    second_values = _validate_xy(second, "prediction")
    if first_values.shape != second_values.shape:
        raise ValueError("future and prediction shapes must match")
    return first_values, second_values


def _prepare_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if not path.name or not path.suffix:
        raise ValueError("output_path must include a filename with an extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
