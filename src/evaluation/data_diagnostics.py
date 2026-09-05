"""Deterministic plots for validating trajectory preprocessing and scaling."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.metrics import PHYSICAL_COORDINATE_UNIT

_FIGURE_DPI = 160


def plot_raw_cleaned_trajectories(
    raw_xy: np.ndarray,
    cleaned_xy: np.ndarray,
    output_path: str | Path,
    *,
    recording_id: str | int,
    vehicle_id: str | int,
) -> Path:
    """Save side-by-side snapshots of one trajectory before and after cleaning."""

    raw_values = _validate_xy(raw_xy, "raw_xy")
    cleaned_values = _validate_xy(cleaned_xy, "cleaned_xy")
    identity = _trajectory_identity(recording_id, vehicle_id)
    path = _prepare_output_path(output_path)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    panels = (
        (axes[0], raw_values, "raw trajectory", "Before cleaning"),
        (axes[1], cleaned_values, "cleaned trajectory", "After cleaning"),
    )
    for axis, values, label, title in panels:
        axis.plot(*values.T, "o-", markersize=3, label=label)
        axis.set(title=title, xlabel="x (m)", ylabel="y (m)")
        axis.axis("equal")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle(f"Trajectory cleaning snapshot | {identity}")
    figure.tight_layout()
    figure.savefig(path, dpi=_FIGURE_DPI)
    plt.close(figure)
    return path


def plot_anomaly_counts(
    counts: Mapping[str, int],
    output_path: str | Path,
    *,
    title: str = "Data cleaning diagnostics",
) -> Path:
    """Save deterministic anomaly-count bars sorted by category name."""

    labels, values = _validate_anomaly_counts(counts)
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")
    path = _prepare_output_path(output_path)

    figure, axis = plt.subplots(figsize=(max(7, len(labels) * 1.2), 4.5))
    positions = np.arange(len(labels))
    bars = axis.bar(positions, values, color="#D97706")
    axis.set_xticks(positions, labels, rotation=25, ha="right")
    axis.set(title=title.strip(), xlabel="diagnostic category", ylabel="count")
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, labels=[str(value) for value in values], padding=3)
    figure.tight_layout()
    figure.savefig(path, dpi=_FIGURE_DPI)
    plt.close(figure)
    return path


def plot_truth_trajectory(
    history_meter: np.ndarray,
    future_meter: np.ndarray,
    output_path: str | Path,
    *,
    recording_id: str | int | None = None,
    vehicle_id: str | int | None = None,
    coordinate_unit: str = PHYSICAL_COORDINATE_UNIT,
) -> Path:
    """Save history and future ground truth in physical meter coordinates."""

    _require_meter_coordinates(coordinate_unit)
    history_values = _validate_xy(history_meter, "history_meter")
    future_values = _validate_xy(future_meter, "future_meter")
    identity = _optional_trajectory_identity(recording_id, vehicle_id)
    path = _prepare_output_path(output_path)

    figure, axis = plt.subplots(figsize=(6, 4.5))
    axis.plot(*history_values.T, "o-", markersize=3, label="history")
    axis.plot(*future_values.T, "o-", markersize=3, label="future truth")
    axis.set(
        title=f"Ground-truth trajectory{identity}",
        xlabel="x (m)",
        ylabel="y (m)",
    )
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=_FIGURE_DPI)
    plt.close(figure)
    return path


def plot_inverse_transform_check(
    original_meter: np.ndarray,
    restored_meter: np.ndarray,
    output_path: str | Path,
    *,
    recording_id: str | int | None = None,
    vehicle_id: str | int | None = None,
    coordinate_unit: str = PHYSICAL_COORDINATE_UNIT,
) -> Path:
    """Overlay original and inverse-transformed coordinates for a scaler spot check."""

    _require_meter_coordinates(coordinate_unit)
    original_values = _validate_xy(original_meter, "original_meter")
    restored_values = _validate_xy(restored_meter, "restored_meter")
    if original_values.shape != restored_values.shape:
        raise ValueError("original_meter and restored_meter shapes must match")
    identity = _optional_trajectory_identity(recording_id, vehicle_id)
    max_error = float(np.max(np.abs(original_values - restored_values)))
    path = _prepare_output_path(output_path)

    figure, axis = plt.subplots(figsize=(6, 4.5))
    axis.plot(*original_values.T, "o-", markersize=4, label="original meter coordinates")
    axis.plot(
        *restored_values.T,
        "x--",
        markersize=4,
        label="inverse-transformed coordinates",
    )
    axis.set(
        title=f"Scaler inverse-transform check{identity} | max error={max_error:.3g} m",
        xlabel="x (m)",
        ylabel="y (m)",
    )
    axis.axis("equal")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=_FIGURE_DPI)
    plt.close(figure)
    return path


def generate_data_diagnostic_figures(
    *,
    raw_xy: np.ndarray,
    cleaned_xy: np.ndarray,
    anomaly_counts: Mapping[str, int],
    history_meter: np.ndarray,
    future_meter: np.ndarray,
    original_meter: np.ndarray,
    restored_meter: np.ndarray,
    output_dir: str | Path,
    recording_id: str | int,
    vehicle_id: str | int,
) -> dict[str, Path]:
    """Generate the complete S1-E diagnostic set with stable artifact names."""

    directory = _prepare_output_directory(output_dir)
    stem = diagnostic_file_stem(recording_id=recording_id, vehicle_id=vehicle_id)
    return {
        "raw_cleaned": plot_raw_cleaned_trajectories(
            raw_xy,
            cleaned_xy,
            directory / f"{stem}_raw_cleaned.png",
            recording_id=recording_id,
            vehicle_id=vehicle_id,
        ),
        "anomaly_counts": plot_anomaly_counts(
            anomaly_counts,
            directory / f"{stem}_anomaly_counts.png",
        ),
        "truth": plot_truth_trajectory(
            history_meter,
            future_meter,
            directory / f"{stem}_truth.png",
            recording_id=recording_id,
            vehicle_id=vehicle_id,
        ),
        "inverse_transform": plot_inverse_transform_check(
            original_meter,
            restored_meter,
            directory / f"{stem}_inverse_transform.png",
            recording_id=recording_id,
            vehicle_id=vehicle_id,
        ),
    }


def diagnostic_file_stem(*, recording_id: str | int, vehicle_id: str | int) -> str:
    """Return a deterministic, path-safe stem for one recording/vehicle pair."""

    recording = _safe_identifier(recording_id, "recording_id")
    vehicle = _safe_identifier(vehicle_id, "vehicle_id")
    return f"recording_{recording}_vehicle_{vehicle}"


def _validate_xy(values: object, name: str) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 2:
        raise ValueError(f"{name} must have shape [T, 2] with T > 0")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"{name} must use a floating-point dtype")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite coordinates")
    return values


def _validate_anomaly_counts(counts: object) -> tuple[list[str], list[int]]:
    if not isinstance(counts, Mapping) or not counts:
        raise ValueError("counts must be a non-empty mapping")
    normalized: list[tuple[str, int]] = []
    for label, value in counts.items():
        if not isinstance(label, str) or not label.strip():
            raise ValueError("anomaly count labels must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("anomaly counts must be non-negative integers")
        normalized.append((label.strip(), value))
    normalized.sort(key=lambda item: item[0])
    return [label for label, _ in normalized], [value for _, value in normalized]


def _require_meter_coordinates(coordinate_unit: str) -> None:
    if coordinate_unit != PHYSICAL_COORDINATE_UNIT:
        raise ValueError("trajectory diagnostics require physical coordinates in meters")


def _trajectory_identity(recording_id: str | int, vehicle_id: str | int) -> str:
    recording = _safe_identifier(recording_id, "recording_id")
    vehicle = _safe_identifier(vehicle_id, "vehicle_id")
    return f"recording={recording}, vehicle={vehicle}"


def _optional_trajectory_identity(
    recording_id: str | int | None, vehicle_id: str | int | None
) -> str:
    if recording_id is None and vehicle_id is None:
        return ""
    if recording_id is None or vehicle_id is None:
        raise ValueError("recording_id and vehicle_id must be provided together")
    return f" | {_trajectory_identity(recording_id, vehicle_id)}"


def _safe_identifier(value: object, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"{name} must be a string or integer")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")
    if not safe:
        raise ValueError(f"{name} must contain at least one filename-safe character")
    return safe


def _prepare_output_directory(output_dir: str | Path) -> Path:
    directory = Path(output_dir)
    if directory.exists() and not directory.is_dir():
        raise ValueError("output_dir must be a directory path")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _prepare_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if not path.name or not path.suffix:
        raise ValueError("output_path must include a filename with an extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
