"""Centralized evaluation from saved predictions, scaler, and loss logs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Protocol

import numpy as np

from src.evaluation.metrics import compute_metrics
from src.evaluation.result_store import ResultRecord, write_csv, write_json
from src.evaluation.visualization import plot_loss_curve, plot_prediction_trajectory


class InverseTransformScaler(Protocol):
    """Small scaler boundary required by the evaluation layer."""

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Recover physical coordinates in meters."""
        ...


@dataclass(frozen=True)
class LossHistory:
    """Finite per-epoch losses used to rebuild the centralized loss figure."""

    train: tuple[float, ...]
    validation: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        _validate_loss_values(self.train, "train")
        if self.validation is not None:
            _validate_loss_values(self.validation, "validation")
            if len(self.validation) != len(self.train):
                raise ValueError("validation loss history must match train loss history length")


@dataclass(frozen=True)
class PhysicalTrajectoryBatch:
    """Prediction and truth after one train-fitted scaler restores meter units."""

    prediction: np.ndarray
    truth: np.ndarray

    @property
    def sample_count(self) -> int:
        return int(self.prediction.shape[0])


@dataclass(frozen=True)
class PhysicalEvaluation:
    """Physical arrays and canonical metrics produced by the shared evaluation path."""

    trajectories: PhysicalTrajectoryBatch
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class CentralizedEvaluationRequest:
    """Saved centralized outputs and provenance required for physical evaluation."""

    prediction: np.ndarray
    truth: np.ndarray
    scaler: InverseTransformScaler
    loss_history: LossHistory
    run_id: str
    code_sha: str
    seed: int
    split_id: str
    total_seconds: float
    output_dir: str | Path
    dataset: str = "highd"
    model: str = "lstm_encoder_decoder"
    artifact_paths: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class CentralizedEvaluationOutput:
    """Canonical record and every artifact created from one evaluation request."""

    record: ResultRecord
    json_path: Path
    csv_path: Path
    loss_curve_path: Path
    trajectory_path: Path


def inverse_transform_batch(
    prediction: np.ndarray,
    truth: np.ndarray,
    scaler: InverseTransformScaler,
) -> PhysicalTrajectoryBatch:
    """Restore matching ``[B,T,2]`` arrays before any physical metric is computed."""

    normalized_prediction = _validate_batch(prediction, "prediction")
    normalized_truth = _validate_batch(truth, "truth")
    if normalized_prediction.shape != normalized_truth.shape:
        raise ValueError(
            "prediction shape "
            f"{normalized_prediction.shape} does not match truth shape {normalized_truth.shape}"
        )
    inverse_transform = getattr(scaler, "inverse_transform", None)
    if not callable(inverse_transform):
        raise TypeError("scaler must expose a callable inverse_transform")

    physical_prediction = _validate_restored(
        inverse_transform(normalized_prediction.copy()),
        normalized_prediction.shape,
        "prediction",
    )
    physical_truth = _validate_restored(
        inverse_transform(normalized_truth.copy()),
        normalized_truth.shape,
        "truth",
    )
    return PhysicalTrajectoryBatch(prediction=physical_prediction, truth=physical_truth)


def evaluate_prediction_arrays(
    prediction: np.ndarray,
    truth: np.ndarray,
    scaler: InverseTransformScaler,
) -> PhysicalEvaluation:
    """Evaluate normalized arrays through the single meter-coordinate metric path."""

    trajectories = inverse_transform_batch(prediction, truth, scaler)
    return PhysicalEvaluation(
        trajectories=trajectories,
        metrics=compute_metrics(trajectories.prediction, trajectories.truth),
    )


def evaluate_centralized(request: CentralizedEvaluationRequest) -> CentralizedEvaluationOutput:
    """Compute meter metrics and write plots plus same-source JSON/CSV artifacts.

    This adapter deliberately has no model or Trainer input. Callers may rebuild all
    artifacts by loading saved prediction/truth arrays and loss logs, then invoking it.
    """

    if not isinstance(request, CentralizedEvaluationRequest):
        raise TypeError("request must be a CentralizedEvaluationRequest")
    if not isinstance(request.loss_history, LossHistory):
        raise TypeError("loss_history must be a LossHistory")
    evaluation = evaluate_prediction_arrays(request.prediction, request.truth, request.scaler)
    physical = evaluation.trajectories
    metrics = evaluation.metrics

    reserved_artifacts = {
        "loss_curve": "figures/loss_curve.png",
        "trajectory": "figures/prediction_trajectory.png",
    }
    overlap = set(request.artifact_paths) & set(reserved_artifacts)
    if overlap:
        raise ValueError(f"artifact_paths cannot override reserved artifacts: {sorted(overlap)}")

    record = ResultRecord(
        run_id=request.run_id,
        code_sha=request.code_sha,
        seed=request.seed,
        split_id=request.split_id,
        mode="centralized",
        sample_count=physical.sample_count,
        ade=metrics["ade"],
        fde=metrics["fde"],
        total_seconds=request.total_seconds,
        artifact_paths={**dict(request.artifact_paths), **reserved_artifacts},
        dataset=request.dataset,
        model=request.model,
    )
    output_dir = Path(request.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    loss_curve_path = plot_loss_curve(
        request.loss_history.train,
        figures_dir / "loss_curve.png",
        validation_losses=request.loss_history.validation,
    )
    trajectory_path = plot_prediction_trajectory(
        physical.truth[0],
        physical.prediction[0],
        figures_dir / "prediction_trajectory.png",
    )
    json_path = write_json(record, output_dir / "metrics.json")
    csv_path = write_csv([record], output_dir / "metrics.csv")
    artifacts = (json_path, csv_path, loss_curve_path, trajectory_path)
    if not all(path.is_file() and path.stat().st_size > 0 for path in artifacts):
        raise RuntimeError("centralized evaluation did not create every declared artifact")
    return CentralizedEvaluationOutput(
        record=record,
        json_path=json_path,
        csv_path=csv_path,
        loss_curve_path=loss_curve_path,
        trajectory_path=trajectory_path,
    )


def _validate_batch(values: object, name: str) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray")
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[1] == 0 or values.shape[2] != 2:
        raise ValueError(f"{name} must have non-empty shape [B, T, 2]")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"{name} must use a floating-point dtype")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    return values


def _validate_restored(values: object, expected_shape: tuple[int, ...], name: str) -> np.ndarray:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"scaler inverse_transform for {name} must return a numpy.ndarray")
    if values.shape != expected_shape:
        raise ValueError(f"scaler inverse_transform changed {name} shape")
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError(f"restored {name} must use a floating-point dtype")
    if not np.isfinite(values).all():
        raise ValueError(f"restored {name} must contain only finite values")
    return values


def _validate_loss_values(values: Sequence[float], name: str) -> None:
    if not values:
        raise ValueError(f"{name} loss history cannot be empty")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError(f"{name} loss history must contain finite numbers")
        if value < 0:
            raise ValueError(f"{name} loss history must be non-negative")
