"""End-to-end centralized experiment orchestration."""

from __future__ import annotations

import json
import logging
import platform
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from src.data.loading import (
    DataLoaderConfig,
    ProcessedDataBundle,
    ProcessedDatasetReader,
    create_dataloaders,
)
from src.evaluation.centralized import (
    CentralizedEvaluationRequest,
    LossHistory,
    evaluate_centralized,
    evaluate_prediction_arrays,
)
from src.evaluation.prediction import PredictionCollection, collect_predictions
from src.evaluation.result_store import ResultRecord, write_csv, write_json
from src.evaluation.visualization import plot_prediction_trajectory
from src.experiments.run_context import RunContext
from src.federated.training_adapter import clone_model_state
from src.models.base import ModelContract, require_torch
from src.models.constant_velocity import ConstantVelocityBaseline
from src.models.initialization import initialize_model
from src.models.lstm_seq2seq import LSTMSeq2Seq
from src.training.centralized import CentralizedTrainingRequest, run_centralized
from src.training.torch_trainer import TorchTrainer, TorchTrainerConfig
from src.training.trainer import FitResult
from src.utils.logging import configure_run_logger
from src.utils.seed import set_global_seed

LoaderFactory = Callable[..., Mapping[str, Any]]
ModelFactory = Callable[[Mapping[str, object]], Any]
TrainerFactory = Callable[[ModelContract, TorchTrainerConfig], Any]


@dataclass(frozen=True)
class CentralizedExperimentRequest:
    """Validated configuration and processed-data identity for one unique run."""

    config_bundle: Mapping[str, Mapping[str, Any]]
    processed_dir: str | Path
    project_root: str | Path
    run_id: str
    output_root: str | Path = "outputs"
    code_sha: str | None = None
    expected_data_version: str | None = None
    expected_split_id: str | None = None
    resume_checkpoint: str | Path | None = None
    initial_state: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class CentralizedExperimentOutput:
    """Completed result records and reproducible artifact locations."""

    record: ResultRecord
    baseline_record: ResultRecord
    loss: float
    best_epoch: int
    output_dir: Path
    checkpoint_path: Path
    metadata_path: Path
    manifest_path: Path


class CentralizedExperiment:
    """Compose existing data, model, Trainer, evaluation, and run-context APIs."""

    def __init__(
        self,
        *,
        reader: ProcessedDatasetReader | None = None,
        loader_factory: LoaderFactory = create_dataloaders,
        model_factory: ModelFactory = LSTMSeq2Seq.from_model_config,
        trainer_factory: TrainerFactory = TorchTrainer,
    ) -> None:
        self._reader = reader if reader is not None else ProcessedDatasetReader()
        self._loader_factory = loader_factory
        self._model_factory = model_factory
        self._trainer_factory = trainer_factory

    def run(self, request: CentralizedExperimentRequest) -> CentralizedExperimentOutput:
        """Run one centralized experiment and preserve a failed run for diagnosis."""

        bundle = _validated_bundle(request.config_bundle)
        run_config = bundle["experiment"]["run"]
        seed = _non_negative_int(run_config["seed"], "experiment.run.seed")
        context = RunContext(
            bundle,
            request.project_root,
            request.run_id,
            code_sha=request.code_sha,
            output_root=request.output_root,
        )
        started = perf_counter()
        metadata_path = context.output_dir / "metadata.json"
        log_path = context.output_dir / "train.log"
        logger = configure_run_logger(context.run_id, log_path)
        context.add_artifact("metadata", metadata_path.name)
        context.add_artifact("log", log_path.name)
        _write_json(
            metadata_path,
            _metadata(context, seed=seed, status="running", data=None, error=None),
        )
        logger.info("centralized experiment started")

        data: ProcessedDataBundle | None = None
        sample_count = 0
        try:
            data = self._reader.load(
                request.processed_dir,
                data_config=bundle["data"],
                expected_data_version=request.expected_data_version,
                expected_split_id=request.expected_split_id,
            )
            context.set_data_identity(data_version=data.data_version, split_id=data.split_id)
            _record_data(context, data)

            model_config = bundle["model"]["model"]
            contract = ModelContract.from_model_config(model_config)
            loader_config = DataLoaderConfig.from_config_bundle(bundle)
            loaders = self._loader_factory(data, contract=contract, config=loader_config)
            _require_loader_splits(loaders)

            set_global_seed(seed)
            model = self._model_factory(model_config)
            trainer_config = TorchTrainerConfig.from_config(
                bundle["model"],
                seed=seed,
                split_id=data.split_id,
                device=bundle["experiment"]["execution"]["device"],
            )
            trainer = self._trainer_factory(contract, trainer_config)
            if request.resume_checkpoint is not None:
                loader = getattr(trainer, "load_checkpoint", None)
                if not callable(loader):
                    raise TypeError("trainer must expose load_checkpoint for resume")
                loader(model, request.resume_checkpoint)
            elif request.initial_state is not None:
                model.load_state_dict(clone_model_state(request.initial_state), strict=True)
            else:
                initialize_model(model, bundle["model"]["training"]["initialization"])
            fit_result = run_centralized(
                trainer,
                CentralizedTrainingRequest(
                    model=model,
                    train_batches=loaders["train"],
                    validation_batches=loaders["validation"],
                    initial_state=clone_model_state(model.state_dict()),
                ),
            )
            logger.info("centralized training completed")

            checkpoint_path = context.output_dir / "checkpoints" / "best.pt"
            saver = getattr(trainer, "save_checkpoint", None)
            if not callable(saver):
                raise TypeError("trainer must expose save_checkpoint for experiment persistence")
            saver(fit_result.checkpoint_payload, checkpoint_path)
            history_path = context.output_dir / "training_history.json"
            _write_json(
                history_path,
                {
                    "best_epoch": fit_result.best_epoch,
                    "epochs": [
                        {
                            "epoch": stat.epoch,
                            "sample_count": stat.sample_count,
                            "train_loss": stat.train_loss,
                            "validation_loss": stat.validation_loss,
                        }
                        for stat in fit_result.epoch_stats
                    ],
                },
            )

            evaluation = trainer.evaluate(model, loaders["test"])
            prediction_before_restore = collect_predictions(
                model,
                loaders["test"],
                contract=contract,
                device=getattr(trainer, "device", trainer_config.device),
            )
            checkpoint_loader = getattr(trainer, "load_checkpoint", None)
            if not callable(checkpoint_loader):
                raise TypeError("trainer must expose load_checkpoint for round-trip verification")
            checkpoint_loader(model, checkpoint_path)
            prediction = collect_predictions(
                model,
                loaders["test"],
                contract=contract,
                device=getattr(trainer, "device", trainer_config.device),
            )
            if not np.array_equal(prediction_before_restore.prediction, prediction.prediction):
                raise RuntimeError("checkpoint predictions differ from pre-save predictions")
            sample_count = prediction.sample_count
            baseline = ConstantVelocityBaseline(contract)
            torch = require_torch()
            baseline_prediction = baseline(
                torch.from_numpy(prediction.history.copy())
            ).detach().cpu().numpy()
            predictions_path = context.output_dir / "predictions.npz"
            _save_predictions(predictions_path, prediction, baseline_prediction)
            baseline_physical = evaluate_prediction_arrays(
                baseline_prediction,
                prediction.truth,
                data.scaler,
            )
            baseline_trajectory_path = plot_prediction_trajectory(
                baseline_physical.trajectories.truth[0],
                baseline_physical.trajectories.prediction[0],
                context.output_dir / "figures" / "baseline_trajectory.png",
            )
            elapsed = perf_counter() - started
            baseline_record = ResultRecord(
                run_id=context.run_id,
                code_sha=context.code_sha,
                seed=seed,
                split_id=data.split_id,
                mode="centralized",
                sample_count=baseline_physical.trajectories.sample_count,
                ade=baseline_physical.metrics["ade"],
                fde=baseline_physical.metrics["fde"],
                total_seconds=elapsed,
                artifact_paths={
                    "predictions": predictions_path.name,
                    "trajectory": "figures/baseline_trajectory.png",
                },
                model="constant_velocity",
            )
            baseline_json = write_json(
                baseline_record, context.output_dir / "baseline" / "metrics.json"
            )
            baseline_csv = write_csv(
                [baseline_record], context.output_dir / "baseline" / "metrics.csv"
            )

            artifact_paths = {
                "checkpoint": "checkpoints/best.pt",
                "training_history": history_path.name,
                "predictions": predictions_path.name,
                "metadata": metadata_path.name,
                "log": log_path.name,
                "baseline_json": "baseline/metrics.json",
                "baseline_csv": "baseline/metrics.csv",
                "baseline_trajectory": "figures/baseline_trajectory.png",
            }
            evaluated = evaluate_centralized(
                CentralizedEvaluationRequest(
                    prediction=prediction.prediction,
                    truth=prediction.truth,
                    scaler=data.scaler,
                    loss_history=_loss_history(fit_result),
                    run_id=context.run_id,
                    code_sha=context.code_sha,
                    seed=seed,
                    split_id=data.split_id,
                    total_seconds=elapsed,
                    output_dir=context.output_dir,
                    artifact_paths=artifact_paths,
                )
            )
            for name, path in {
                **artifact_paths,
                "metrics_json": evaluated.json_path.relative_to(context.output_dir),
                "metrics_csv": evaluated.csv_path.relative_to(context.output_dir),
                "loss_curve": evaluated.loss_curve_path.relative_to(context.output_dir),
                "trajectory": evaluated.trajectory_path.relative_to(context.output_dir),
            }.items():
                context.add_artifact(name, path)
            _require_files(
                checkpoint_path,
                history_path,
                predictions_path,
                baseline_json,
                baseline_csv,
                baseline_trajectory_path,
                evaluated.json_path,
                evaluated.csv_path,
                evaluated.loss_curve_path,
                evaluated.trajectory_path,
            )
            _write_json(
                metadata_path,
                _metadata(context, seed=seed, status="completed", data=data, error=None),
            )
            logger.info("centralized evaluation completed")
            _close_logger(logger)
            manifest_path = context.export_manifest()
            return CentralizedExperimentOutput(
                record=evaluated.record,
                baseline_record=baseline_record,
                loss=float(evaluation.loss),
                best_epoch=fit_result.best_epoch,
                output_dir=context.output_dir,
                checkpoint_path=checkpoint_path,
                metadata_path=metadata_path,
                manifest_path=manifest_path,
            )
        except Exception as exc:
            logger.exception("centralized experiment failed")
            _close_logger(logger)
            self._record_failure(
                context,
                request,
                seed=seed,
                data=data,
                sample_count=sample_count,
                error=exc,
                elapsed=perf_counter() - started,
                metadata_path=metadata_path,
            )
            raise

    @staticmethod
    def _record_failure(
        context: RunContext,
        request: CentralizedExperimentRequest,
        *,
        seed: int,
        data: ProcessedDataBundle | None,
        sample_count: int,
        error: Exception,
        elapsed: float,
        metadata_path: Path,
    ) -> None:
        data_version = data.data_version if data is not None else request.expected_data_version
        split_id = data.split_id if data is not None else request.expected_split_id
        context.set_data_identity(
            data_version=data_version or "unavailable",
            split_id=split_id or "unavailable",
        )
        error_text = f"{type(error).__name__}: {error}"
        record = ResultRecord(
            run_id=context.run_id,
            code_sha=context.code_sha,
            seed=seed,
            split_id=split_id or "unavailable",
            mode="centralized",
            sample_count=sample_count,
            ade=0.0,
            fde=0.0,
            total_seconds=elapsed,
            status="failed",
            error=error_text,
            artifact_paths={"metadata": metadata_path.name, "log": "train.log"},
        )
        metrics_json = write_json(record, context.output_dir / "metrics.json")
        metrics_csv = write_csv([record], context.output_dir / "metrics.csv")
        context.add_artifact("metrics_json", metrics_json.name)
        context.add_artifact("metrics_csv", metrics_csv.name)
        _write_json(
            metadata_path,
            _metadata(context, seed=seed, status="failed", data=data, error=error_text),
        )
        context.export_manifest()


def _validated_bundle(
    bundle: Mapping[str, Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    if not isinstance(bundle, Mapping) or set(bundle) != {"data", "model", "experiment"}:
        raise ValueError("config_bundle must contain data, model, and experiment")
    values = {name: bundle[name] for name in ("data", "model", "experiment")}
    if not all(isinstance(value, Mapping) for value in values.values()):
        raise TypeError("config bundle sections must be mappings")
    return values


def _require_loader_splits(loaders: Mapping[str, Any]) -> None:
    if set(loaders) != {"train", "validation", "test"}:
        raise ValueError("loader factory must return train, validation, and test")


def _loss_history(result: FitResult) -> LossHistory:
    train = tuple(float(stat.train_loss) for stat in result.epoch_stats)
    validation_values = tuple(stat.validation_loss for stat in result.epoch_stats)
    if all(value is None for value in validation_values):
        validation = None
    elif all(value is not None for value in validation_values):
        validation = tuple(float(value) for value in validation_values if value is not None)
    else:
        raise ValueError("validation loss must be present for every epoch or no epoch")
    return LossHistory(train=train, validation=validation)


def _save_predictions(
    path: Path,
    prediction: PredictionCollection,
    baseline_prediction: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        history=prediction.history,
        truth=prediction.truth,
        lstm_prediction=prediction.prediction,
        baseline_prediction=baseline_prediction,
    )


def _record_data(context: RunContext, data: ProcessedDataBundle) -> None:
    for split_name in ("train", "validation", "test"):
        context.add_split_manifest(
            split_name,
            {"sample_count": len(data.datasets[split_name]), "split_id": data.split_id},
        )
    source_manifest = data.source / "split_manifest.json"
    if source_manifest.is_file():
        context.record_data_file(source_manifest)


def _metadata(
    context: RunContext,
    *,
    seed: int,
    status: str,
    data: ProcessedDataBundle | None,
    error: str | None,
) -> dict[str, object]:
    torch = require_torch()
    return {
        "schema_version": 1,
        "run_id": context.run_id,
        "status": status,
        "error": error,
        "code_sha": context.code_sha,
        "seed": seed,
        "data_version": data.data_version if data is not None else None,
        "split_id": data.split_id if data is not None else None,
        "environment": {
            "python": platform.python_version(),
            "platform": sys.platform,
            "torch": torch.__version__,
        },
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_files(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"experiment artifacts are missing or empty: {missing}")


def _close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        if getattr(handler, "_trajectory_fl_run_handler", False):
            handler.flush()
            logger.removeHandler(handler)
            handler.close()


def _non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
