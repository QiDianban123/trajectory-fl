"""PyTorch implementation of the shared trajectory Trainer contract."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from pickle import UnpicklingError
from tempfile import NamedTemporaryFile
from typing import Any

from src.models.base import (
    ModelContract,
    ModelContractError,
    TrajectoryPredictor,
    require_torch,
    validate_checkpoint_payload,
    validate_prediction_tensor,
)
from src.training.trainer import (
    EpochStats,
    EvaluationResult,
    FitResult,
    TrajectoryBatch,
)


@dataclass(frozen=True)
class TorchTrainerConfig:
    """Training settings required to reproduce one local optimization run."""

    epochs: int
    learning_rate: float
    gradient_clip_norm: float
    seed: int
    split_id: str
    device: str = "cpu"
    loss: str = "mse"
    optimizer: str = "adam"

    def __post_init__(self) -> None:
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int) or self.epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        for name in ("learning_rate", "gradient_clip_norm"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.split_id, str) or not self.split_id.strip():
            raise ValueError("split_id must be a non-empty string")
        if self.device not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda, or auto")
        if self.loss != "mse":
            raise ValueError("TorchTrainer supports the frozen mse loss")
        if self.optimizer != "adam":
            raise ValueError("TorchTrainer supports the frozen adam optimizer")

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, object],
        *,
        seed: int,
        split_id: str,
        device: str = "cpu",
    ) -> "TorchTrainerConfig":
        """Build settings from the validated top-level model configuration."""

        training = config.get("training")
        if not isinstance(training, Mapping):
            raise ValueError("model config training must be a mapping")
        required = (
            "epochs",
            "learning_rate",
            "gradient_clip_norm",
            "loss",
            "optimizer",
        )
        missing = [key for key in required if key not in training]
        if missing:
            raise ValueError(f"training config is missing keys: {', '.join(missing)}")
        return cls(
            epochs=training["epochs"],  # type: ignore[arg-type]
            learning_rate=training["learning_rate"],  # type: ignore[arg-type]
            gradient_clip_norm=training["gradient_clip_norm"],  # type: ignore[arg-type]
            seed=seed,
            split_id=split_id,
            device=device,
            loss=training["loss"],  # type: ignore[arg-type]
            optimizer=training["optimizer"],  # type: ignore[arg-type]
        )


class TorchTrainer:
    """Own optimizer, device transfer, gradient clipping, and train/eval modes."""

    def __init__(self, contract: ModelContract, config: TorchTrainerConfig) -> None:
        torch = require_torch()
        self.contract = contract
        self.config = config
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

    def fit(
        self,
        model: TrajectoryPredictor,
        train_batches: Iterable[TrajectoryBatch],
        validation_batches: Iterable[TrajectoryBatch] | None,
        *,
        initial_state: Mapping[str, Any],
    ) -> FitResult:
        """Train from an isolated state and restore the best epoch into ``model``."""

        torch = require_torch()
        module = self._prepare_model(model)
        self._load_state(module, initial_state, source="initial_state")
        reusable_train = _make_reusable(train_batches)
        reusable_validation = (
            _make_reusable(validation_batches) if validation_batches is not None else None
        )
        optimizer = torch.optim.Adam(module.parameters(), lr=float(self.config.learning_rate))
        criterion = torch.nn.MSELoss()
        statistics: list[EpochStats] = []
        best_epoch = -1
        best_score = float("inf")
        best_state: dict[str, Any] | None = None

        for epoch in range(self.config.epochs):
            module.train()
            train_count = 0
            weighted_train_loss = 0.0
            for raw_batch in reusable_train:
                batch = self._move_batch(raw_batch)
                optimizer.zero_grad(set_to_none=True)
                prediction = module(batch.history)
                validate_prediction_tensor(prediction, batch.history, self.contract)
                loss = criterion(prediction, batch.future)
                if not torch.isfinite(loss).item():
                    raise ValueError("training loss must be finite")
                loss.backward()
                parameters = [
                    parameter for parameter in module.parameters() if parameter.grad is not None
                ]
                if not parameters:
                    raise ValueError("training produced no parameter gradients")
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    max_norm=float(self.config.gradient_clip_norm),
                    error_if_nonfinite=True,
                )
                optimizer.step()
                batch_count = int(batch.history.shape[0])
                train_count += batch_count
                weighted_train_loss += float(loss.detach().item()) * batch_count

            if train_count == 0:
                raise ValueError("train_batches must contain at least one sample")
            train_loss = weighted_train_loss / train_count
            validation_loss = None
            selection_loss = train_loss
            if reusable_validation is not None:
                evaluation = self._evaluate(module, reusable_validation)
                validation_loss = evaluation.loss
                selection_loss = validation_loss
            stat = EpochStats(
                epoch=epoch,
                sample_count=train_count,
                train_loss=train_loss,
                validation_loss=validation_loss,
            )
            statistics.append(stat)
            if selection_loss < best_score:
                best_score = selection_loss
                best_epoch = epoch
                best_state = _clone_state_dict(module.state_dict())

        if best_state is None:  # Defensive; the non-empty epoch loop always selects once.
            raise RuntimeError("training did not produce a checkpoint state")
        module.load_state_dict(best_state, strict=True)
        module.eval()
        best_stat = statistics[best_epoch]
        metrics: dict[str, float] = {"train_loss": best_stat.train_loss, "loss": best_score}
        if best_stat.validation_loss is not None:
            metrics["validation_loss"] = best_stat.validation_loss
        payload: dict[str, object] = {
            "schema_version": 1,
            "model_state": _clone_state_dict(best_state),
            "model_config": self._model_config(module),
            "seed": self.config.seed,
            "epoch": best_epoch,
            "split_id": self.config.split_id,
            "metrics": metrics,
        }
        validate_checkpoint_payload(payload)
        return FitResult(
            epoch_stats=tuple(statistics),
            best_epoch=best_epoch,
            checkpoint_payload=payload,
        )

    def evaluate(
        self, model: TrajectoryPredictor, batches: Iterable[TrajectoryBatch]
    ) -> EvaluationResult:
        """Evaluate under ``no_grad`` without changing the caller's model mode."""

        module = self._prepare_model(model)
        return self._evaluate(module, batches)

    def save_checkpoint(self, payload: Mapping[str, object], path: str | Path) -> Path:
        """Atomically persist a validated checkpoint envelope."""

        torch = require_torch()
        validate_checkpoint_payload(payload)
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with NamedTemporaryFile(
                prefix=f".{destination.name}.",
                suffix=".tmp",
                dir=destination.parent,
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            torch.save(dict(payload), temporary_path)
            temporary_path.replace(destination)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        return destination

    def load_checkpoint(
        self, model: TrajectoryPredictor, path: str | Path
    ) -> dict[str, object]:
        """Load, validate, and restore a checkpoint into a compatible model."""

        torch = require_torch()
        source = Path(path)
        if not source.is_file():
            raise ValueError(f"checkpoint does not exist: {source}")
        try:
            loaded = torch.load(source, map_location=self.device, weights_only=True)
        except (OSError, RuntimeError, EOFError, UnpicklingError) as exc:
            raise ValueError(f"cannot load checkpoint {source}: {exc}") from exc
        if not isinstance(loaded, Mapping):
            raise ValueError("checkpoint root must be a mapping")
        payload = dict(loaded)
        validate_checkpoint_payload(payload)
        if payload["split_id"] != self.config.split_id:
            raise ValueError("checkpoint split_id does not match the trainer split_id")
        module = self._prepare_model(model)
        if dict(payload["model_config"]) != self._model_config(module):  # type: ignore[arg-type]
            raise ValueError("checkpoint model_config does not match the target model")
        self._load_state(module, payload["model_state"], source="checkpoint.model_state")  # type: ignore[arg-type]
        module.eval()
        return payload

    def _prepare_model(self, model: TrajectoryPredictor) -> Any:
        torch = require_torch()
        if not isinstance(model, torch.nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if getattr(model, "contract", None) != self.contract:
            raise ModelContractError("model contract does not match the trainer contract")
        return model.to(self.device)

    def _move_batch(self, batch: TrajectoryBatch) -> TrajectoryBatch:
        if not isinstance(batch, TrajectoryBatch):
            raise TypeError("batches must contain TrajectoryBatch values")
        batch.validate(self.contract)
        if batch.meta:
            split_ids = {item.get("split_id") for item in batch.meta}
            if split_ids != {self.config.split_id}:
                raise ValueError("batch metadata split_id does not match the trainer split_id")
        moved = TrajectoryBatch(
            history=batch.history.to(self.device),
            future=batch.future.to(self.device),
            meta=batch.meta,
        )
        moved.validate(self.contract)
        return moved

    def _evaluate(
        self, module: Any, batches: Iterable[TrajectoryBatch]
    ) -> EvaluationResult:
        torch = require_torch()
        criterion = torch.nn.MSELoss()
        was_training = module.training
        sample_count = 0
        weighted_loss = 0.0
        module.eval()
        try:
            with torch.no_grad():
                for raw_batch in batches:
                    batch = self._move_batch(raw_batch)
                    prediction = module(batch.history)
                    validate_prediction_tensor(prediction, batch.history, self.contract)
                    loss = criterion(prediction, batch.future)
                    if not torch.isfinite(loss).item():
                        raise ValueError("evaluation loss must be finite")
                    batch_count = int(batch.history.shape[0])
                    sample_count += batch_count
                    weighted_loss += float(loss.item()) * batch_count
        finally:
            module.train(was_training)
        if sample_count == 0:
            raise ValueError("evaluation batches must contain at least one sample")
        return EvaluationResult(sample_count=sample_count, loss=weighted_loss / sample_count)

    def _load_state(
        self, module: Any, state: Mapping[str, Any], *, source: str
    ) -> None:
        torch = require_torch()
        if not isinstance(state, Mapping):
            raise TypeError(f"{source} must be a state_dict mapping")
        expected = module.state_dict()
        if set(state) != set(expected):
            missing = sorted(set(expected) - set(state))
            unexpected = sorted(set(state) - set(expected))
            raise ValueError(
                f"{source} keys do not match model; missing={missing}, unexpected={unexpected}"
            )
        copied: dict[str, Any] = {}
        for key, expected_tensor in expected.items():
            value = state[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{source}[{key!r}] must be a torch.Tensor")
            if value.shape != expected_tensor.shape:
                raise ValueError(f"{source}[{key!r}] shape does not match model")
            if value.dtype != expected_tensor.dtype:
                raise ValueError(f"{source}[{key!r}] dtype does not match model")
            numeric = value.is_floating_point() or value.is_complex()
            if numeric and not torch.isfinite(value).all():
                raise ValueError(f"{source}[{key!r}] must contain only finite values")
            copied[key] = value.detach().clone()
        try:
            module.load_state_dict(copied, strict=True)
        except RuntimeError as exc:
            raise ValueError(f"cannot restore {source}: {exc}") from exc

    @staticmethod
    def _model_config(module: Any) -> dict[str, object]:
        exporter = getattr(module, "to_config", None)
        if not callable(exporter):
            raise TypeError("model must expose to_config() for checkpoint reproducibility")
        config = exporter()
        if not isinstance(config, Mapping):
            raise TypeError("model.to_config() must return a mapping")
        return deepcopy(dict(config))


def _make_reusable(batches: Iterable[TrajectoryBatch]) -> Iterable[TrajectoryBatch]:
    iterator = iter(batches)
    if isinstance(batches, Iterator) or iterator is batches:
        return tuple(iterator)
    return batches


def _clone_state_dict(state: Mapping[str, Any]) -> dict[str, Any]:
    torch = require_torch()
    cloned: dict[str, Any] = {}
    for key, value in state.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"state_dict[{key!r}] must be a torch.Tensor")
        cloned[key] = value.detach().cpu().clone()
    return cloned
