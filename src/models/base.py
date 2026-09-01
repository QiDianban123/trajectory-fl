"""Model contracts frozen on Day 2; concrete networks start on Day 5."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias

try:
    import torch
    from torch import nn
except ImportError:  # Keep interface inspection useful before optional dependencies are installed.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]


TrajectoryTensor: TypeAlias = Any
CHECKPOINT_REQUIRED_KEYS = (
    "schema_version",
    "model_state",
    "model_config",
    "seed",
    "epoch",
    "split_id",
    "metrics",
)


class ModelContractError(ValueError):
    """A tensor, model configuration, or checkpoint violates the P0 contract."""


def require_torch() -> Any:
    """Return PyTorch or raise a readable error at the first runtime use."""

    if torch is None:
        raise RuntimeError(
            "PyTorch is required for model and training operations. "
            "Install the project dependencies with `python -m pip install -r requirements.txt`."
        )
    return torch


@dataclass(frozen=True)
class ModelContract:
    """Shape, coordinate, dtype, and device rules shared by all P0 predictors."""

    history_steps: int
    future_steps: int
    input_size: int = 2
    output_size: int = 2
    dtype: str = "float32"
    coordinate_representation: str = "absolute_position"
    device_policy: str = "trainer_managed"

    def __post_init__(self) -> None:
        for name in ("history_steps", "future_steps", "input_size", "output_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ModelContractError(f"{name} must be a positive integer")
        if self.input_size != 2 or self.output_size != 2:
            raise ModelContractError("P0 trajectory predictors require two-dimensional coordinates")
        if self.dtype != "float32":
            raise ModelContractError("P0 trajectory predictors require float32 tensors")
        if self.coordinate_representation != "absolute_position":
            raise ModelContractError("P0 predicts normalized absolute_position coordinates")
        if self.device_policy != "trainer_managed":
            raise ModelContractError("device_policy must be trainer_managed")

    @classmethod
    def from_model_config(cls, config: Mapping[str, object]) -> "ModelContract":
        """Build the contract from the validated ``model`` config section."""

        required = (
            "history_steps",
            "future_steps",
            "input_size",
            "output_size",
            "dtype",
            "coordinate_representation",
            "device_policy",
        )
        missing = [key for key in required if key not in config]
        if missing:
            raise ModelContractError(f"model config is missing keys: {', '.join(missing)}")
        return cls(
            history_steps=config["history_steps"],  # type: ignore[arg-type]
            future_steps=config["future_steps"],  # type: ignore[arg-type]
            input_size=config["input_size"],  # type: ignore[arg-type]
            output_size=config["output_size"],  # type: ignore[arg-type]
            dtype=config["dtype"],  # type: ignore[arg-type]
            coordinate_representation=config["coordinate_representation"],  # type: ignore[arg-type]
            device_policy=config["device_policy"],  # type: ignore[arg-type]
        )


class TrajectoryPredictor(Protocol):
    """Predict normalized future coordinates from normalized history coordinates."""

    contract: ModelContract

    def forward(self, history: TrajectoryTensor) -> TrajectoryTensor:
        """Return a finite float32 tensor with shape ``[B, T_f, 2]``."""
        ...


def validate_history_tensor(history: TrajectoryTensor, contract: ModelContract) -> None:
    """Validate a batch before a predictor receives it."""

    _validate_tensor(
        history,
        expected_shape=(None, contract.history_steps, contract.input_size),
        name="history",
    )


def validate_future_tensor(future: TrajectoryTensor, contract: ModelContract) -> None:
    """Validate normalized future labels before loss or evaluation."""

    _validate_tensor(
        future,
        expected_shape=(None, contract.future_steps, contract.output_size),
        name="future",
    )


def validate_prediction_tensor(
    prediction: TrajectoryTensor, history: TrajectoryTensor, contract: ModelContract
) -> None:
    """Validate output shape, dtype, finite values, and caller-owned device."""

    _validate_tensor(
        prediction,
        expected_shape=(history.shape[0], contract.future_steps, contract.output_size),
        name="prediction",
    )
    if prediction.device != history.device:
        raise ModelContractError("prediction must remain on the same device as history")


def validate_checkpoint_payload(payload: Mapping[str, object]) -> None:
    """Validate the metadata envelope around a future ``state_dict`` checkpoint."""

    missing = [key for key in CHECKPOINT_REQUIRED_KEYS if key not in payload]
    if missing:
        raise ModelContractError(f"checkpoint is missing keys: {', '.join(missing)}")
    if payload["schema_version"] != 1:
        raise ModelContractError("checkpoint.schema_version must be 1")
    if not isinstance(payload["model_state"], Mapping):
        raise ModelContractError("checkpoint.model_state must be a state_dict mapping")
    if not isinstance(payload["model_config"], Mapping):
        raise ModelContractError("checkpoint.model_config must be a mapping")
    for key in ("seed", "epoch"):
        value = payload[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ModelContractError(f"checkpoint.{key} must be a non-negative integer")
    if not isinstance(payload["split_id"], str) or not payload["split_id"].strip():
        raise ModelContractError("checkpoint.split_id must be a non-empty string")
    if not isinstance(payload["metrics"], Mapping):
        raise ModelContractError("checkpoint.metrics must be a mapping")


def _validate_tensor(
    tensor: TrajectoryTensor,
    expected_shape: tuple[int | None, int, int],
    name: str,
) -> None:
    torch_module = require_torch()
    if not isinstance(tensor, torch_module.Tensor):
        raise ModelContractError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 3:
        raise ModelContractError(f"{name} must have shape [B, T, 2]")
    expected_batch, expected_steps, expected_features = expected_shape
    if tensor.shape[0] <= 0 or tensor.shape[1:] != (expected_steps, expected_features):
        raise ModelContractError(
            f"{name} must have shape [B, {expected_steps}, {expected_features}] with B > 0"
        )
    if expected_batch is not None and tensor.shape[0] != expected_batch:
        raise ModelContractError(f"{name} batch size must equal history batch size")
    if tensor.dtype != torch_module.float32:
        raise ModelContractError(f"{name} must use torch.float32")
    if not torch_module.isfinite(tensor).all().item():
        raise ModelContractError(f"{name} must contain only finite values")


if torch is None:

    class BaseTrajectoryModel(ABC):
        """Importable base when PyTorch is absent; runtime use gives a clear error."""

        def __init__(self, contract: ModelContract) -> None:
            require_torch()
            self.contract = contract

        @abstractmethod
        def forward(self, history: TrajectoryTensor) -> TrajectoryTensor:
            raise NotImplementedError

else:

    class BaseTrajectoryModel(nn.Module, ABC):
        """Base for D5 concrete networks; it owns no optimizer or file I/O."""

        def __init__(self, contract: ModelContract) -> None:
            super().__init__()
            self.contract = contract

        def validate_history(self, history: TrajectoryTensor) -> None:
            validate_history_tensor(history, self.contract)

        def validate_prediction(
            self, prediction: TrajectoryTensor, history: TrajectoryTensor
        ) -> None:
            validate_prediction_tensor(prediction, history, self.contract)

        @abstractmethod
        def forward(self, history: TrajectoryTensor) -> TrajectoryTensor:
            """Return normalized future coordinates; implementations validate I/O."""
            raise NotImplementedError
