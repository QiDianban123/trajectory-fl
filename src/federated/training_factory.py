"""S3 factories for fresh client models and local-epoch Trainers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.federated.contracts import validate_model_state
from src.federated.training_adapter import clone_model_state
from src.models.lstm_seq2seq import LSTMSeq2Seq
from src.training.torch_trainer import TorchTrainer, TorchTrainerConfig


def create_model_from_config(
    model_config: Mapping[str, object],
) -> LSTMSeq2Seq:
    """Create one fresh model; A/G own the single shared baseline initialization."""

    return LSTMSeq2Seq.from_model_config(model_config)


def build_isolated_client_trainer(
    client_id: str,
    *,
    model_config: Mapping[str, object],
    seed: int,
    split_id: str,
    local_epochs: int,
) -> TorchTrainer:
    """Return a fresh shared Trainer implementation configured for one client."""

    if not isinstance(client_id, str) or not client_id.strip():
        raise ValueError("client_id must be a non-empty string")
    if isinstance(local_epochs, bool) or not isinstance(local_epochs, int) or local_epochs <= 0:
        raise ValueError("local_epochs must be a positive integer")
    config = dict(model_config)
    training = dict(config["training"])
    training["epochs"] = local_epochs
    config["training"] = training
    return TorchTrainer(
        LSTMSeq2Seq.from_model_config(config["model"]).contract,
        TorchTrainerConfig.from_config(config, seed=seed, split_id=split_id),
    )


def load_isolated_state(model: LSTMSeq2Seq, state: Mapping[str, Any]) -> None:
    """Load a clone so one client can never retain caller or peer tensor storage."""

    validate_model_state(state, model.state_dict())
    model.load_state_dict(clone_model_state(state), strict=True)
