"""S3 factories for fresh client models and local-epoch Trainers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.federated.training_adapter import clone_model_state
from src.models.initialization import initialize_model
from src.models.lstm_seq2seq import LSTMSeq2Seq
from src.training.torch_trainer import TorchTrainer, TorchTrainerConfig


def create_model_from_config(
    model_config: Mapping[str, object], *, initialization: Mapping[str, Any], seed: int
) -> LSTMSeq2Seq:
    """Create one fresh, reproducibly initialized model without federated state."""

    import torch

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = LSTMSeq2Seq.from_model_config(model_config)
        initialize_model(model, initialization)
    return model


def build_isolated_client_trainer(
    model_config: Mapping[str, object], *, seed: int, split_id: str, local_epochs: int
) -> TorchTrainer:
    """Return a fresh shared Trainer implementation configured for one client."""

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

    model.load_state_dict(clone_model_state(state), strict=True)
