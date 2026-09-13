"""Federated-learning contracts; numerical implementation starts on Day 8."""

from src.federated.aggregation import AggregationRequest, Aggregator
from src.federated.client import ClientTrainRequest, FederatedClient
from src.federated.contracts import ClientFailure, ClientSelection, ClientUpdate
from src.federated.training_adapter import (
    CLIENT_TRAINING_STAT_KEYS,
    LocalTrainerAdapter,
    ModelStateSnapshot,
    clone_model_state,
    fit_result_to_client_update,
    fit_result_to_last_epoch_client_update,
    model_state_id,
    snapshot_model_state,
)

__all__ = [
    "AggregationRequest",
    "Aggregator",
    "ClientFailure",
    "ClientSelection",
    "ClientTrainRequest",
    "ClientUpdate",
    "CLIENT_TRAINING_STAT_KEYS",
    "FederatedClient",
    "LocalTrainerAdapter",
    "ModelStateSnapshot",
    "clone_model_state",
    "fit_result_to_client_update",
    "fit_result_to_last_epoch_client_update",
    "model_state_id",
    "snapshot_model_state",
]
