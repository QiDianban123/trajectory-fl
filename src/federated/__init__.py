"""Federated-learning contracts; numerical implementation starts on Day 8."""

from src.federated.aggregation import AggregationRequest, Aggregator
from src.federated.client import ClientTrainRequest, FederatedClient
from src.federated.contracts import ClientFailure, ClientSelection, ClientUpdate

__all__ = [
    "AggregationRequest",
    "Aggregator",
    "ClientFailure",
    "ClientSelection",
    "ClientTrainRequest",
    "ClientUpdate",
    "FederatedClient",
]
