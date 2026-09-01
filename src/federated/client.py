"""Federated client request and execution interface without local-training code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.federated.contracts import ClientResult, FederatedContractError, ModelState


@dataclass(frozen=True)
class ClientTrainRequest:
    """One immutable global-state dispatch to one selected client."""

    round_index: int
    global_state_id: str
    global_state: ModelState

    def __post_init__(self) -> None:
        if isinstance(self.round_index, bool) or not isinstance(self.round_index, int):
            raise FederatedContractError("round_index must be a non-negative integer")
        if self.round_index < 0:
            raise FederatedContractError("round_index must be a non-negative integer")
        if not isinstance(self.global_state_id, str) or not self.global_state_id.strip():
            raise FederatedContractError("global_state_id must be a non-empty string")
        if not self.global_state:
            raise FederatedContractError("global_state must be non-empty")


class FederatedClient(Protocol):
    """Client boundary; implementations must delegate local optimization to Trainer."""

    client_id: str

    def local_train(self, request: ClientTrainRequest) -> ClientResult:
        """Return exactly one successful update or explicit failure record."""
        ...
