"""D1 contracts for the D2 federated-interface review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol


ModelState = Mapping[str, Any]


@dataclass(frozen=True)
class ClientUpdate:
    """A single client result used for sample-count-weighted aggregation."""

    client_id: str
    state: ModelState
    sample_count: int
    stats: Mapping[str, float]


class LocalTrainer(Protocol):
    def local_train(self, global_state: ModelState) -> ClientUpdate:
        """Train only on local data and return a non-empty update."""
        ...


class Aggregator(Protocol):
    def aggregate(self, updates: list[ClientUpdate]) -> ModelState:
        """Aggregate accepted client updates; implementation must validate counts."""
        ...
