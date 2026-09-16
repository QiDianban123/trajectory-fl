"""S3 Local-only and one-round federated execution over existing client interfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter

from src.federated.aggregation import AggregationRequest, AggregationResult, FedAvgAggregator
from src.federated.client import ClientTrainRequest, FederatedClient
from src.federated.contracts import (
    ClientFailure,
    ClientResult,
    ClientSelection,
    ClientUpdate,
    ModelState,
)
from src.federated.server import partition_client_results, validate_selection_against_available
from src.federated.training_adapter import clone_model_state, model_state_id


@dataclass(frozen=True)
class RoundExecution:
    selection: ClientSelection
    results: tuple[ClientResult, ...]
    aggregation: AggregationResult | None
    elapsed_seconds: float

    def to_manifest(self) -> dict[str, object]:
        """Return UI/G-consumable facts without persisting or recomputing metrics."""

        updates = [item for item in self.results if isinstance(item, ClientUpdate)]
        failures = [item for item in self.results if isinstance(item, ClientFailure)]
        total = sum(item.sample_count for item in updates)
        weights = {
            item.client_id: item.sample_count / total for item in updates
        } if total else {}
        return {
            "round_index": self.selection.round_index,
            "selected_client_ids": list(self.selection.client_ids),
            "successful_client_ids": [item.client_id for item in updates],
            "failures": [
                {"client_id": item.client_id, "stage": item.stage, "message": item.message}
                for item in failures
            ],
            "total_train_sample_count": total,
            "aggregation_weights": weights,
            "output_global_state_id": (
                self.aggregation.global_state_id if self.aggregation else None
            ),
            "elapsed_seconds": self.elapsed_seconds,
        }


@dataclass
class InMemoryFederatedServer:
    """State-owning D boundary; G persists manifests and chooses recovery storage."""

    global_state: ModelState
    global_state_id: str
    next_round_index: int = 0

    @classmethod
    def from_state(cls, state: ModelState) -> "InMemoryFederatedServer":
        cloned = clone_model_state(state)
        return cls(cloned, model_state_id(cloned))

    def apply(self, execution: RoundExecution) -> None:
        if execution.selection.round_index != self.next_round_index:
            raise ValueError("round is stale or already applied")
        if execution.aggregation is None:
            raise ValueError("cannot advance global state after an all-failure round")
        if execution.aggregation.round_index != self.next_round_index:
            raise ValueError("aggregation round does not match server state")
        self.global_state = clone_model_state(execution.aggregation.state)
        self.global_state_id = model_state_id(self.global_state)
        if self.global_state_id != execution.aggregation.global_state_id:
            raise ValueError("aggregation state identity is inconsistent")
        self.next_round_index += 1

    def recovery_state(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "next_round_index": self.next_round_index,
            "global_state_id": self.global_state_id,
            "global_state": clone_model_state(self.global_state),
        }


class DeterministicClientSelector:
    """Select the lexicographically first available clients for a traceable smoke round."""

    def select(
        self, round_index: int, available_client_ids: Sequence[str], target_count: int
    ) -> ClientSelection:
        if target_count <= 0 or target_count > len(available_client_ids):
            raise ValueError("target_count must select between one and all available clients")
        return ClientSelection(round_index, tuple(sorted(available_client_ids)[:target_count]))


def run_local_only_clients(
    clients: Mapping[str, FederatedClient], global_state: ModelState
) -> tuple[ClientResult, ...]:
    """Run every non-empty supplied client from an isolated copy of one baseline."""

    state_id = model_state_id(global_state)
    results: list[ClientResult] = []
    for client_id in sorted(clients):
        client = clients[client_id]
        try:
            request = ClientTrainRequest(0, state_id, clone_model_state(global_state))
            results.append(client.local_train(request))
        except Exception as exc:  # client failures must remain visible to the caller
            results.append(ClientFailure(client_id, 0, "local_train", type(exc).__name__, str(exc)))
    return tuple(results)


def run_federated_round(
    clients: Mapping[str, FederatedClient],
    global_state: ModelState,
    selection: ClientSelection,
) -> RoundExecution:
    """Execute exactly one selected-client round and aggregate only successful updates."""

    validate_selection_against_available(selection, tuple(clients))
    state_id = model_state_id(global_state)
    started = perf_counter()
    results: list[ClientResult] = []
    for client_id in selection.client_ids:
        try:
            request = ClientTrainRequest(
                selection.round_index, state_id, clone_model_state(global_state)
            )
            results.append(clients[client_id].local_train(request))
        except Exception as exc:
            results.append(
                ClientFailure(
                    client_id, selection.round_index, "local_train", type(exc).__name__, str(exc)
                )
            )
    updates, _ = partition_client_results(selection, results)
    aggregation = None
    if updates:
        request = AggregationRequest(
            clone_model_state(global_state), state_id, selection.round_index, updates
        )
        aggregation = FedAvgAggregator().aggregate(request)
    return RoundExecution(selection, tuple(results), aggregation, perf_counter() - started)
