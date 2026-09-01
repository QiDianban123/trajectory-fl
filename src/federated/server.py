"""Server selection and result-accounting contracts without a round loop."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from src.federated.aggregation import AggregationRequest, AggregationResult
from src.federated.contracts import (
    ClientFailure,
    ClientResult,
    ClientSelection,
    ClientUpdate,
    FederatedContractError,
    ModelState,
)


class ClientSelector(Protocol):
    """Select a traceable unique subset; policy implementation starts on Day 8."""

    def select(
        self, round_index: int, available_client_ids: Sequence[str], target_count: int
    ) -> ClientSelection:
        ...


def validate_selection_against_available(
    selection: ClientSelection, available_client_ids: Sequence[str]
) -> None:
    """Reject unknown clients and duplicate availability declarations."""

    if len(set(available_client_ids)) != len(available_client_ids):
        raise FederatedContractError("available client IDs must be unique")
    unknown = sorted(set(selection.client_ids) - set(available_client_ids))
    if unknown:
        raise FederatedContractError(f"selection contains unavailable clients: {unknown}")


def partition_client_results(
    selection: ClientSelection, results: Sequence[ClientResult]
) -> tuple[tuple[ClientUpdate, ...], tuple[ClientFailure, ...]]:
    """Require one explicit result per selected client and retain all failures."""

    by_client: dict[str, ClientResult] = {}
    for result in results:
        if result.round_index != selection.round_index:
            raise FederatedContractError("client result belongs to a different round")
        if result.client_id not in selection.client_ids:
            raise FederatedContractError("received a result from an unselected client")
        if result.client_id in by_client:
            raise FederatedContractError("received multiple results from one client")
        by_client[result.client_id] = result

    missing = sorted(set(selection.client_ids) - set(by_client))
    if missing:
        raise FederatedContractError(f"selected clients missing explicit results: {missing}")
    updates = tuple(result for result in results if isinstance(result, ClientUpdate))
    failures = tuple(result for result in results if isinstance(result, ClientFailure))
    return updates, failures


class FederatedServer(Protocol):
    """Future orchestration boundary; no round-control implementation exists on D2."""

    def build_aggregation_request(
        self,
        global_state: ModelState,
        global_state_id: str,
        selection: ClientSelection,
        results: Sequence[ClientResult],
    ) -> AggregationRequest:
        ...

    def apply_aggregation(self, result: AggregationResult) -> None:
        ...
