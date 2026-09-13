"""D execution boundary: selected clients, failures, aggregation, and recovery facts."""

from __future__ import annotations

import pytest
import torch

from src.federated.client import ClientTrainRequest
from src.federated.contracts import ClientUpdate
from src.federated.execution import (
    DeterministicClientSelector,
    InMemoryFederatedServer,
    run_federated_round,
    run_local_only_clients,
)
from src.federated.training_adapter import clone_model_state


class OffsetClient:
    def __init__(self, client_id: str, offset: float, samples: int, *, fail: bool = False) -> None:
        self.client_id, self.offset, self.samples, self.fail = client_id, offset, samples, fail

    def local_train(self, request: ClientTrainRequest):
        if self.fail:
            raise RuntimeError("planned failure")
        state = clone_model_state(request.global_state)
        state["weight"].add_(self.offset)
        return ClientUpdate(
            self.client_id,
            request.round_index,
            request.global_state_id,
            state,
            self.samples,
            {"train_loss": 1.0},
        )


def test_two_clients_one_round_and_server_state_advance() -> None:
    state = {"weight": torch.tensor([0.0]), "counter": torch.tensor(7, dtype=torch.int64)}
    clients = {"rsu_02": OffsetClient("rsu_02", 5.0, 3), "rsu_01": OffsetClient("rsu_01", 1.0, 1)}
    selector = DeterministicClientSelector()
    server = InMemoryFederatedServer.from_state(state)
    selection = selector.select(0, tuple(clients), 2)
    execution = run_federated_round(clients, server.global_state, selection)
    assert execution.aggregation is not None
    assert execution.aggregation.state["weight"].tolist() == pytest.approx([4.0])
    assert execution.aggregation.state["counter"].item() == 7
    assert state["weight"].item() == 0.0
    assert execution.to_manifest()["aggregation_weights"] == {"rsu_01": 0.25, "rsu_02": 0.75}
    server.apply(execution)
    assert server.next_round_index == 1
    assert server.recovery_state()["global_state_id"] == server.global_state_id


def test_partial_and_all_failure_are_visible_and_not_aggregated() -> None:
    state = {"weight": torch.tensor([0.0])}
    clients = {"ok": OffsetClient("ok", 2.0, 2), "bad": OffsetClient("bad", 0.0, 1, fail=True)}
    selection = DeterministicClientSelector().select(0, tuple(clients), 2)
    partial = run_federated_round(clients, state, selection)
    assert partial.aggregation is not None
    assert partial.to_manifest()["failures"][0]["client_id"] == "bad"
    failed_selector = DeterministicClientSelector().select(0, ("bad",), 1)
    all_failed = run_federated_round({"bad": clients["bad"]}, state, failed_selector)
    assert all_failed.aggregation is None
    assert all_failed.to_manifest()["aggregation_weights"] == {}
    with pytest.raises(ValueError, match="all-failure"):
        InMemoryFederatedServer.from_state(state).apply(all_failed)


def test_local_only_runs_every_client_and_retains_failures() -> None:
    results = run_local_only_clients(
        {"ok": OffsetClient("ok", 1.0, 1), "bad": OffsetClient("bad", 1.0, 1, fail=True)},
        {"weight": torch.tensor([0.0])},
    )
    assert [result.client_id for result in results] == ["bad", "ok"]
    assert type(results[0]).__name__ == "ClientFailure"
