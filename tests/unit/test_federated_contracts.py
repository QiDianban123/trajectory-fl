"""Artificial-state tests for Day 2 federated contracts; no FedAvg arithmetic."""

from __future__ import annotations

import pytest
import torch

from src.federated.aggregation import (
    AggregationRequest,
    NonFloatingBufferPolicy,
)
from src.federated.contracts import (
    ClientFailure,
    ClientSelection,
    ClientUpdate,
    FederatedContractError,
    ModelStateError,
    validate_client_update,
)
from src.federated.server import (
    partition_client_results,
    validate_selection_against_available,
)


def _global_state() -> dict[str, torch.Tensor]:
    return {
        "weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "num_batches_tracked": torch.tensor(3, dtype=torch.int64),
    }


def _update(
    client_id: str = "rsu-1",
    *,
    sample_count: int = 4,
    state: dict[str, torch.Tensor] | None = None,
) -> ClientUpdate:
    client_state = state or {
        "weight": torch.tensor([[2.0, 4.0]], dtype=torch.float32),
        # The value may differ locally; aggregation must preserve the global buffer.
        "num_batches_tracked": torch.tensor(9, dtype=torch.int64),
    }
    return ClientUpdate(
        client_id=client_id,
        round_index=0,
        global_state_id="global-0",
        state=client_state,
        sample_count=sample_count,
        stats={"train_loss": 0.5},
    )


def test_valid_update_matches_global_schema_and_preserves_nonfloat_policy() -> None:
    update = _update()
    validate_client_update(update, _global_state())
    request = AggregationRequest(
        global_state=_global_state(),
        global_state_id="global-0",
        round_index=0,
        updates=(update,),
    )
    assert request.total_sample_count == 4
    assert request.non_floating_policy is NonFloatingBufferPolicy.PRESERVE_GLOBAL


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ({"weight": torch.ones((1, 2), dtype=torch.float32)}, "state keys differ"),
        (
            {
                "weight": torch.ones((2, 2), dtype=torch.float32),
                "num_batches_tracked": torch.tensor(3, dtype=torch.int64),
            },
            "shape",
        ),
        (
            {
                "weight": torch.ones((1, 2), dtype=torch.float64),
                "num_batches_tracked": torch.tensor(3, dtype=torch.int64),
            },
            "dtype",
        ),
        (
            {
                "weight": torch.tensor([[float("nan"), 1.0]], dtype=torch.float32),
                "num_batches_tracked": torch.tensor(3, dtype=torch.int64),
            },
            "finite",
        ),
    ],
)
def test_update_rejects_key_shape_and_nonfinite_mismatches(
    state: dict[str, torch.Tensor], message: str
) -> None:
    with pytest.raises(ModelStateError, match=message):
        validate_client_update(_update(state=state), _global_state())


def test_update_requires_positive_effective_sample_count() -> None:
    with pytest.raises(FederatedContractError, match="positive integer"):
        _update(sample_count=0)


def test_selection_and_results_require_exact_client_accounting() -> None:
    selection = ClientSelection(round_index=0, client_ids=("rsu-1", "rsu-2"))
    validate_selection_against_available(selection, ["rsu-1", "rsu-2", "rsu-3"])
    failure = ClientFailure(
        client_id="rsu-2",
        round_index=0,
        stage="local_train",
        error_type="RuntimeError",
        message="client training failed",
    )
    updates, failures = partition_client_results(selection, [_update(), failure])
    assert [update.client_id for update in updates] == ["rsu-1"]
    assert [item.client_id for item in failures] == ["rsu-2"]

    with pytest.raises(FederatedContractError, match="missing explicit results"):
        partition_client_results(selection, [_update()])


def test_aggregation_rejects_duplicate_or_stale_updates() -> None:
    with pytest.raises(FederatedContractError, match="unique client IDs"):
        AggregationRequest(
            global_state=_global_state(),
            global_state_id="global-0",
            round_index=0,
            updates=(_update(), _update()),
        )

    stale = ClientUpdate(
        client_id="rsu-2",
        round_index=0,
        global_state_id="older-global",
        state=_global_state(),
        sample_count=2,
        stats={},
    )
    with pytest.raises(FederatedContractError, match="dispatched global_state_id"):
        AggregationRequest(
            global_state=_global_state(),
            global_state_id="global-0",
            round_index=0,
            updates=(stale,),
        )
