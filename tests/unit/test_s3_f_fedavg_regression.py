"""Independent S3-F regression examples for FedAvg and invalid client results."""

from __future__ import annotations

import pytest
import torch

from src.federated.aggregation import AggregationRequest, FedAvgAggregator
from src.federated.contracts import (
    ClientSelection,
    ClientUpdate,
    FederatedContractError,
    ModelStateError,
)
from src.federated.server import partition_client_results
from src.federated.training_adapter import clone_model_state, model_state_id


def _global() -> dict[str, torch.Tensor]:
    return {
        "parameter": torch.tensor([0.0, 0.0]),
        "counter": torch.tensor(9, dtype=torch.int64),
    }


def _update(client_id: str, values: list[float], *, samples: int, state_id: str) -> ClientUpdate:
    return ClientUpdate(
        client_id=client_id,
        round_index=0,
        global_state_id=state_id,
        state={
            "parameter": torch.tensor(values),
            "counter": torch.tensor(123, dtype=torch.int64),
        },
        sample_count=samples,
        stats={"epoch_count": 1.0},
    )


def test_manual_weighted_fedavg_preserves_integer_buffer_and_input_storage() -> None:
    global_state = _global()
    before = clone_model_state(global_state)
    state_id = model_state_id(global_state)
    first = _update("rsu_01", [1.0, 3.0], samples=1, state_id=state_id)
    second = _update("rsu_02", [5.0, 7.0], samples=3, state_id=state_id)

    result = FedAvgAggregator().aggregate(
        AggregationRequest(global_state, state_id, 0, (first, second))
    )

    # Hand-computed: (1 / 4) * [1, 3] + (3 / 4) * [5, 7] = [4, 6].
    assert torch.equal(result.state["parameter"], torch.tensor([4.0, 6.0]))
    assert result.state["counter"].item() == 9
    assert result.total_sample_count == 4
    assert result.state["parameter"].data_ptr() != first.state["parameter"].data_ptr()
    assert all(torch.equal(global_state[key], before[key]) for key in before)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_manual_fedavg_rejects_nonfinite_client_parameter(bad_value: float) -> None:
    global_state = _global()
    state_id = model_state_id(global_state)
    with pytest.raises(ModelStateError, match="finite"):
        AggregationRequest(
            global_state,
            state_id,
            0,
            (_update("rsu_01", [bad_value, 3.0], samples=1, state_id=state_id),),
        )


def test_selection_rejects_unselected_update_and_missing_selected_result() -> None:
    global_state = _global()
    state_id = model_state_id(global_state)
    selection = ClientSelection(round_index=0, client_ids=("rsu_01", "rsu_02"))
    outsider = _update("rsu_03", [1.0, 3.0], samples=1, state_id=state_id)
    with pytest.raises(FederatedContractError, match="unselected"):
        partition_client_results(selection, (outsider,))
    with pytest.raises(FederatedContractError, match="missing explicit results"):
        partition_client_results(
            selection, (_update("rsu_01", [1.0, 3.0], samples=1, state_id=state_id),)
        )
