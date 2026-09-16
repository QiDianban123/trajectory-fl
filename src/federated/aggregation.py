"""FedAvg request/result contracts; numerical aggregation starts on Day 8."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from src.federated.contracts import (
    ClientUpdate,
    FederatedContractError,
    ModelState,
    validate_client_update,
    validate_model_state,
)
from src.federated.training_adapter import clone_model_state, model_state_id


class NonFloatingBufferPolicy(str, Enum):
    """Frozen handling for integer/bool state_dict buffers."""

    PRESERVE_GLOBAL = "preserve_global"


@dataclass(frozen=True)
class AggregationRequest:
    """Validated inputs for future sample-count-weighted FedAvg."""

    global_state: ModelState
    global_state_id: str
    round_index: int
    updates: tuple[ClientUpdate, ...]
    non_floating_policy: NonFloatingBufferPolicy = NonFloatingBufferPolicy.PRESERVE_GLOBAL

    def __post_init__(self) -> None:
        if not isinstance(self.global_state_id, str) or not self.global_state_id.strip():
            raise FederatedContractError("global_state_id must be a non-empty string")
        if isinstance(self.round_index, bool) or not isinstance(self.round_index, int):
            raise FederatedContractError("round_index must be a non-negative integer")
        if self.round_index < 0:
            raise FederatedContractError("round_index must be a non-negative integer")
        if not self.updates:
            raise FederatedContractError("aggregation requires at least one successful update")
        if self.non_floating_policy is not NonFloatingBufferPolicy.PRESERVE_GLOBAL:
            raise FederatedContractError("non-floating buffers must preserve global values")

        client_ids = [update.client_id for update in self.updates]
        if len(client_ids) != len(set(client_ids)):
            raise FederatedContractError("aggregation updates must have unique client IDs")
        for update in self.updates:
            if update.round_index != self.round_index:
                raise FederatedContractError("all updates must belong to the aggregation round")
            if update.global_state_id != self.global_state_id:
                raise FederatedContractError("all updates must use the dispatched global_state_id")
            validate_client_update(update, self.global_state)

    @property
    def total_sample_count(self) -> int:
        """Denominator that D8 FedAvg must use for floating parameter weights."""

        return sum(update.sample_count for update in self.updates)


@dataclass(frozen=True)
class AggregationResult:
    """The state and audit metadata returned by a future Aggregator implementation."""

    state: ModelState
    global_state_id: str
    round_index: int
    participating_client_ids: tuple[str, ...]
    total_sample_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.global_state_id, str) or not self.global_state_id.strip():
            raise FederatedContractError("global_state_id must be a non-empty string")
        if isinstance(self.round_index, bool) or not isinstance(self.round_index, int):
            raise FederatedContractError("round_index must be a non-negative integer")
        if self.round_index < 0:
            raise FederatedContractError("round_index must be a non-negative integer")
        if not self.participating_client_ids:
            raise FederatedContractError("aggregation result requires participating clients")
        if len(set(self.participating_client_ids)) != len(self.participating_client_ids):
            raise FederatedContractError("participating client IDs must be unique")
        if self.total_sample_count <= 0:
            raise FederatedContractError("total_sample_count must be positive")

    def validate_against(self, reference_state: ModelState) -> None:
        validate_model_state(self.state, reference_state)


class Aggregator(Protocol):
    """D8 implementation boundary; D2 intentionally provides no arithmetic."""

    def aggregate(self, request: AggregationRequest) -> AggregationResult:
        """Weight floating entries by sample_count and preserve global non-floats."""
        ...


class FedAvgAggregator:
    """Numerical S3 FedAvg over validated client updates."""

    def aggregate(self, request: AggregationRequest) -> AggregationResult:
        if request.global_state_id != model_state_id(request.global_state):
            raise FederatedContractError("global_state_id does not match global_state contents")
        global_state = clone_model_state(request.global_state)
        for update in request.updates:
            validate_client_update(update, global_state)
        total = request.total_sample_count
        state = {}
        for key, global_value in global_state.items():
            if global_value.is_floating_point():
                value = global_value.detach().clone().zero_()
                for update in request.updates:
                    value.add_(
                        update.state[key].to(value.device), alpha=update.sample_count / total
                    )
                state[key] = value
            else:
                state[key] = global_value.detach().clone()
        result = AggregationResult(
            state=state,
            global_state_id=model_state_id(state),
            round_index=request.round_index,
            participating_client_ids=tuple(sorted(update.client_id for update in request.updates)),
            total_sample_count=total,
        )
        result.validate_against(global_state)
        return result
