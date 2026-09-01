"""Shared federated types and validation contracts frozen on Day 2."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal, TypeAlias

from src.models.base import require_torch

ModelState: TypeAlias = Mapping[str, Any]
FailureStage = Literal["selection", "download", "local_train", "upload", "validation"]


class FederatedContractError(ValueError):
    """A federated request, result, or state violates the frozen interface."""


class ModelStateError(FederatedContractError):
    """A client state is incompatible with the reference global state."""


@dataclass(frozen=True)
class ClientUpdate:
    """One successful local result, before any numerical aggregation."""

    client_id: str
    round_index: int
    global_state_id: str
    state: ModelState
    sample_count: int
    stats: Mapping[str, float]

    def __post_init__(self) -> None:
        _non_empty_string(self.client_id, "client_id")
        _non_negative_int(self.round_index, "round_index")
        _non_empty_string(self.global_state_id, "global_state_id")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int):
            raise FederatedContractError("sample_count must be a positive integer")
        if self.sample_count <= 0:
            raise FederatedContractError("sample_count must be a positive integer")
        if not isinstance(self.state, Mapping) or not self.state:
            raise FederatedContractError("state must be a non-empty state_dict mapping")
        _validate_stats(self.stats)


@dataclass(frozen=True)
class ClientFailure:
    """A selected client's explicit failure; servers must log it, never skip silently."""

    client_id: str
    round_index: int
    stage: FailureStage
    error_type: str
    message: str
    retryable: bool = False

    def __post_init__(self) -> None:
        _non_empty_string(self.client_id, "client_id")
        _non_negative_int(self.round_index, "round_index")
        if self.stage not in ("selection", "download", "local_train", "upload", "validation"):
            raise FederatedContractError("unsupported client failure stage")
        _non_empty_string(self.error_type, "error_type")
        _non_empty_string(self.message, "message")
        if not isinstance(self.retryable, bool):
            raise FederatedContractError("retryable must be boolean")


ClientResult: TypeAlias = ClientUpdate | ClientFailure


@dataclass(frozen=True)
class ClientSelection:
    """The server's immutable, traceable selection for one future round."""

    round_index: int
    client_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _non_negative_int(self.round_index, "round_index")
        if not self.client_ids:
            raise FederatedContractError("at least one client must be selected")
        invalid_ids = [
            client_id
            for client_id in self.client_ids
            if not isinstance(client_id, str) or not client_id.strip()
        ]
        if invalid_ids:
            raise FederatedContractError("selected client IDs must be non-empty strings")
        if len(set(self.client_ids)) != len(self.client_ids):
            raise FederatedContractError("selected client IDs must be unique")


def validate_model_state(state: ModelState, reference_state: ModelState) -> None:
    """Reject key, shape, dtype, tensor-type, or finite-value mismatches.

    Non-floating buffers are checked for key/shape/dtype compatibility but their
    values are intentionally ignored because the server preserves global values.
    """

    torch = require_torch()
    if not isinstance(state, Mapping) or not state:
        raise ModelStateError("client state must be a non-empty mapping")
    if not isinstance(reference_state, Mapping) or not reference_state:
        raise ModelStateError("reference global state must be a non-empty mapping")
    if set(state) != set(reference_state):
        missing = sorted(set(reference_state) - set(state))
        extra = sorted(set(state) - set(reference_state))
        raise ModelStateError(f"state keys differ; missing={missing}, extra={extra}")

    for key, reference_value in reference_state.items():
        value = state[key]
        if not isinstance(reference_value, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise ModelStateError(f"state entry {key!r} must be a torch.Tensor")
        if value.shape != reference_value.shape:
            raise ModelStateError(
                f"state entry {key!r} shape {tuple(value.shape)} does not match "
                f"{tuple(reference_value.shape)}"
            )
        if value.dtype != reference_value.dtype:
            raise ModelStateError(
                f"state entry {key!r} dtype {value.dtype} does not match {reference_value.dtype}"
            )
        if reference_value.is_floating_point() and not torch.isfinite(reference_value).all().item():
            raise ModelStateError(f"global floating state entry {key!r} must contain finite values")
        if value.is_floating_point() and not torch.isfinite(value).all().item():
            raise ModelStateError(f"floating state entry {key!r} must contain finite values")


def validate_client_update(update: ClientUpdate, reference_state: ModelState) -> None:
    """Validate one successful client result against the dispatched global state."""

    validate_model_state(update.state, reference_state)


def _validate_stats(stats: Mapping[str, float]) -> None:
    if not isinstance(stats, Mapping):
        raise FederatedContractError("stats must be a mapping")
    for key, value in stats.items():
        if not isinstance(key, str) or not key.strip():
            raise FederatedContractError("stats keys must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise FederatedContractError(f"stats value for {key!r} must be finite")


def _non_empty_string(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise FederatedContractError(f"{name} must be a non-empty string")


def _non_negative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FederatedContractError(f"{name} must be a non-negative integer")
