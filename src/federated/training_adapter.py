"""Bridge the shared Trainer result to the frozen federated client contract."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from struct import pack
from typing import Any

import numpy as np

from src.federated.client import ClientTrainRequest
from src.federated.contracts import (
    ClientUpdate,
    FederatedContractError,
    ModelState,
    ModelStateError,
    validate_model_state,
)
from src.models.base import TrajectoryPredictor, require_torch
from src.training.trainer import FitResult, Trainer, TrajectoryBatch

STATE_ID_ALGORITHM = "sha256"
CLIENT_TRAINING_STAT_KEYS = frozenset(
    {"best_epoch", "epoch_count", "train_loss", "validation_loss"}
)


@dataclass(frozen=True)
class ModelStateSnapshot:
    """An isolated model state paired with its content-derived identity."""

    state: ModelState
    state_id: str


def clone_model_state(state: ModelState) -> dict[str, Any]:
    """Validate and deeply clone a state dict without retaining tensor storage."""

    torch = require_torch()
    if not isinstance(state, Mapping) or not state:
        raise ModelStateError("model state must be a non-empty mapping")
    cloned: dict[str, Any] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not key:
            raise ModelStateError("model state keys must be non-empty strings")
        if not isinstance(value, torch.Tensor):
            raise ModelStateError(f"state entry {key!r} must be a torch.Tensor")
        if value.layout is not torch.strided:
            raise ModelStateError(f"state entry {key!r} must use strided tensor layout")
        if value.is_floating_point() and not torch.isfinite(value).all().item():
            raise ModelStateError(f"floating state entry {key!r} must contain finite values")
        cloned[key] = value.detach().clone(memory_format=torch.preserve_format)
    return cloned


def model_state_id(state: ModelState) -> str:
    """Return an order-independent SHA-256 identity for tensor names and contents."""

    torch = require_torch()
    cloned = clone_model_state(state)
    digest = sha256()
    for key in sorted(cloned):
        value = cloned[key].detach().cpu().contiguous()
        # Length-prefix every field so no two adjacent values can be ambiguous.
        _hash_field(digest, key.encode("utf-8"))
        _hash_field(digest, str(value.dtype).encode("ascii"))
        digest.update(pack(">I", value.ndim))
        for dimension in value.shape:
            digest.update(pack(">Q", dimension))
        raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        _hash_field(digest, raw)
    return f"{STATE_ID_ALGORITHM}:{digest.hexdigest()}"


def snapshot_model_state(state: ModelState) -> ModelStateSnapshot:
    """Create one immutable-baseline snapshot for dispatch to local clients."""

    cloned = clone_model_state(state)
    return ModelStateSnapshot(state=cloned, state_id=model_state_id(cloned))


def fit_result_to_client_update(
    fit_result: FitResult,
    *,
    client_id: str,
    round_index: int,
    global_state_id: str,
    reference_state: ModelState,
) -> ClientUpdate:
    """Map the best Trainer checkpoint to the shared ClientUpdate schema."""

    checkpoint_state = fit_result.checkpoint_payload["model_state"]
    if not isinstance(checkpoint_state, Mapping):  # Also guarded by FitResult's contract.
        raise ModelStateError("checkpoint.model_state must be a state_dict mapping")
    state = clone_model_state(checkpoint_state)
    validate_model_state(state, reference_state)

    by_epoch = {stat.epoch: stat for stat in fit_result.epoch_stats}
    best = by_epoch[fit_result.best_epoch]
    sample_counts = {stat.sample_count for stat in fit_result.epoch_stats}
    if len(sample_counts) != 1:
        raise FederatedContractError("FitResult sample_count must be stable across epochs")
    stats: dict[str, float] = {
        "best_epoch": float(fit_result.best_epoch),
        "epoch_count": float(len(fit_result.epoch_stats)),
        "train_loss": float(best.train_loss),
    }
    if best.validation_loss is not None:
        stats["validation_loss"] = float(best.validation_loss)
    if not set(stats).issubset(CLIENT_TRAINING_STAT_KEYS):  # Defensive schema assertion.
        raise FederatedContractError("local training produced unsupported statistic fields")

    return ClientUpdate(
        client_id=client_id,
        round_index=round_index,
        global_state_id=global_state_id,
        state=state,
        sample_count=best.sample_count,
        stats=stats,
    )


def fit_result_to_last_epoch_client_update(
    fit_result: FitResult,
    *,
    client_id: str,
    round_index: int,
    global_state_id: str,
    reference_state: ModelState,
) -> ClientUpdate:
    """Map the frozen S3 final-local-epoch state to one ClientUpdate."""

    payload = fit_result.last_checkpoint_payload
    if payload is None:
        raise FederatedContractError("S3 local training requires last_checkpoint_payload")
    checkpoint_state = payload["model_state"]
    if not isinstance(checkpoint_state, Mapping):
        raise ModelStateError("last checkpoint model_state must be a state_dict mapping")
    state = clone_model_state(checkpoint_state)
    validate_model_state(state, reference_state)
    sample_counts = {stat.sample_count for stat in fit_result.epoch_stats}
    if len(sample_counts) != 1:
        raise FederatedContractError("FitResult sample_count must be stable across epochs")
    last = fit_result.epoch_stats[-1]
    stats: dict[str, float] = {
        "best_epoch": float(fit_result.best_epoch),
        "epoch_count": float(len(fit_result.epoch_stats)),
        "train_loss": float(last.train_loss),
    }
    if last.validation_loss is not None:
        stats["validation_loss"] = float(last.validation_loss)
    return ClientUpdate(
        client_id=client_id,
        round_index=round_index,
        global_state_id=global_state_id,
        state=state,
        sample_count=last.sample_count,
        stats=stats,
    )


class LocalTrainerAdapter:
    """A federated client implementation that delegates optimization to Trainer."""

    def __init__(
        self,
        client_id: str,
        trainer: Trainer,
        model: TrajectoryPredictor,
        train_batches: Iterable[TrajectoryBatch],
        validation_batches: Iterable[TrajectoryBatch] | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        if not isinstance(client_id, str) or not client_id.strip():
            raise FederatedContractError("client_id must be a non-empty string")
        self.client_id = client_id
        self._trainer = trainer
        self._model = model
        self._train_batches = _require_reiterable(train_batches, "train_batches")
        self._validation_batches = (
            _require_reiterable(validation_batches, "validation_batches")
            if validation_batches is not None
            else None
        )
        inferred_seed = getattr(getattr(trainer, "config", None), "seed", 0)
        self._seed = inferred_seed if seed is None else seed
        if isinstance(self._seed, bool) or not isinstance(self._seed, int) or self._seed < 0:
            raise FederatedContractError("seed must be a non-negative integer")

    def local_train(self, request: ClientTrainRequest) -> ClientUpdate:
        """Train from an isolated verified baseline and return one client update."""

        reference_state = clone_model_state(request.global_state)
        actual_state_id = model_state_id(reference_state)
        if request.global_state_id != actual_state_id:
            raise FederatedContractError(
                "global_state_id does not match the dispatched global_state contents"
            )
        initial_state = clone_model_state(reference_state)
        with _isolated_random_state(self._seed):
            result = self._trainer.fit(
                self._model,
                self._train_batches,
                self._validation_batches,
                initial_state=initial_state,
            )
        # Re-hash the caller-owned state to make baseline mutation observable immediately.
        if model_state_id(request.global_state) != request.global_state_id:
            raise ModelStateError("dispatched global_state was mutated during local training")
        return fit_result_to_last_epoch_client_update(
            result,
            client_id=self.client_id,
            round_index=request.round_index,
            global_state_id=request.global_state_id,
            reference_state=reference_state,
        )


def _hash_field(digest: Any, value: bytes) -> None:
    digest.update(pack(">Q", len(value)))
    digest.update(value)


def _require_reiterable(
    batches: Iterable[TrajectoryBatch], name: str
) -> Iterable[TrajectoryBatch]:
    try:
        iterator = iter(batches)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of TrajectoryBatch values") from exc
    if isinstance(batches, Iterator) or iterator is batches:
        raise TypeError(f"{name} must be re-iterable; one-shot iterators are not supported")
    return batches


@contextmanager
def _isolated_random_state(seed: int) -> Any:
    """Make one local fit deterministic without changing caller RNG streams."""

    torch = require_torch()
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=cuda_devices):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
