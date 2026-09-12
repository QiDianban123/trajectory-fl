"""Tests for the Trainer-to-federated state and result adapter."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import pytest
import torch

from src.federated.aggregation import AggregationRequest
from src.federated.client import ClientTrainRequest
from src.federated.contracts import FederatedContractError, ModelStateError
from src.federated.training_adapter import (
    CLIENT_TRAINING_STAT_KEYS,
    LocalTrainerAdapter,
    clone_model_state,
    fit_result_to_client_update,
    fit_result_to_last_epoch_client_update,
    model_state_id,
    snapshot_model_state,
)
from src.models.base import ModelContract
from src.models.lstm_seq2seq import LSTMSeq2Seq, LSTMSeq2SeqConfig
from src.training.trainer import EpochStats, EvaluationResult, FitResult, TrajectoryBatch

CONTRACT = ModelContract(history_steps=2, future_steps=3)


def _model() -> LSTMSeq2Seq:
    return LSTMSeq2Seq(CONTRACT, LSTMSeq2SeqConfig(hidden_size=4, num_layers=1, dropout=0.0))


def _fit_result(state: Mapping[str, Any], sample_count: int = 4) -> FitResult:
    payload = {
        "schema_version": 1,
        "model_state": state,
        "model_config": {"name": "lstm_encoder_decoder"},
        "seed": 42,
        "epoch": 1,
        "split_id": "highd-split-42",
        "metrics": {"loss": 0.75},
    }
    return FitResult(
        epoch_stats=(
            EpochStats(epoch=0, sample_count=sample_count, train_loss=1.0, validation_loss=1.5),
            EpochStats(epoch=1, sample_count=sample_count, train_loss=0.5, validation_loss=0.75),
        ),
        best_epoch=1,
        checkpoint_payload=payload,
        last_checkpoint_payload=payload,
    )


class MutatingTrainer:
    """Trainer double proving the adapter never passes caller-owned tensors."""

    def __init__(self, trained_state: Mapping[str, Any]) -> None:
        self.trained_state = trained_state
        self.received_initial_state: Mapping[str, Any] | None = None

    def fit(
        self,
        model: LSTMSeq2Seq,
        train_batches: Iterable[TrajectoryBatch],
        validation_batches: Iterable[TrajectoryBatch] | None,
        *,
        initial_state: Mapping[str, Any],
    ) -> FitResult:
        self.received_initial_state = initial_state
        first = next(iter(initial_state.values()))
        first.add_(100)
        return _fit_result(self.trained_state)

    def evaluate(
        self, model: LSTMSeq2Seq, batches: Iterable[TrajectoryBatch]
    ) -> EvaluationResult:
        return EvaluationResult(sample_count=1, loss=0.0)


class BaselineOffsetTrainer(MutatingTrainer):
    """Return a state derived only from the isolated baseline passed by the adapter."""

    def fit(
        self,
        model: LSTMSeq2Seq,
        train_batches: Iterable[TrajectoryBatch],
        validation_batches: Iterable[TrajectoryBatch] | None,
        *,
        initial_state: Mapping[str, Any],
    ) -> FitResult:
        self.received_initial_state = initial_state
        trained = clone_model_state(initial_state)
        next(iter(trained.values())).add_(0.25)
        return _fit_result(trained)


def test_state_snapshot_is_deep_and_hash_is_stable_across_mapping_order() -> None:
    state = {
        "weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        "counter": torch.tensor(3, dtype=torch.int64),
    }
    reordered = {"counter": state["counter"].clone(), "weight": state["weight"].clone()}
    snapshot = snapshot_model_state(state)

    assert snapshot.state_id == model_state_id(reordered)
    assert snapshot.state_id.startswith("sha256:")
    assert all(snapshot.state[key].data_ptr() != value.data_ptr() for key, value in state.items())
    assert model_state_id({"other": state["weight"]}) != model_state_id(
        {"weight": state["weight"]}
    )
    assert model_state_id({"value": torch.tensor([1], dtype=torch.int32)}) != model_state_id(
        {"value": torch.tensor([1.4013e-45], dtype=torch.float32)}
    )
    assert model_state_id({"value": torch.tensor([[1.0, 2.0]])}) != model_state_id(
        {"value": torch.tensor([1.0, 2.0])}
    )
    state["weight"].add_(10)
    assert torch.equal(snapshot.state["weight"], torch.tensor([[1.0, 2.0]]))
    assert model_state_id(state) != snapshot.state_id


@pytest.mark.parametrize(
    "state, message",
    [
        ({}, "non-empty"),
        ({"": torch.tensor(1)}, "keys"),
        ({"weight": "bad"}, "torch.Tensor"),
        ({"weight": torch.tensor(float("nan"))}, "finite"),
    ],
)
def test_clone_model_state_rejects_invalid_state(state: dict[str, Any], message: str) -> None:
    with pytest.raises(ModelStateError, match=message):
        clone_model_state(state)


def test_fit_result_maps_shared_fields_and_is_aggregation_compatible() -> None:
    baseline = _model().state_dict()
    trained = clone_model_state(baseline)
    next(iter(trained.values())).add_(0.25)
    update = fit_result_to_client_update(
        _fit_result(trained),
        client_id="rsu_01",
        round_index=2,
        global_state_id="global-state",
        reference_state=baseline,
    )

    assert update.sample_count == 4
    assert update.stats == {
        "best_epoch": 1.0,
        "epoch_count": 2.0,
        "train_loss": 0.5,
        "validation_loss": 0.75,
    }
    assert set(update.stats).issubset(CLIENT_TRAINING_STAT_KEYS)
    request = AggregationRequest(
        global_state=baseline,
        global_state_id="global-state",
        round_index=2,
        updates=(update,),
    )
    assert request.total_sample_count == 4
    first_key = next(iter(trained))
    assert update.state[first_key].data_ptr() != trained[first_key].data_ptr()


def test_last_epoch_adapter_keeps_best_adapter_compatible() -> None:
    baseline = _model().state_dict()
    best = clone_model_state(baseline)
    last = clone_model_state(baseline)
    next(iter(best.values())).add_(1.0)
    next(iter(last.values())).add_(2.0)
    result = _fit_result(best)
    last_payload = dict(result.checkpoint_payload)
    last_payload["model_state"] = last
    result = FitResult(
        epoch_stats=result.epoch_stats,
        best_epoch=result.best_epoch,
        checkpoint_payload=result.checkpoint_payload,
        last_checkpoint_payload=last_payload,
    )
    old = fit_result_to_client_update(
        result, client_id="rsu_01", round_index=0, global_state_id="state", reference_state=baseline
    )
    current = fit_result_to_last_epoch_client_update(
        result, client_id="rsu_01", round_index=0, global_state_id="state", reference_state=baseline
    )
    key = next(iter(baseline))
    assert torch.equal(old.state[key], best[key])
    assert torch.equal(current.state[key], last[key])


def test_fit_result_rejects_state_shape_dtype_and_unstable_sample_count() -> None:
    baseline = _model().state_dict()
    key = next(iter(baseline))
    for invalid in (
        {**baseline, key: torch.zeros((1,), dtype=baseline[key].dtype)},
        {**baseline, key: baseline[key].to(torch.float64)},
    ):
        with pytest.raises(ModelStateError):
            fit_result_to_client_update(
                _fit_result(invalid),
                client_id="rsu_01",
                round_index=0,
                global_state_id="state",
                reference_state=baseline,
            )

    inconsistent = _fit_result(baseline)
    inconsistent = FitResult(
        epoch_stats=(inconsistent.epoch_stats[0], EpochStats(1, 5, 0.5, 0.75)),
        best_epoch=1,
        checkpoint_payload=inconsistent.checkpoint_payload,
    )
    with pytest.raises(FederatedContractError, match="stable across epochs"):
        fit_result_to_client_update(
            inconsistent,
            client_id="rsu_01",
            round_index=0,
            global_state_id="state",
            reference_state=baseline,
        )


def test_local_adapter_delegates_with_isolated_baseline_and_preserves_request() -> None:
    model = _model()
    global_state = clone_model_state(model.state_dict())
    baseline = clone_model_state(global_state)
    trained = clone_model_state(global_state)
    next(iter(trained.values())).add_(0.5)
    trainer = MutatingTrainer(trained)
    adapter = LocalTrainerAdapter("rsu_01", trainer, model, ())
    state_id = model_state_id(global_state)

    update = adapter.local_train(ClientTrainRequest(3, state_id, global_state))

    assert update.client_id == "rsu_01"
    assert update.round_index == 3
    assert update.global_state_id == state_id
    assert all(torch.equal(global_state[key], baseline[key]) for key in baseline)
    assert trainer.received_initial_state is not None
    assert all(
        trainer.received_initial_state[key].data_ptr() != global_state[key].data_ptr()
        for key in global_state
    )


def test_local_adapter_rejects_state_id_mismatch_before_training() -> None:
    model = _model()
    state = model.state_dict()
    trainer = MutatingTrainer(state)
    adapter = LocalTrainerAdapter("rsu_01", trainer, model, ())

    with pytest.raises(FederatedContractError, match="does not match"):
        adapter.local_train(ClientTrainRequest(0, "sha256:stale", state))
    assert trainer.received_initial_state is None


def test_local_adapter_rejects_one_shot_batches_and_invalid_seed() -> None:
    model = _model()
    trainer = MutatingTrainer(model.state_dict())

    with pytest.raises(TypeError, match="train_batches must be re-iterable"):
        LocalTrainerAdapter("rsu_01", trainer, model, iter(()))
    with pytest.raises(TypeError, match="validation_batches must be re-iterable"):
        LocalTrainerAdapter("rsu_01", trainer, model, (), iter(()))
    with pytest.raises(FederatedContractError, match="seed must be a non-negative integer"):
        LocalTrainerAdapter("rsu_01", trainer, model, (), seed=-1)


def test_client_execution_order_does_not_change_updates_or_global_baseline() -> None:
    model_a = _model()
    model_b = _model()
    model_b.load_state_dict(model_a.state_dict())
    global_state = clone_model_state(model_a.state_dict())
    original = clone_model_state(global_state)
    state_id = model_state_id(global_state)
    request = ClientTrainRequest(0, state_id, global_state)
    adapters = {
        "rsu_01": LocalTrainerAdapter(
            "rsu_01", BaselineOffsetTrainer(global_state), model_a, ()
        ),
        "rsu_02": LocalTrainerAdapter(
            "rsu_02", BaselineOffsetTrainer(global_state), model_b, ()
        ),
    }

    first_order = {client_id: adapters[client_id].local_train(request) for client_id in adapters}
    second_order = {
        client_id: adapters[client_id].local_train(request)
        for client_id in reversed(tuple(adapters))
    }

    for client_id in adapters:
        for key in global_state:
            assert torch.equal(
                first_order[client_id].state[key], second_order[client_id].state[key]
            )
            assert torch.equal(global_state[key], original[key])
