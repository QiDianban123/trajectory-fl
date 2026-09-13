"""S2-F production learning and checkpoint recovery matrix."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch

from src.models import LSTMSeq2Seq, LSTMSeq2SeqConfig, ModelContract
from src.training import TorchTrainer, TorchTrainerConfig, TrajectoryBatch

CONTRACT = ModelContract(history_steps=4, future_steps=3)
SPLIT_ID = "s2-f-recovery"


def _model() -> LSTMSeq2Seq:
    return LSTMSeq2Seq(
        CONTRACT,
        LSTMSeq2SeqConfig(hidden_size=8, num_layers=2, dropout=0.5),
    )


def _trainer(*, epochs: int = 3) -> TorchTrainer:
    return TorchTrainer(
        CONTRACT,
        TorchTrainerConfig(
            epochs=epochs,
            learning_rate=0.02,
            gradient_clip_norm=1.0,
            seed=42,
            split_id=SPLIT_ID,
        ),
    )


def _batch() -> TrajectoryBatch:
    return TrajectoryBatch(
        torch.zeros((4, 4, 2), dtype=torch.float32),
        torch.ones((4, 3, 2), dtype=torch.float32),
    )


def _fit_from(state: dict[str, torch.Tensor]) -> dict[str, object]:
    model = _model()
    model.load_state_dict(state)
    return dict(
        _trainer().fit(model, [_batch()], [_batch()], initial_state=state).checkpoint_payload
    )


def test_fit_seed_reproduces_dropout_state_and_preserves_caller_rng() -> None:
    torch.manual_seed(9)
    initial_state = deepcopy(_model().state_dict())
    models = [_model(), _model()]
    for model in models:
        model.load_state_dict(initial_state)
    torch.manual_seed(123)
    expected_next = torch.rand(4)
    torch.manual_seed(123)

    first = _trainer().fit(
        models[0], [_batch()], [_batch()], initial_state=initial_state
    ).checkpoint_payload
    actual_next = torch.rand(4)
    second = _trainer().fit(
        models[1], [_batch()], [_batch()], initial_state=initial_state
    ).checkpoint_payload

    assert torch.equal(actual_next, expected_next)
    for key in first["model_state"]:
        assert torch.equal(first["model_state"][key], second["model_state"][key])
    assert first["metrics"] == second["metrics"]


def test_checkpoint_rejects_missing_schema_and_state_corruption(tmp_path: Path) -> None:
    trainer = _trainer(epochs=1)
    with pytest.raises(ValueError, match="does not exist"):
        trainer.load_checkpoint(_model(), tmp_path / "missing.pt")

    torch.manual_seed(9)
    valid = _fit_from(deepcopy(_model().state_dict()))
    first_key = next(iter(valid["model_state"]))
    invalid_payloads = []

    wrong_schema = deepcopy(valid)
    wrong_schema["schema_version"] = 99
    invalid_payloads.append((wrong_schema, "schema_version"))

    missing_key = deepcopy(valid)
    missing_key["model_state"].pop(first_key)
    invalid_payloads.append((missing_key, "keys do not match"))

    wrong_shape = deepcopy(valid)
    wrong_shape["model_state"][first_key] = torch.zeros(1)
    invalid_payloads.append((wrong_shape, "shape"))

    wrong_dtype = deepcopy(valid)
    wrong_dtype["model_state"][first_key] = valid["model_state"][first_key].double()
    invalid_payloads.append((wrong_dtype, "dtype"))

    for index, (payload, message) in enumerate(invalid_payloads):
        path = tmp_path / f"invalid-{index}.pt"
        torch.save(payload, path)
        with pytest.raises(ValueError, match=message):
            trainer.load_checkpoint(_model(), path)
