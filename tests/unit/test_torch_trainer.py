"""Learning, evaluation, and checkpoint tests for the shared TorchTrainer."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch

from src.models import LSTMSeq2Seq, LSTMSeq2SeqConfig, ModelContract
from src.training import TorchTrainer, TorchTrainerConfig, TrajectoryBatch

CONTRACT = ModelContract(history_steps=4, future_steps=3)
SPLIT_ID = "highd-test-split"


def _model() -> LSTMSeq2Seq:
    return LSTMSeq2Seq(CONTRACT, LSTMSeq2SeqConfig(hidden_size=8))


def _batch(batch_size: int = 4, *, include_meta: bool = False) -> TrajectoryBatch:
    history = torch.zeros(batch_size, 4, 2, dtype=torch.float32)
    target = torch.tensor(
        [[0.25, -0.25], [0.5, -0.5], [0.75, -0.75]], dtype=torch.float32
    )
    future = target.unsqueeze(0).repeat(batch_size, 1, 1)
    meta = tuple({"split_id": SPLIT_ID, "row": index} for index in range(batch_size))
    return TrajectoryBatch(history, future, meta if include_meta else ())


def _trainer(
    *, epochs: int = 1, learning_rate: float = 0.01, split_id: str = SPLIT_ID
) -> TorchTrainer:
    return TorchTrainer(
        CONTRACT,
        TorchTrainerConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            gradient_clip_norm=0.5,
            seed=42,
            split_id=split_id,
        ),
    )


def test_fit_overfits_small_batch_and_returns_best_checkpoint() -> None:
    torch.manual_seed(7)
    model = _model()
    initial_state = deepcopy(model.state_dict())
    initial_snapshot = deepcopy(initial_state)
    initial_loss = _trainer().evaluate(model, [_batch()]).loss

    trainer = _trainer(epochs=100, learning_rate=0.03)
    result = trainer.fit(model, iter([_batch()]), [_batch()], initial_state=initial_state)
    final_loss = trainer.evaluate(model, [_batch()]).loss

    assert len(result.epoch_stats) == 100
    assert all(stat.sample_count == 4 for stat in result.epoch_stats)
    assert result.best_epoch == result.checkpoint_payload["epoch"]
    assert result.last_checkpoint_payload is not None
    assert result.last_checkpoint_payload["epoch"] == result.epoch_stats[-1].epoch
    assert final_loss < initial_loss * 0.05
    assert final_loss == pytest.approx(
        result.epoch_stats[result.best_epoch].validation_loss, rel=1e-6
    )
    assert all(
        torch.equal(initial_state[key], initial_snapshot[key]) for key in initial_state
    )
    checkpoint_state = result.checkpoint_payload["model_state"]
    assert all(value.device.type == "cpu" for value in checkpoint_state.values())


def test_fit_clips_gradients_and_validates_batch_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[float, bool]] = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def recording_clip(
        parameters: Any, max_norm: float, *, error_if_nonfinite: bool
    ) -> torch.Tensor:
        calls.append((max_norm, error_if_nonfinite))
        return real_clip(parameters, max_norm=max_norm, error_if_nonfinite=error_if_nonfinite)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)
    model = _model()
    _trainer().fit(model, [_batch(include_meta=True)], None, initial_state=model.state_dict())

    assert calls == [(0.5, True)]

    wrong_split = TrajectoryBatch(
        _batch().history,
        _batch().future,
        ({"split_id": "other-split"},) * 4,
    )
    with pytest.raises(ValueError, match="metadata split_id"):
        _trainer().evaluate(model, [wrong_split])


def test_evaluate_uses_no_grad_and_restores_model_mode() -> None:
    model = _model()
    calls: list[tuple[bool, bool]] = []
    original_forward = model.forward

    def recording_forward(history: torch.Tensor) -> torch.Tensor:
        calls.append((torch.is_grad_enabled(), model.training))
        return original_forward(history)

    model.forward = recording_forward  # type: ignore[method-assign]
    model.train()

    result = _trainer().evaluate(model, [_batch(batch_size=3)])

    assert result.sample_count == 3
    assert result.loss >= 0
    assert calls == [(False, False)]
    assert model.training is True


def test_checkpoint_round_trip_restores_identical_predictions(tmp_path: Path) -> None:
    torch.manual_seed(11)
    model = _model()
    trainer = _trainer(epochs=4)
    result = trainer.fit(model, [_batch()], [_batch()], initial_state=model.state_dict())
    with torch.no_grad():
        expected = model(_batch(batch_size=2).history).clone()

    checkpoint_path = trainer.save_checkpoint(
        result.checkpoint_payload, tmp_path / "checkpoints" / "best.pt"
    )
    restored = _model()
    loaded = trainer.load_checkpoint(restored, checkpoint_path)
    with torch.no_grad():
        actual = restored(_batch(batch_size=2).history)

    assert loaded["split_id"] == SPLIT_ID
    assert restored.training is False
    assert torch.equal(actual, expected)


def test_checkpoint_rejects_corruption_split_and_model_mismatch(tmp_path: Path) -> None:
    trainer = _trainer()
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    with pytest.raises(ValueError, match="cannot load checkpoint"):
        trainer.load_checkpoint(_model(), corrupt)

    model = _model()
    result = trainer.fit(model, [_batch()], None, initial_state=model.state_dict())
    checkpoint = trainer.save_checkpoint(result.checkpoint_payload, tmp_path / "best.pt")
    with pytest.raises(ValueError, match="split_id"):
        _trainer(split_id="different-split").load_checkpoint(_model(), checkpoint)

    incompatible = LSTMSeq2Seq(CONTRACT, LSTMSeq2SeqConfig(hidden_size=12))
    with pytest.raises(ValueError, match="model_config"):
        trainer.load_checkpoint(incompatible, checkpoint)


def test_trainer_rejects_invalid_dtype_empty_batches_and_nonfinite_state() -> None:
    model = _model()
    with pytest.raises(ValueError, match="at least one sample"):
        _trainer().fit(model, [], None, initial_state=model.state_dict())
    with pytest.raises(ValueError, match="at least one sample"):
        _trainer().evaluate(model, [])

    wrong_dtype = TrajectoryBatch(
        _batch().history.double(),
        _batch().future.double(),
    )
    with pytest.raises(ValueError, match="torch.float32"):
        _trainer().evaluate(model, [wrong_dtype])

    invalid_state = deepcopy(model.state_dict())
    first_key = next(iter(invalid_state))
    invalid_state[first_key].view(-1)[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        _trainer().fit(model, [_batch()], None, initial_state=invalid_state)


def test_trainer_config_reads_frozen_model_config(
    config_bundle: dict[str, dict[str, object]],
) -> None:
    config = TorchTrainerConfig.from_config(
        config_bundle["model"], seed=9, split_id=SPLIT_ID, device="auto"
    )

    assert config.epochs == 1
    assert config.learning_rate == pytest.approx(0.001)
    assert config.gradient_clip_norm == pytest.approx(1.0)
    assert config.seed == 9
    assert config.device == "auto"
