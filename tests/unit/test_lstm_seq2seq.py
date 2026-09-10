"""Contract and gradient tests for the S2 LSTM predictor."""

from __future__ import annotations

import pytest
import torch

from src.models import LSTMSeq2Seq, LSTMSeq2SeqConfig, ModelContract, ModelContractError

CONTRACT = ModelContract(history_steps=4, future_steps=3)


def _model() -> LSTMSeq2Seq:
    return LSTMSeq2Seq(CONTRACT, LSTMSeq2SeqConfig(hidden_size=8, num_layers=1))


def test_forward_returns_finite_contract_shape_and_gradients() -> None:
    model = _model()
    history = torch.randn(2, 4, 2, dtype=torch.float32)

    prediction = model(history)
    prediction.square().mean().backward()

    assert prediction.shape == (2, 3, 2)
    assert prediction.dtype == torch.float32
    assert prediction.device == history.device
    assert torch.isfinite(prediction).all()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


@pytest.mark.parametrize(
    ("history", "message"),
    [
        (torch.zeros(2, 3, 2), "B, 4, 2"),
        (torch.zeros(2, 4, 3), "B, 4, 2"),
        (torch.zeros(2, 4, 2, dtype=torch.float64), "torch.float32"),
        (torch.full((2, 4, 2), float("nan")), "finite"),
        (torch.full((2, 4, 2), float("inf")), "finite"),
    ],
)
def test_forward_rejects_invalid_history(history: torch.Tensor, message: str) -> None:
    with pytest.raises(ModelContractError, match=message):
        _model()(history)


def test_forward_rejects_parameter_dtype_and_device_mismatch() -> None:
    history = torch.zeros(1, 4, 2, dtype=torch.float32)
    with pytest.raises(ModelContractError, match="same dtype"):
        _model().double()(history)

    model_on_meta = _model().to("meta")
    with pytest.raises(ModelContractError, match="same device"):
        model_on_meta(history)


def test_from_model_config_preserves_architecture_contract() -> None:
    config: dict[str, object] = {
        "name": "lstm_encoder_decoder",
        "history_steps": 4,
        "future_steps": 3,
        "input_size": 2,
        "output_size": 2,
        "hidden_size": 12,
        "num_layers": 2,
        "dropout": 0.2,
        "dtype": "float32",
        "coordinate_representation": "absolute_position",
        "device_policy": "trainer_managed",
    }

    model = LSTMSeq2Seq.from_model_config(config)

    assert model.contract == CONTRACT
    assert model.config == LSTMSeq2SeqConfig(hidden_size=12, num_layers=2, dropout=0.2)
    assert model.to_config() == config


def test_model_config_rejects_invalid_values() -> None:
    with pytest.raises(ModelContractError, match="hidden_size"):
        LSTMSeq2SeqConfig(hidden_size=0)
    with pytest.raises(ModelContractError, match="num_layers"):
        LSTMSeq2SeqConfig(num_layers=0)
    with pytest.raises(ModelContractError, match="dropout"):
        LSTMSeq2SeqConfig(dropout=1.0)
