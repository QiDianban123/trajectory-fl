"""Configured model initialization tests."""

import torch

from src.models import (
    LSTMSeq2Seq,
    LSTMSeq2SeqConfig,
    ModelContract,
    initialize_model,
)

INITIALIZATION = {
    "owner": "experiment_runner",
    "strategy": "xavier_uniform",
    "share_initial_state": True,
}


def _model() -> LSTMSeq2Seq:
    return LSTMSeq2Seq(
        ModelContract(history_steps=2, future_steps=3),
        LSTMSeq2SeqConfig(hidden_size=4),
    )


def test_xavier_initialization_is_seeded_and_zeroes_biases() -> None:
    torch.manual_seed(42)
    first = _model()
    initialize_model(first, INITIALIZATION)
    torch.manual_seed(42)
    second = _model()
    initialize_model(second, INITIALIZATION)

    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
        if name.endswith("bias_ih_l0") or name.endswith("bias_hh_l0"):
            assert torch.count_nonzero(value) == 0
