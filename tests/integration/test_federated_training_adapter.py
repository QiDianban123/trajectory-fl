"""Integration of the S2-C TorchTrainer with the federated client adapter."""

from __future__ import annotations

import torch

from src.federated.aggregation import AggregationRequest
from src.federated.client import ClientTrainRequest
from src.federated.training_adapter import LocalTrainerAdapter, snapshot_model_state
from src.models.base import ModelContract
from src.models.lstm_seq2seq import LSTMSeq2Seq, LSTMSeq2SeqConfig
from src.training.torch_trainer import TorchTrainer, TorchTrainerConfig
from src.training.trainer import TrajectoryBatch


def test_torch_trainer_result_becomes_valid_client_update() -> None:
    torch.manual_seed(42)
    contract = ModelContract(history_steps=2, future_steps=3)
    model = LSTMSeq2Seq(
        contract,
        LSTMSeq2SeqConfig(hidden_size=4, num_layers=1, dropout=0.0),
    )
    trainer = TorchTrainer(
        contract,
        TorchTrainerConfig(
            epochs=2,
            learning_rate=0.01,
            gradient_clip_norm=1.0,
            seed=42,
            split_id="highd-split-42",
        ),
    )
    batch = TrajectoryBatch(
        history=torch.zeros((3, 2, 2), dtype=torch.float32),
        future=torch.full((3, 3, 2), 0.25, dtype=torch.float32),
        meta=tuple({"split_id": "highd-split-42"} for _ in range(3)),
    )
    baseline = snapshot_model_state(model.state_dict())
    adapter = LocalTrainerAdapter("rsu_01", trainer, model, [batch], [batch])

    update = adapter.local_train(ClientTrainRequest(2, baseline.state_id, baseline.state))
    request = AggregationRequest(
        global_state=baseline.state,
        global_state_id=baseline.state_id,
        round_index=2,
        updates=(update,),
    )

    assert request.total_sample_count == 3
    assert update.stats["epoch_count"] == 2.0
    assert update.stats["train_loss"] >= 0.0
    assert update.stats["validation_loss"] >= 0.0
    model_state = model.state_dict()
    for key in update.state:
        assert update.state[key].data_ptr() != baseline.state[key].data_ptr()
        assert update.state[key].data_ptr() != model_state[key].data_ptr()
