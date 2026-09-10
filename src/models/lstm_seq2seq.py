"""Lightweight LSTM encoder-decoder for normalized absolute trajectories."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.models.base import BaseTrajectoryModel, ModelContract, ModelContractError, require_torch


@dataclass(frozen=True)
class LSTMSeq2SeqConfig:
    """Architecture settings kept separate from training and file concerns."""

    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.hidden_size, bool) or not isinstance(self.hidden_size, int):
            raise ModelContractError("hidden_size must be a positive integer")
        if self.hidden_size <= 0:
            raise ModelContractError("hidden_size must be a positive integer")
        if isinstance(self.num_layers, bool) or not isinstance(self.num_layers, int):
            raise ModelContractError("num_layers must be a positive integer")
        if self.num_layers <= 0:
            raise ModelContractError("num_layers must be a positive integer")
        if isinstance(self.dropout, bool) or not isinstance(self.dropout, (int, float)):
            raise ModelContractError("dropout must be a number in [0, 1)")
        if not 0.0 <= float(self.dropout) < 1.0:
            raise ModelContractError("dropout must be a number in [0, 1)")

    @classmethod
    def from_model_config(cls, config: Mapping[str, object]) -> "LSTMSeq2SeqConfig":
        """Read architecture fields from the validated ``model`` section."""

        missing = [key for key in ("hidden_size", "num_layers", "dropout") if key not in config]
        if missing:
            raise ModelContractError(f"model config is missing keys: {', '.join(missing)}")
        return cls(
            hidden_size=config["hidden_size"],  # type: ignore[arg-type]
            num_layers=config["num_layers"],  # type: ignore[arg-type]
            dropout=config["dropout"],  # type: ignore[arg-type]
        )


class LSTMSeq2Seq(BaseTrajectoryModel):
    """Encode history and decode a fixed-length absolute-position sequence.

    The decoder receives the final observed position at each future step. The
    recurrent hidden state carries the temporal signal; the model remains fully
    differentiable and returns coordinates in the same normalized space.
    """

    def __init__(self, contract: ModelContract, config: LSTMSeq2SeqConfig) -> None:
        super().__init__(contract)
        torch = require_torch()
        self.config = config
        recurrent_dropout = float(config.dropout) if config.num_layers > 1 else 0.0
        self.encoder = torch.nn.LSTM(
            input_size=contract.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.decoder = torch.nn.LSTM(
            input_size=contract.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output_projection = torch.nn.Linear(config.hidden_size, contract.output_size)

    @classmethod
    def from_model_config(cls, config: Mapping[str, object]) -> "LSTMSeq2Seq":
        """Construct the model from the validated repository model section."""

        if config.get("name") != "lstm_encoder_decoder":
            raise ModelContractError("model.name must be lstm_encoder_decoder")
        return cls(
            ModelContract.from_model_config(config),
            LSTMSeq2SeqConfig.from_model_config(config),
        )

    def to_config(self) -> dict[str, object]:
        """Return the complete architecture contract for checkpoint metadata."""

        return {
            "name": "lstm_encoder_decoder",
            "history_steps": self.contract.history_steps,
            "future_steps": self.contract.future_steps,
            "input_size": self.contract.input_size,
            "output_size": self.contract.output_size,
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "dropout": float(self.config.dropout),
            "dtype": self.contract.dtype,
            "coordinate_representation": self.contract.coordinate_representation,
            "device_policy": self.contract.device_policy,
        }

    def forward(self, history: Any) -> Any:
        """Return ``float32 [B, T_f, 2]`` on the input tensor's device."""

        self.validate_history(history)
        parameter = next(self.parameters())
        if parameter.device != history.device:
            raise ModelContractError("model parameters and history must be on the same device")
        if parameter.dtype != history.dtype:
            raise ModelContractError("model parameters and history must use the same dtype")

        _, encoder_state = self.encoder(history)
        decoder_input = history[:, -1:, :].expand(-1, self.contract.future_steps, -1)
        decoded, _ = self.decoder(decoder_input, encoder_state)
        prediction = self.output_projection(decoded)
        self.validate_prediction(prediction, history)
        return prediction
