"""Parameter-free constant-velocity trajectory baseline."""

from __future__ import annotations

from typing import Any

from src.models.base import BaseTrajectoryModel, ModelContract, ModelContractError, require_torch


class ConstantVelocityBaseline(BaseTrajectoryModel):
    """Extrapolate the last observed displacement for every future step."""

    def __init__(self, contract: ModelContract) -> None:
        if contract.history_steps < 2:
            raise ModelContractError("constant velocity requires at least two history steps")
        super().__init__(contract)

    def forward(self, history: Any) -> Any:
        """Return the frozen ``[B, T_f, 2]`` normalized future contract."""

        torch = require_torch()
        self.validate_history(history)
        velocity = history[:, -1, :] - history[:, -2, :]
        offsets = torch.arange(
            1,
            self.contract.future_steps + 1,
            dtype=history.dtype,
            device=history.device,
        ).view(1, -1, 1)
        prediction = history[:, -1:, :] + offsets * velocity.unsqueeze(1)
        self.validate_prediction(prediction, history)
        return prediction
