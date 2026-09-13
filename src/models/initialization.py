"""Configured and reproducible model initialization."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.models.base import TrajectoryPredictor, require_torch


def initialize_model(
    model: TrajectoryPredictor, initialization: Mapping[str, Any]
) -> None:
    """Apply the validated experiment-owned initialization strategy in place."""

    torch = require_torch()
    if not isinstance(model, torch.nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if initialization.get("owner") != "experiment_runner":
        raise ValueError("initialization owner must be experiment_runner")
    if initialization.get("strategy") != "xavier_uniform":
        raise ValueError("unsupported initialization strategy")
    if initialization.get("share_initial_state") is not True:
        raise ValueError("initialization must share one initial state")
    for name, parameter in model.named_parameters():
        if parameter.ndim >= 2:
            torch.nn.init.xavier_uniform_(parameter)
        elif parameter.ndim == 1:
            torch.nn.init.zeros_(parameter)
