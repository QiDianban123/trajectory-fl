"""S3-C factories preserve model/state isolation without a second trainer loop."""

from __future__ import annotations

from copy import deepcopy

from src.federated.training_adapter import model_state_id, snapshot_model_state
from src.federated.training_factory import (
    build_isolated_client_trainer,
    create_model_from_config,
    load_isolated_state,
)


def test_client_factory_creates_independent_models_and_local_epoch_trainers(config_bundle) -> None:
    model_config = deepcopy(config_bundle["model"])
    model_section = model_config["model"]
    initialization = model_config["training"]["initialization"]
    first = create_model_from_config(model_section, initialization=initialization, seed=42)
    second = create_model_from_config(model_section, initialization=initialization, seed=42)
    baseline = snapshot_model_state(first.state_dict())

    assert model_state_id(second.state_dict()) == baseline.state_id
    assert all(
        first.state_dict()[key].data_ptr() != second.state_dict()[key].data_ptr()
        for key in baseline.state
    )
    load_isolated_state(second, baseline.state)
    assert all(
        second.state_dict()[key].data_ptr() != baseline.state[key].data_ptr()
        for key in baseline.state
    )

    trainer = build_isolated_client_trainer(
        model_config, seed=42, split_id="highd-split-42", local_epochs=3
    )
    assert trainer.config.epochs == 3
    assert model_config["training"]["epochs"] == 1
