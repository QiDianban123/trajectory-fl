"""S3-G production-chain tests with real LocalTrainerAdapter/TorchTrainer."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch

from src.data.client_loading import (
    ClientDataBundle,
    ClientDataLoaders,
    ClientDataProfile,
)
from src.experiments.three_mode import (
    ExperimentInputError,
    FederatedExperiment,
    FederatedRunRequest,
    LocalOnlyExperiment,
    LocalOnlyRunRequest,
    stable_config_digest,
    validate_three_mode_matrix,
)
from src.federated.training_adapter import model_state_id
from src.federated.training_factory import build_isolated_client_trainer, create_model_from_config
from src.training.trainer import TrajectoryBatch

MODEL_CONFIG = {
    "model": {
        "name": "lstm_encoder_decoder",
        "history_steps": 2,
        "future_steps": 2,
        "input_size": 2,
        "output_size": 2,
        "hidden_size": 3,
        "num_layers": 1,
        "dropout": 0.0,
        "dtype": "float32",
        "coordinate_representation": "absolute_position",
        "device_policy": "trainer_managed",
    },
    "training": {
        "epochs": 1,
        "learning_rate": 0.01,
        "gradient_clip_norm": 1.0,
        "loss": "mse",
        "optimizer": "adam",
    },
}


def _bundle(order: tuple[str, ...] = ("rsu_01", "rsu_02")) -> ClientDataBundle:
    clients = {}
    for index, client_id in enumerate(order):
        value = float(int(client_id[-1])) / 10.0
        batch = TrajectoryBatch(
            history=torch.full((2, 2, 2), value),
            future=torch.full((2, 2, 2), value + 0.1),
            meta=tuple({"split_id": "split"} for _ in range(2)),
        )
        profile = ClientDataProfile(
            client_id=client_id,
            x_min=float(index),
            x_max=float(index + 1),
            vehicle_count=1,
            group_ids=(f"group-{client_id}",),
            train_sample_count=2,
            validation_sample_count=2,
            test_sample_count=2,
            train_x_min=float(index),
            train_x_max=float(index + 1),
        )
        clients[client_id] = ClientDataLoaders(
            client_id,
            MappingProxyType({}),
            MappingProxyType({"train": [batch], "validation": [batch], "test": [batch]}),
            profile,
        )
    return ClientDataBundle(
        "data",
        "split",
        "partition",
        "scaler",
        tuple((client_id, float(i), float(i + 1)) for i, client_id in enumerate(order)),
        MappingProxyType(clients),
    )


def _initial_state():
    torch.manual_seed(7)
    return create_model_from_config(MODEL_CONFIG["model"]).state_dict()


def _identity(state):
    return {
        "data_version": "data",
        "split_id": "split",
        "partition_id": "partition",
        "scaler_id": "scaler",
        "model_config_digest": stable_config_digest(MODEL_CONFIG),
        "seed": 7,
        "initial_state_id": model_state_id(state),
        "metric_schema": "ade-fde-meter-v1",
    }


def _evaluate(client_id, model, client_data):
    del client_id
    batch = client_data.loaders["test"][0]
    with torch.no_grad():
        prediction = model(batch.history).detach().cpu().numpy()
    truth = batch.future.detach().cpu().numpy()
    distance = np.linalg.norm(prediction - truth, axis=-1)
    return {
        "evaluation_sample_count": int(prediction.shape[0]),
        "evaluation_loss": float(np.mean((prediction - truth) ** 2)),
        "ade": float(distance.mean()),
        "fde": float(distance[:, -1].mean()),
    }


def _local_request(root: Path, bundle=None):
    state = _initial_state()
    bundle = bundle or _bundle()
    return LocalOnlyRunRequest(
        identity=_identity(state),
        initial_state=state,
        client_loaders=bundle,
        model_config=MODEL_CONFIG,
        output_dir=root,
        run_id=root.name,
        local_epochs=1,
        planned_budget={
            "sample_visits": 4,
            "local_epochs": 1,
            "rounds": 1,
            "selected_clients": ["rsu_01", "rsu_02"],
        },
        evaluation_factory=_evaluate,
        code_sha="test-sha",
    )


def _federated_request(root: Path, *, rounds=1, bundle=None, callback=None):
    state = _initial_state()
    bundle = bundle or _bundle()
    selected = [client for _ in range(rounds) for client in ("rsu_01", "rsu_02")]
    return FederatedRunRequest(
        identity=_identity(state),
        initial_state=state,
        client_loaders=bundle,
        model_config=MODEL_CONFIG,
        output_dir=root,
        run_id=root.name,
        local_epochs=1,
        rounds=rounds,
        clients_per_round=2,
        planned_budget={
            "sample_visits": 4 * rounds,
            "local_epochs": 1,
            "rounds": rounds,
            "selected_clients": selected,
        },
        evaluation_factory=_evaluate,
        code_sha="test-sha",
        boundary_callback=callback,
    )


def test_real_two_client_local_and_one_federated_round(tmp_path: Path) -> None:
    local = LocalOnlyExperiment().run(_local_request(tmp_path / "local"))
    federated = FederatedExperiment().run(_federated_request(tmp_path / "fed"))
    assert local.status == federated.status == "completed"
    assert len(local.client_records) == 2
    assert federated.round_records[0].successful_client_ids == ("rsu_01", "rsu_02")
    assert federated.round_records[0].sample_visits == 4
    assert federated.round_records[0].total_train_sample_count == 4
    manifest = json.loads(federated.manifest_path.read_text())
    assert manifest["schema_version"] == 2
    assert manifest["rounds"][0]["input_global_state_id"] == model_state_id(_initial_state())
    assert Path(federated.output_dir / manifest["artifacts"]["results_csv"]).is_file()


def test_client_order_does_not_change_results_or_global_hash(tmp_path: Path) -> None:
    first = FederatedExperiment().run(_federated_request(tmp_path / "first"))
    second = FederatedExperiment().run(
        _federated_request(tmp_path / "second", bundle=_bundle(("rsu_02", "rsu_01")))
    )
    assert [item.final_state_id for item in first.client_records] == [
        item.final_state_id for item in second.client_records
    ]
    assert (
        first.round_records[0].output_global_state_id
        == second.round_records[0].output_global_state_id
    )
    assert [(item.record.ade, item.record.fde) for item in first.client_records] == pytest.approx(
        [(item.record.ade, item.record.fde) for item in second.client_records]
    )


def test_round_boundary_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    baseline = FederatedExperiment().run(_federated_request(tmp_path / "baseline", rounds=2))
    interrupted_root = tmp_path / "resumed"

    def interrupt(kind, index):
        if kind == "round" and index == 0:
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        FederatedExperiment().run(
            _federated_request(interrupted_root, rounds=2, callback=interrupt)
        )
    resume_request = replace(
        _federated_request(interrupted_root, rounds=2),
        resume_checkpoint="recovery.json",
    )
    resumed = FederatedExperiment().resume(resume_request)
    assert resumed.status == "completed"
    assert [item.output_global_state_id for item in resumed.round_records] == [
        item.output_global_state_id for item in baseline.round_records
    ]
    assert [item.final_state_id for item in resumed.client_records] == [
        item.final_state_id for item in baseline.client_records
    ]
    assert [item.sample_visits for item in resumed.client_records] == [
        item.sample_visits for item in baseline.client_records
    ]
    assert [item.sample_visits for item in resumed.round_records] == [4, 4]
    assert resumed.completed_round_indices == (0, 1)
    baseline_recovery = json.loads(baseline.recovery_path.read_text())
    resumed_recovery = json.loads(resumed.recovery_path.read_text())
    assert baseline_recovery["selector_state"] == resumed_recovery["selector_state"]
    assert baseline_recovery["artifact_paths"] == resumed_recovery["artifact_paths"]
    for key in ("rng", "torch_rng"):
        assert (baseline.output_dir / baseline_recovery["rng_state_paths"][key]).read_bytes() == (
            resumed.output_dir / resumed_recovery["rng_state_paths"][key]
        ).read_bytes()


def test_local_client_boundary_resume_matches_uninterrupted_run(tmp_path: Path) -> None:
    baseline = LocalOnlyExperiment().run(_local_request(tmp_path / "local-baseline"))
    interrupted_root = tmp_path / "local-resumed"

    def interrupt(kind, index):
        if kind == "client" and index == 0:
            raise RuntimeError("client boundary")

    with pytest.raises(RuntimeError, match="client boundary"):
        LocalOnlyExperiment().run(
            replace(_local_request(interrupted_root), boundary_callback=interrupt)
        )
    resumed = LocalOnlyExperiment().resume(
        replace(
            _local_request(interrupted_root),
            resume_checkpoint="recovery.json",
        )
    )
    assert resumed.status == "completed"
    assert [item.final_state_id for item in resumed.client_records] == [
        item.final_state_id for item in baseline.client_records
    ]
    assert [item.sample_visits for item in resumed.client_records] == [2, 2]
    assert (
        json.loads(resumed.recovery_path.read_text())["artifact_paths"]
        == json.loads(baseline.recovery_path.read_text())["artifact_paths"]
    )


def test_partial_all_failure_and_empty_clients_remain_structured(tmp_path: Path) -> None:
    def fail_selected(client_id, **kwargs):
        if client_id == "rsu_02":
            raise RuntimeError("planned client failure")
        return build_isolated_client_trainer(client_id, **kwargs)

    partial = LocalOnlyExperiment().run(
        replace(_local_request(tmp_path / "partial"), trainer_factory=fail_selected)
    )
    assert partial.status == "failed" and partial.exit_code == 1
    assert [item.status for item in partial.client_records] == ["completed", "failed"]
    assert partial.summary is not None

    def fail_all(client_id, **kwargs):
        del client_id, kwargs
        raise RuntimeError("all fail")

    failed = LocalOnlyExperiment().run(
        replace(_local_request(tmp_path / "all-fail"), trainer_factory=fail_all)
    )
    assert failed.status == "failed" and failed.summary is None
    assert json.loads(failed.manifest_path.read_text())["summary"] is None

    empty_bundle = _bundle()
    empty_clients = {}
    for client_id, client in empty_bundle.clients.items():
        empty_clients[client_id] = replace(
            client,
            loaders=MappingProxyType({"train": [], "validation": [], "test": []}),
            profile=replace(
                client.profile,
                train_sample_count=0,
                validation_sample_count=0,
                test_sample_count=0,
            ),
        )
    empty_bundle = replace(empty_bundle, clients=MappingProxyType(empty_clients))
    empty = _local_request(tmp_path / "empty", bundle=empty_bundle)
    empty = replace(
        empty,
        planned_budget={
            "sample_visits": 0,
            "local_epochs": 1,
            "rounds": 1,
            "selected_clients": [],
        },
    )
    empty_result = LocalOnlyExperiment().run(empty)
    assert empty_result.exit_code == 1 and empty_result.summary is None
    assert {item.status for item in empty_result.client_records} == {"skipped"}


def test_identity_budget_bad_checkpoint_and_completed_overwrite_exit_two(tmp_path: Path) -> None:
    bad_identity = _local_request(tmp_path / "bad-identity")
    bad_identity = replace(bad_identity, identity={**bad_identity.identity, "split_id": "wrong"})
    with pytest.raises(ExperimentInputError) as caught:
        LocalOnlyExperiment().run(bad_identity)
    assert caught.value.exit_code == 2
    assert not Path(bad_identity.output_dir).exists()

    bad_budget = replace(
        _local_request(tmp_path / "bad-budget"),
        planned_budget={
            "sample_visits": 3,
            "local_epochs": 1,
            "rounds": 1,
            "selected_clients": ["rsu_01", "rsu_02"],
        },
    )
    with pytest.raises(ExperimentInputError):
        LocalOnlyExperiment().run(bad_budget)
    assert not Path(bad_budget.output_dir).exists()

    complete_request = _local_request(tmp_path / "complete")
    LocalOnlyExperiment().run(complete_request)
    with pytest.raises(ExperimentInputError, match="already exists"):
        LocalOnlyExperiment().run(complete_request)
    with pytest.raises(ExperimentInputError, match="completed runs"):
        LocalOnlyExperiment().resume(replace(complete_request, resume_checkpoint="recovery.json"))

    interrupted = _federated_request(
        tmp_path / "broken", rounds=2, callback=lambda *_: (_ for _ in ()).throw(RuntimeError("x"))
    )
    with pytest.raises(RuntimeError):
        FederatedExperiment().run(interrupted)
    (Path(interrupted.output_dir) / "checkpoints/current_state.pt").write_bytes(b"bad")
    with pytest.raises(ExperimentInputError, match="cannot load state checkpoint"):
        FederatedExperiment().resume(
            replace(interrupted, resume_checkpoint="recovery.json", boundary_callback=None)
        )


def test_three_mode_matrix_rejects_each_identity_and_budget_field() -> None:
    state = _initial_state()
    identity = _identity(state)
    budget = {
        "sample_visits": 4,
        "local_epochs": 1,
        "rounds": 1,
        "selected_clients": ["rsu_01", "rsu_02"],
    }
    identities = {mode: dict(identity) for mode in ("centralized", "local_only", "federated")}
    budgets = {mode: dict(budget) for mode in identities}
    assert validate_three_mode_matrix(identities, budgets).exit_code == 0
    for field in identity:
        changed = {mode: dict(value) for mode, value in identities.items()}
        changed["federated"][field] = 8 if field == "seed" else "other"
        with pytest.raises(ExperimentInputError, match="identity mismatch"):
            validate_three_mode_matrix(changed, budgets)
    for field in budget:
        changed = {mode: dict(value) for mode, value in budgets.items()}
        changed["federated"][field] = ["rsu_01"] if field == "selected_clients" else 2
        with pytest.raises(ExperimentInputError, match="planned budgets"):
            validate_three_mode_matrix(identities, changed)
