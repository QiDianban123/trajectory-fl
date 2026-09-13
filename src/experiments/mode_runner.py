"""Production dispatch for the frozen S3 three-mode contract."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.data.client_loading import create_client_dataloaders
from src.data.loading import DataLoaderConfig, ProcessedDatasetReader
from src.data.partition import ClientPartition, PartitionManifest
from src.evaluation.centralized import evaluate_prediction_arrays
from src.evaluation.federated_results import ComparisonIdentity
from src.evaluation.prediction import collect_predictions
from src.experiments.centralized import CentralizedExperiment, CentralizedExperimentRequest
from src.experiments.three_mode import (
    ExperimentInputError,
    FederatedExperiment,
    FederatedRunRequest,
    LocalOnlyExperiment,
    LocalOnlyRunRequest,
    ThreeModeExperiment,
    ThreeModeRunRequest,
    stable_config_digest,
)
from src.federated.training_adapter import clone_model_state, model_state_id
from src.federated.training_factory import create_model_from_config
from src.models.base import ModelContract
from src.models.initialization import initialize_model
from src.training.torch_trainer import TorchTrainer, TorchTrainerConfig
from src.utils.paths import resolve_within, validate_run_id
from src.utils.seed import set_global_seed

Mode = Literal["centralized", "local_only", "federated"]
MODES: tuple[Mode, ...] = ("centralized", "local_only", "federated")


@dataclass(frozen=True)
class ModeDispatchResult:
    mode: Mode
    run_id: str
    output_dir: Path
    result: object
    actual_budget: Mapping[str, object]
    exit_code: int


def run_mode(
    bundle,
    *,
    mode,
    project_root,
    processed_dir,
    output_root,
    run_id,
    expected_data_version=None,
    expected_split_id=None,
    resume_checkpoint=None,
    code_sha=None,
):
    context = _prepare(
        bundle, project_root, processed_dir, output_root, expected_data_version, expected_split_id
    )
    return _run(bundle, context, mode, run_id, resume_checkpoint, code_sha or _git_sha(context[0]))


def run_three_mode_matrix(
    bundle, *, project_root, processed_dir, output_root, run_id, code_sha=None
):
    validate_run_id(run_id)
    context = _prepare(bundle, project_root, processed_dir, output_root, None, None)
    sha = code_sha or _git_sha(context[0])
    plans = {m: _budget(context[4], context[7], m) for m in MODES}
    return ThreeModeExperiment().run(
        ThreeModeRunRequest(
            identities={m: context[5] for m in MODES},
            planned_budgets=plans,
            runners={
                m: (
                    lambda current=m: _run(
                        bundle, context, current, f"{run_id}-{current}", None, sha
                    )
                )
                for m in MODES
            },
        )
    )


def _prepare(bundle, project_root, processed_dir, output_root, data_version, split_id):
    root = Path(project_root).resolve()
    processed = _path(root, Path(processed_dir), "processed_dir")
    output = _path(root, Path(output_root), "output_root")
    if output != (root / "outputs").resolve():
        raise ExperimentInputError("output_root must be outputs")
    if not any(processed.is_relative_to(p) for p in ((root / "data/processed").resolve(), output)):
        raise ExperimentInputError("processed_dir must be under data/processed or outputs")
    settings = _settings(bundle)
    data = ProcessedDatasetReader().load(
        processed,
        data_config=bundle["data"],
        expected_data_version=data_version,
        expected_split_id=split_id,
    )
    clients = create_client_dataloaders(
        data,
        _partition(processed),
        contract=ModelContract.from_model_config(bundle["model"]["model"]),
        config=DataLoaderConfig.from_config_bundle(bundle),
    )
    seed = int(bundle["experiment"]["run"]["seed"])
    set_global_seed(seed)
    model = create_model_from_config(bundle["model"]["model"])
    initialize_model(model, bundle["model"]["training"]["initialization"])
    state = clone_model_state(model.state_dict())
    identity = ComparisonIdentity(
        data_version=clients.data_version,
        split_id=clients.split_id,
        partition_id=clients.partition_id,
        scaler_id=clients.scaler_id,
        model_config_digest=stable_config_digest(bundle["model"]),
        seed=seed,
        initial_state_id=model_state_id(state),
        metric_schema=str(bundle["experiment"]["three_mode"]["metric_schema"]),
    )
    return root, processed, output, data, clients, identity, state, settings


def _run(bundle, c, mode, run_id, resume, sha):
    root, processed, output_root, data, clients, identity, state, settings = c
    if mode not in MODES:
        raise ExperimentInputError(f"unsupported mode {mode!r}")
    validate_run_id(run_id)
    run_root = output_root / run_id
    resume = _resume(run_root, resume)
    planned = _budget(clients, settings, mode)
    if mode == "centralized":
        if resume:
            raise ExperimentInputError("centralized recovery.json is unsupported")
        effective = deepcopy(bundle)
        epochs = settings["rounds"] * settings["local_epochs"]
        effective["experiment"]["run"]["mode"] = mode
        effective["model"]["training"]["epochs"] = epochs
        result = CentralizedExperiment().run(
            CentralizedExperimentRequest(
                config_bundle=effective,
                processed_dir=processed,
                project_root=root,
                output_root=output_root,
                run_id=run_id,
                code_sha=sha,
                expected_data_version=identity.data_version,
                expected_split_id=identity.split_id,
                initial_state=clone_model_state(state),
            )
        )
        actual = _central_actual(result.output_dir, clients, epochs)
        _central_manifest(result, identity, planned, actual)
        return ModeDispatchResult(
            mode, run_id, run_root, result, actual, 0 if planned == actual else 1
        )
    epochs = (
        settings["rounds"] * settings["local_epochs"]
        if mode == "local_only"
        else settings["local_epochs"]
    )
    common = dict(
        identity=identity,
        initial_state=clone_model_state(state),
        client_loaders=clients,
        model_config=bundle["model"],
        output_dir=run_root,
        run_id=run_id,
        local_epochs=epochs,
        planned_budget=planned,
        evaluation_factory=_evaluator(data.scaler, bundle["model"]),
        config_snapshot=bundle,
        resume_checkpoint=resume,
        code_sha=sha,
    )
    result = (
        LocalOnlyExperiment().run(LocalOnlyRunRequest(**common))
        if mode == "local_only"
        else FederatedExperiment().run(
            FederatedRunRequest(
                **common, rounds=settings["rounds"], clients_per_round=settings["clients_per_round"]
            )
        )
    )
    return ModeDispatchResult(
        mode, run_id, run_root, result, result.actual_budget, result.exit_code
    )


def _settings(bundle):
    raw = bundle["experiment"].get("three_mode")
    if not isinstance(raw, Mapping):
        raise ExperimentInputError("three_mode is required")
    return {k: int(raw[k]) for k in ("rounds", "local_epochs", "clients_per_round")}


def _budget(clients, s, mode):
    ids = list(clients.trainable_client_ids)
    if s["clients_per_round"] != len(ids):
        raise ExperimentInputError("fair matrix requires all trainable clients")
    selected = ids * s["rounds"] if mode == "federated" else ids
    epochs = s["local_epochs"] if mode == "federated" else s["rounds"] * s["local_epochs"]
    visits = sum(clients.clients[x].profile.train_sample_count for x in selected) * epochs
    return {
        "sample_visits": visits,
        "local_epochs": epochs,
        "rounds": s["rounds"] if mode == "federated" else 1,
        "selected_clients": selected,
    }


def _evaluator(scaler, config):
    def evaluate(_cid, model, client):
        trainer = TorchTrainer(
            model.contract,
            TorchTrainerConfig.from_config(
                config, seed=0, split_id=client.datasets["test"].split_id
            ),
        )
        loss = trainer.evaluate(model, client.loaders["test"]).loss
        pred = collect_predictions(
            model, client.loaders["test"], contract=model.contract, device=trainer.device
        )
        metrics = evaluate_prediction_arrays(pred.prediction, pred.truth, scaler).metrics
        return {
            "evaluation_sample_count": pred.sample_count,
            "evaluation_loss": loss,
            "ade": metrics["ade"],
            "fde": metrics["fde"],
        }

    return evaluate


def _partition(directory):
    try:
        raw = json.loads((directory / "partition_manifest.json").read_text(encoding="utf-8"))
        clients = tuple(
            ClientPartition(
                client_id=x["client_id"],
                x_min=x["x_min"],
                x_max=x["x_max"],
                group_ids=tuple(x["group_ids"]),
                sample_count=x["sample_count"],
            )
            for x in raw["clients"]
        )
        return PartitionManifest(
            tuple(raw["region_edges"]),
            clients,
            axis=raw["axis"],
            num_clients_requested=raw["num_clients_requested"],
        )
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        raise ExperimentInputError(f"invalid partition manifest: {exc}") from exc


def _path(root, value, name):
    try:
        return resolve_within(
            root, value.resolve().relative_to(root) if value.is_absolute() else value
        )
    except ValueError as exc:
        raise ExperimentInputError(f"{name} escapes project") from exc


def _resume(run_root, value):
    if value is None:
        return None
    result = (value if Path(value).is_absolute() else run_root / value).resolve()
    if result != (run_root / "checkpoints/recovery.json").resolve():
        raise ExperimentInputError("resume_checkpoint must be this run's recovery.json")
    return result


def _central_actual(directory, clients, epochs):
    rows = json.loads((directory / "training_history.json").read_text())["epochs"]
    return {
        "sample_visits": sum(int(x["sample_count"]) for x in rows),
        "local_epochs": epochs,
        "rounds": 1,
        "selected_clients": list(clients.trainable_client_ids),
    }


def _central_manifest(result, identity, planned, actual):
    comparable = planned == actual
    payload = {
        "schema_version": 2,
        "run_id": result.record.run_id,
        "status": "completed" if comparable else "failed",
        "error": None if comparable else {"code": "BudgetMismatch"},
        "identity": identity.to_identity_dict(),
        "fairness": {
            **identity.to_identity_dict(),
            "planned_budget": planned,
            "actual_budget": actual,
            "comparable": comparable,
            "reason": None if comparable else "planned_budget != actual_budget",
        },
        "clients": [],
        "rounds": [],
        "summary": result.record.to_dict() if comparable else None,
        "artifacts": {
            "metrics_json": "metrics.json",
            "metrics_csv": "metrics.csv",
            "loss_curve": "figures/loss_curve.png",
            "trajectory": "figures/prediction_trajectory.png",
        },
    }
    (result.output_dir / "s3_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _git_sha(root):
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"
