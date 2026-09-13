"""S3-G orchestration and durable recovery for local-only/federated runs.

This module deliberately owns no optimizer, client-loader, aggregation, metric,
or summary algorithm.  It composes the frozen B/C/D/E boundaries and persists
their structured facts at complete-client or complete-round boundaries.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np

from src.data.client_loading import ClientDataBundle, ClientDataLoaders
from src.evaluation.federated_results import (
    ClientResultRecord,
    ComparisonIdentity,
    FairnessRecord,
    RoundRecord,
    summarize_client_results,
)
from src.evaluation.result_store import ResultRecord
from src.federated.contracts import ClientFailure, ClientUpdate, ModelState
from src.federated.execution import (
    DeterministicClientSelector,
    InMemoryFederatedServer,
    run_federated_round,
    run_local_only_clients,
)
from src.federated.training_adapter import (
    LocalTrainerAdapter,
    clone_model_state,
    model_state_id,
)
from src.federated.training_factory import (
    build_isolated_client_trainer,
    create_model_from_config,
    load_isolated_state,
)
from src.models.base import require_torch

MODE_MANIFEST_SCHEMA_VERSION = 2
RESUME_SCHEMA_VERSION = 1
IdentityDict = dict[str, str | int]
BudgetDict = dict[str, object]
EvaluationFactory = Callable[[str, Any, ClientDataLoaders], Mapping[str, object]]
ModelFactory = Callable[[Mapping[str, object]], Any]
TrainerFactory = Callable[..., Any]


class ExperimentInputError(ValueError):
    """A usage/identity/recovery error that CLI adapters must map to exit 2."""

    exit_code = 2


@dataclass(frozen=True)
class LocalOnlyRunRequest:
    identity: ComparisonIdentity | Mapping[str, object]
    initial_state: ModelState
    client_loaders: ClientDataBundle
    model_config: Mapping[str, object]
    output_dir: str | Path
    run_id: str
    local_epochs: int
    planned_budget: Mapping[str, object]
    evaluation_factory: EvaluationFactory
    model_factory: ModelFactory = create_model_from_config
    trainer_factory: TrainerFactory = build_isolated_client_trainer
    code_sha: str = "unknown"
    config_snapshot: Mapping[str, object] = field(default_factory=dict)
    resume_checkpoint: str | Path | None = None
    boundary_callback: Callable[[str, int], None] | None = None


@dataclass(frozen=True)
class FederatedRunRequest:
    identity: ComparisonIdentity | Mapping[str, object]
    initial_state: ModelState
    client_loaders: ClientDataBundle
    model_config: Mapping[str, object]
    output_dir: str | Path
    run_id: str
    local_epochs: int
    rounds: int
    clients_per_round: int
    planned_budget: Mapping[str, object]
    evaluation_factory: EvaluationFactory
    model_factory: ModelFactory = create_model_from_config
    trainer_factory: TrainerFactory = build_isolated_client_trainer
    selector: Any = field(default_factory=DeterministicClientSelector)
    code_sha: str = "unknown"
    config_snapshot: Mapping[str, object] = field(default_factory=dict)
    resume_checkpoint: str | Path | None = None
    boundary_callback: Callable[[str, int], None] | None = None


@dataclass(frozen=True)
class ModeRunResult:
    run_id: str
    mode: Literal["local_only", "federated"]
    status: Literal["completed", "failed"]
    exit_code: int
    client_records: tuple[ClientResultRecord, ...]
    round_records: tuple[RoundRecord, ...]
    completed_client_ids: tuple[str, ...]
    completed_round_indices: tuple[int, ...]
    actual_budget: Mapping[str, object]
    fairness: FairnessRecord
    summary: ResultRecord | None
    output_dir: Path
    manifest_path: Path
    recovery_path: Path
    terminal_failures: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True)
class ThreeModeMatrixResult:
    """Top-level matrix status; mode artifacts remain owned by each runner."""

    status: Literal["completed", "failed"]
    exit_code: int
    fairness: Mapping[str, FairnessRecord]
    reason: str | None


class LocalOnlyExperiment:
    """Run every trainable client from the same state and commit per-client boundaries."""

    def run(self, request: LocalOnlyRunRequest) -> ModeRunResult:
        prepared = _preflight_local(request)
        root, recovered = _open_run(
            mode="local_only",
            output_dir=request.output_dir,
            run_id=request.run_id,
            identity=prepared.identity,
            initial_state=request.initial_state,
            planned_budget=prepared.planned_budget,
            config_snapshot=request.config_snapshot,
            resume_checkpoint=request.resume_checkpoint,
        )
        records = _records_from_manifest(
            recovered.manifest, identity=prepared.identity, code_sha=request.code_sha
        )
        completed_ids = set(recovered.completed_client_ids)
        failures: list[Mapping[str, object]] = list(recovered.terminal_failures)
        artifacts = dict(recovered.artifacts)
        _restore_rng_state(root, recovered.recovery)
        _restore_loader_states(request.client_loaders, root, recovered.recovery)

        for client_id in sorted(request.client_loaders.clients):
            if client_id in completed_ids:
                continue
            client_data = request.client_loaders.clients[client_id]
            if client_data.is_empty_train:
                record = _noncompleted_record(
                    request,
                    mode="local_only",
                    client_data=client_data,
                    status="skipped",
                    error_code="empty_train",
                    error_message="client has no training samples",
                )
            else:
                record = self._run_client(request, client_data)
            records[client_id] = record
            if record.status != "failed":
                completed_ids.add(client_id)
            elif record.status == "failed":
                failures.append(_failure_fact(record, stage="local_train"))
            client_manifest = Path("clients") / client_id / "manifest.json"
            _write_json(root / client_manifest, record.to_dict())
            artifacts[f"client_{client_id}"] = client_manifest.as_posix()
            for name, path in record.artifact_paths.items():
                artifacts[f"client_{client_id}_{name}"] = path
            actual = _local_actual_budget(records, request.local_epochs)
            _commit_recovery(
                root,
                mode="local_only",
                identity=prepared.identity,
                initial_state=request.initial_state,
                current_state=request.initial_state,
                completed_client_ids=sorted(completed_ids),
                completed_round_indices=[],
                planned_budget=prepared.planned_budget,
                actual_budget=actual,
                records=records,
                rounds=[],
                artifacts=artifacts,
                failures=failures,
                selector_state={"type": "all_clients", "next_index": len(records)},
                bundle=request.client_loaders,
                resume_from=recovered.resume_from,
            )
            if request.boundary_callback is not None:
                request.boundary_callback("client", len(records) - 1)

        actual = _local_actual_budget(records, request.local_epochs)
        return _finalize(
            root=root,
            run_id=request.run_id,
            mode="local_only",
            identity=prepared.identity,
            planned_budget=prepared.planned_budget,
            actual_budget=actual,
            records=records,
            rounds=[],
            completed_client_ids=sorted(completed_ids),
            completed_round_indices=[],
            failures=failures,
            artifacts=artifacts,
            initial_state=request.initial_state,
            current_state=request.initial_state,
            selector_state={"type": "all_clients", "next_index": len(records)},
            bundle=request.client_loaders,
            resume_from=recovered.resume_from,
        )

    def resume(self, request: LocalOnlyRunRequest) -> ModeRunResult:
        if request.resume_checkpoint is None:
            raise ExperimentInputError("resume requires resume_checkpoint")
        return self.run(request)

    @staticmethod
    def _run_client(
        request: LocalOnlyRunRequest, client_data: ClientDataLoaders
    ) -> ClientResultRecord:
        started = perf_counter()
        try:
            model, adapter = _build_adapter(request, client_data)
        except Exception as exc:
            return _noncompleted_record(
                request,
                mode="local_only",
                client_data=client_data,
                status="failed",
                error_code=type(exc).__name__,
                error_message=str(exc),
                total_seconds=perf_counter() - started,
            )
        result = run_local_only_clients({client_data.client_id: adapter}, request.initial_state)[0]
        if isinstance(result, ClientFailure):
            return _noncompleted_record(
                request,
                mode="local_only",
                client_data=client_data,
                status="failed",
                error_code=result.error_type,
                error_message=result.message,
                total_seconds=perf_counter() - started,
            )
        try:
            return _completed_client_record(
                request,
                mode="local_only",
                client_data=client_data,
                model=model,
                update=result,
                elapsed=perf_counter() - started,
            )
        except Exception as exc:
            return _noncompleted_record(
                request,
                mode="local_only",
                client_data=client_data,
                status="failed",
                error_code=type(exc).__name__,
                error_message=str(exc),
                train_sample_count=result.sample_count,
                sample_visits=result.sample_count * int(result.stats["epoch_count"]),
                total_seconds=perf_counter() - started,
                final_state_id=model_state_id(result.state),
            )


class FederatedExperiment:
    """Advance global state only after a complete aggregated round is durably recorded."""

    def run(self, request: FederatedRunRequest) -> ModeRunResult:
        prepared = _preflight_federated(request)
        root, recovered = _open_run(
            mode="federated",
            output_dir=request.output_dir,
            run_id=request.run_id,
            identity=prepared.identity,
            initial_state=request.initial_state,
            planned_budget=prepared.planned_budget,
            config_snapshot=request.config_snapshot,
            resume_checkpoint=request.resume_checkpoint,
        )
        records = _records_from_manifest(
            recovered.manifest, identity=prepared.identity, code_sha=request.code_sha
        )
        rounds = _rounds_from_manifest(recovered.manifest)
        failures: list[Mapping[str, object]] = list(recovered.terminal_failures)
        artifacts = dict(recovered.artifacts)
        server = InMemoryFederatedServer.from_state(recovered.current_state)
        server.next_round_index = len(rounds)
        if server.global_state_id != recovered.current_state_id:
            raise ExperimentInputError("recovery current_state_id does not match checkpoint")
        if rounds and rounds[-1].output_global_state_id != server.global_state_id:
            raise ExperimentInputError("recovery state does not match the last complete round")
        selector_state = recovered.recovery.get("selector_state")
        if rounds and (
            not isinstance(selector_state, Mapping)
            or selector_state.get("next_round_index") != len(rounds)
        ):
            raise ExperimentInputError("selector state does not match the round boundary")
        _restore_rng_state(root, recovered.recovery)
        _restore_loader_states(request.client_loaders, root, recovered.recovery)

        available = request.client_loaders.trainable_client_ids
        for round_index in range(server.next_round_index, request.rounds):
            selection = request.selector.select(round_index, available, request.clients_per_round)
            adapters: dict[str, Any] = {}
            models: dict[str, Any] = {}
            for client_id in selection.client_ids:
                try:
                    model, adapter = _build_adapter(
                        request, request.client_loaders.clients[client_id]
                    )
                    models[client_id], adapters[client_id] = model, adapter
                except Exception as exc:
                    adapters[client_id] = _RaisingClient(client_id, exc)
            input_state_id = server.global_state_id
            execution = run_federated_round(adapters, server.global_state, selection)
            round_client_records: dict[str, ClientResultRecord] = {}
            for result in execution.results:
                client_data = request.client_loaders.clients[result.client_id]
                if isinstance(result, ClientFailure):
                    record = _noncompleted_record(
                        request,
                        mode="federated",
                        client_data=client_data,
                        status="failed",
                        error_code=result.error_type,
                        error_message=result.message,
                        total_seconds=execution.elapsed_seconds,
                    )
                    failures.append(
                        _failure_fact(record, stage=result.stage, round_index=round_index)
                    )
                else:
                    try:
                        record = _completed_client_record(
                            request,
                            mode="federated",
                            client_data=client_data,
                            model=models[result.client_id],
                            update=result,
                            elapsed=execution.elapsed_seconds,
                            run_suffix=f":round-{round_index}",
                        )
                    except Exception as exc:
                        record = _noncompleted_record(
                            request,
                            mode="federated",
                            client_data=client_data,
                            status="failed",
                            error_code=type(exc).__name__,
                            error_message=str(exc),
                            train_sample_count=result.sample_count,
                            sample_visits=result.sample_count * int(result.stats["epoch_count"]),
                            total_seconds=execution.elapsed_seconds,
                            final_state_id=model_state_id(result.state),
                        )
                        failures.append(
                            _failure_fact(record, stage="evaluation", round_index=round_index)
                        )
                records[result.client_id] = record
                round_client_records[result.client_id] = record
                for name, path in record.artifact_paths.items():
                    artifacts[f"round_{round_index:04d}_{result.client_id}_{name}"] = path

            round_record = _round_record(
                execution,
                round_client_records,
                input_state_id=input_state_id,
                local_epochs=request.local_epochs,
            )
            round_path = Path("rounds") / f"round_{round_index:04d}.json"
            _write_json(root / round_path, round_record.to_dict())
            artifacts[f"round_{round_index:04d}"] = round_path.as_posix()
            if execution.aggregation is None:
                rounds.append(round_record)
                actual = _federated_actual_budget(rounds, request.local_epochs)
                return _finalize(
                    root=root,
                    run_id=request.run_id,
                    mode="federated",
                    identity=prepared.identity,
                    planned_budget=prepared.planned_budget,
                    actual_budget=actual,
                    records=records,
                    rounds=rounds,
                    completed_client_ids=sorted(
                        key for key, value in records.items() if value.status == "completed"
                    ),
                    completed_round_indices=[
                        item.round_index for item in rounds if item.status == "completed"
                    ],
                    failures=failures,
                    artifacts=artifacts,
                    initial_state=request.initial_state,
                    current_state=server.global_state,
                    selector_state={
                        "type": type(request.selector).__name__,
                        "next_round_index": round_index,
                    },
                    bundle=request.client_loaders,
                    resume_from=recovered.resume_from,
                )
            candidate_state = clone_model_state(execution.aggregation.state)
            state_path = Path("checkpoints") / f"global_round_{round_index:04d}.pt"
            _write_state(root / state_path, candidate_state)
            artifacts[f"global_round_{round_index:04d}"] = state_path.as_posix()
            server.apply(execution)
            rounds.append(round_record)
            actual = _federated_actual_budget(rounds, request.local_epochs)
            _commit_recovery(
                root,
                mode="federated",
                identity=prepared.identity,
                initial_state=request.initial_state,
                current_state=server.global_state,
                completed_client_ids=sorted(
                    key for key, value in records.items() if value.status == "completed"
                ),
                completed_round_indices=[
                    item.round_index for item in rounds if item.status == "completed"
                ],
                planned_budget=prepared.planned_budget,
                actual_budget=actual,
                records=records,
                rounds=rounds,
                artifacts=artifacts,
                failures=failures,
                selector_state={
                    "type": type(request.selector).__name__,
                    "next_round_index": server.next_round_index,
                },
                bundle=request.client_loaders,
                resume_from=recovered.resume_from,
            )
            if request.boundary_callback is not None:
                request.boundary_callback("round", round_index)

        actual = _federated_actual_budget(rounds, request.local_epochs)
        return _finalize(
            root=root,
            run_id=request.run_id,
            mode="federated",
            identity=prepared.identity,
            planned_budget=prepared.planned_budget,
            actual_budget=actual,
            records=records,
            rounds=rounds,
            completed_client_ids=sorted(
                key for key, value in records.items() if value.status == "completed"
            ),
            completed_round_indices=[
                item.round_index for item in rounds if item.status == "completed"
            ],
            failures=failures,
            artifacts=artifacts,
            initial_state=request.initial_state,
            current_state=server.global_state,
            selector_state={
                "type": type(request.selector).__name__,
                "next_round_index": server.next_round_index,
            },
            bundle=request.client_loaders,
            resume_from=recovered.resume_from,
        )

    def resume(self, request: FederatedRunRequest) -> ModeRunResult:
        if request.resume_checkpoint is None:
            raise ExperimentInputError("resume requires resume_checkpoint")
        return self.run(request)


def validate_three_mode_matrix(
    identities: Mapping[str, ComparisonIdentity | Mapping[str, object]],
    planned_budgets: Mapping[str, Mapping[str, object]],
    actual_budgets: Mapping[str, Mapping[str, object]] | None = None,
) -> ThreeModeMatrixResult:
    """Validate the exact three-mode identity/budget matrix without side effects."""

    required_modes = {"centralized", "local_only", "federated"}
    if set(identities) != required_modes or set(planned_budgets) != required_modes:
        raise ExperimentInputError(
            "mode matrix must contain centralized, local_only, and federated"
        )
    normalized = {mode: _identity_dict(value) for mode, value in identities.items()}
    reference = normalized["centralized"]
    mismatches = [mode for mode, value in normalized.items() if value != reference]
    if mismatches:
        raise ExperimentInputError(f"mode identity mismatch: {', '.join(sorted(mismatches))}")
    planned = {mode: _budget(value) for mode, value in planned_budgets.items()}
    if any(value != planned["centralized"] for value in planned.values()):
        raise ExperimentInputError("planned budgets differ across modes")
    actual = (
        planned
        if actual_budgets is None
        else {mode: _budget(actual_budgets[mode]) for mode in required_modes}
    )
    fairness = {
        mode: FairnessRecord(
            **reference,
            planned_budget=planned[mode],
            actual_budget=actual[mode],
            comparable=planned[mode] == actual[mode],
            reason=None if planned[mode] == actual[mode] else "planned_budget != actual_budget",
        )
        for mode in sorted(required_modes)
    }
    comparable = all(item.comparable for item in fairness.values())
    return ThreeModeMatrixResult(
        status="completed" if comparable else "failed",
        exit_code=0 if comparable else 1,
        fairness=fairness,
        reason=None if comparable else "one or more modes exceeded or missed the planned budget",
    )


@dataclass(frozen=True)
class _Prepared:
    identity: IdentityDict
    planned_budget: BudgetDict


@dataclass(frozen=True)
class _Recovered:
    manifest: Mapping[str, object]
    recovery: Mapping[str, object]
    current_state: ModelState
    current_state_id: str
    completed_client_ids: tuple[str, ...]
    terminal_failures: tuple[Mapping[str, object], ...]
    artifacts: Mapping[str, str]
    resume_from: str | None


class _RaisingClient:
    """Convert client-construction failures into D's explicit failure boundary."""

    def __init__(self, client_id: str, error: Exception) -> None:
        self.client_id = client_id
        self._error = error

    def local_train(self, request: object) -> ClientUpdate:
        del request
        raise self._error


def _preflight_local(request: LocalOnlyRunRequest) -> _Prepared:
    _positive_int(request.local_epochs, "local_epochs")
    identity = _validate_common(request)
    planned = _budget(request.planned_budget)
    expected_clients = list(request.client_loaders.trainable_client_ids)
    expected_visits = sum(
        request.client_loaders.clients[client_id].profile.train_sample_count * request.local_epochs
        for client_id in expected_clients
    )
    expected = {
        "sample_visits": expected_visits,
        "local_epochs": request.local_epochs,
        "rounds": 1,
        "selected_clients": expected_clients,
    }
    if planned != expected:
        raise ExperimentInputError(f"planned budget does not match local-only plan: {expected}")
    return _Prepared(identity, planned)


def _preflight_federated(request: FederatedRunRequest) -> _Prepared:
    _positive_int(request.local_epochs, "local_epochs")
    _positive_int(request.rounds, "rounds")
    _positive_int(request.clients_per_round, "clients_per_round")
    identity = _validate_common(request)
    available = request.client_loaders.trainable_client_ids
    if not available:
        raise ExperimentInputError("federated run requires at least one trainable client")
    selections = [
        request.selector.select(index, available, request.clients_per_round)
        for index in range(request.rounds)
    ]
    flattened = [client_id for selection in selections for client_id in selection.client_ids]
    expected = {
        "sample_visits": sum(
            request.client_loaders.clients[client_id].profile.train_sample_count
            * request.local_epochs
            for client_id in flattened
        ),
        "local_epochs": request.local_epochs,
        "rounds": request.rounds,
        "selected_clients": flattened,
    }
    planned = _budget(request.planned_budget)
    if planned != expected:
        raise ExperimentInputError(f"planned budget does not match federated plan: {expected}")
    return _Prepared(identity, planned)


def _validate_common(request: Any) -> IdentityDict:
    identity = _identity_dict(request.identity)
    bundle = request.client_loaders
    actual = {
        "data_version": bundle.data_version,
        "split_id": bundle.split_id,
        "partition_id": bundle.partition_id,
        "scaler_id": bundle.scaler_id,
        "model_config_digest": stable_config_digest(request.model_config),
        "seed": identity["seed"],
        "initial_state_id": model_state_id(request.initial_state),
        "metric_schema": identity["metric_schema"],
    }
    if identity != actual:
        differing = sorted(key for key in identity if identity[key] != actual[key])
        raise ExperimentInputError(f"experiment identity mismatch: {', '.join(differing)}")
    if not request.run_id or not isinstance(request.run_id, str):
        raise ExperimentInputError("run_id must be a non-empty string")
    root = Path(request.output_dir)
    if request.resume_checkpoint is None and root.exists():
        raise ExperimentInputError("run output already exists")
    if request.resume_checkpoint is not None and not root.is_dir():
        raise ExperimentInputError("resume run output does not exist")
    return identity


def stable_config_digest(config: Mapping[str, object]) -> str:
    try:
        encoded = json.dumps(
            config, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as exc:
        raise ExperimentInputError(f"model config is not canonical JSON: {exc}") from exc
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _snapshot_digest(config: Mapping[str, object]) -> str:
    return stable_config_digest(config)


def _identity_dict(value: ComparisonIdentity | Mapping[str, object]) -> IdentityDict:
    raw = value.to_identity_dict() if isinstance(value, ComparisonIdentity) else dict(value)
    required = {
        "data_version",
        "split_id",
        "partition_id",
        "scaler_id",
        "model_config_digest",
        "seed",
        "initial_state_id",
        "metric_schema",
    }
    if set(raw) != required:
        raise ExperimentInputError(f"identity must contain exactly {sorted(required)}")
    for key in required - {"seed"}:
        if not isinstance(raw[key], str) or not str(raw[key]).strip():
            raise ExperimentInputError(f"identity.{key} must be a non-empty string")
    seed = raw["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ExperimentInputError("identity.seed must be a non-negative integer")
    return {key: raw[key] for key in sorted(required)}  # type: ignore[return-value]


def _budget(value: Mapping[str, object]) -> BudgetDict:
    required = {"sample_visits", "local_epochs", "rounds", "selected_clients"}
    raw = dict(value)
    if set(raw) != required:
        raise ExperimentInputError(f"budget must contain exactly {sorted(required)}")
    for key in ("sample_visits", "local_epochs", "rounds"):
        if isinstance(raw[key], bool) or not isinstance(raw[key], int) or raw[key] < 0:  # type: ignore[operator]
            raise ExperimentInputError(f"budget.{key} must be a non-negative integer")
    clients = raw["selected_clients"]
    if not isinstance(clients, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in clients
    ):
        raise ExperimentInputError("budget.selected_clients must be a list of client IDs")
    raw["selected_clients"] = list(clients)
    return raw


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ExperimentInputError(f"{name} must be a positive integer")
    return value


def _build_adapter(request: Any, client_data: ClientDataLoaders) -> tuple[Any, LocalTrainerAdapter]:
    model_section = request.model_config.get("model")
    if not isinstance(model_section, Mapping):
        raise ExperimentInputError("model_config.model must be a mapping")
    seed = _identity_dict(request.identity)["seed"]
    model = request.model_factory(model_section)
    load_isolated_state(model, request.initial_state)
    trainer = request.trainer_factory(
        client_data.client_id,
        model_config=request.model_config,
        seed=seed,
        split_id=request.client_loaders.split_id,
        local_epochs=request.local_epochs,
    )
    adapter = LocalTrainerAdapter(
        client_data.client_id,
        trainer,
        model,
        client_data.loaders["train"],
        client_data.loaders["validation"] if client_data.profile.validation_sample_count else None,
        seed=seed,
    )
    return model, adapter


def _completed_client_record(
    request: Any,
    *,
    mode: Literal["local_only", "federated"],
    client_data: ClientDataLoaders,
    model: Any,
    update: ClientUpdate,
    elapsed: float,
    run_suffix: str = "",
) -> ClientResultRecord:
    load_isolated_state(model, update.state)
    if client_data.profile.test_sample_count == 0:
        return _noncompleted_record(
            request,
            mode=mode,
            client_data=client_data,
            status="skipped",
            error_code="empty_evaluation",
            error_message="client test split is empty",
            train_sample_count=update.sample_count,
            sample_visits=update.sample_count * int(update.stats["epoch_count"]),
            total_seconds=elapsed,
            final_state_id=model_state_id(update.state),
        )
    evaluated = dict(request.evaluation_factory(client_data.client_id, model, client_data))
    required = {"evaluation_sample_count", "evaluation_loss", "ade", "fde"}
    if not required.issubset(evaluated):
        missing = sorted(required - set(evaluated))
        raise ValueError(f"evaluation result is missing fields: {missing}")
    count = evaluated["evaluation_sample_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("evaluation_sample_count must be a positive integer")
    if count != client_data.profile.test_sample_count:
        raise ValueError("evaluation count does not match the frozen client profile")
    record = ResultRecord(
        run_id=f"{request.run_id}:{client_data.client_id}{run_suffix}",
        code_sha=request.code_sha,
        seed=_identity_dict(request.identity)["seed"],
        split_id=request.client_loaders.split_id,
        mode=mode,
        sample_count=count,
        ade=float(evaluated["ade"]),
        fde=float(evaluated["fde"]),
        total_seconds=elapsed,
    )
    epochs = int(update.stats["epoch_count"])
    suffix = run_suffix.replace(":", "_")
    state_relative = Path("clients") / client_data.client_id / f"final_state{suffix}.pt"
    _write_state(Path(request.output_dir) / state_relative, update.state)
    return ClientResultRecord(
        record=record,
        client_id=client_data.client_id,
        train_sample_count=update.sample_count,
        sample_visits=update.sample_count * epochs,
        evaluation_sample_count=count,
        train_loss=update.stats.get("train_loss"),
        validation_loss=update.stats.get("validation_loss"),
        evaluation_loss=float(evaluated["evaluation_loss"]),
        total_seconds=elapsed,
        initial_state_id=update.global_state_id,
        final_state_id=model_state_id(update.state),
        client_profile=client_data.profile.to_dict(),
        artifact_paths={"final_state": state_relative.as_posix()},
    )


def _noncompleted_record(
    request: Any,
    *,
    mode: Literal["local_only", "federated"],
    client_data: ClientDataLoaders,
    status: Literal["failed", "skipped"],
    error_code: str,
    error_message: str,
    train_sample_count: int = 0,
    sample_visits: int = 0,
    total_seconds: float = 0.0,
    final_state_id: str | None = None,
) -> ClientResultRecord:
    record = ResultRecord(
        run_id=f"{request.run_id}:{client_data.client_id}",
        code_sha=request.code_sha,
        seed=_identity_dict(request.identity)["seed"],
        split_id=request.client_loaders.split_id,
        mode=mode,
        sample_count=0,
        ade=0.0,
        fde=0.0,
        total_seconds=total_seconds,
        status="failed",
        error=f"{error_code}: {error_message}",
    )
    return ClientResultRecord(
        record=record,
        client_id=client_data.client_id,
        train_sample_count=train_sample_count,
        sample_visits=sample_visits,
        status=status,
        error=record.error,
        error_code=error_code,
        error_message=error_message,
        evaluation_sample_count=0,
        total_seconds=total_seconds,
        initial_state_id=model_state_id(request.initial_state),
        final_state_id=final_state_id,
        client_profile=client_data.profile.to_dict(),
    )


def _round_record(
    execution: Any,
    records: Mapping[str, ClientResultRecord],
    *,
    input_state_id: str,
    local_epochs: int,
) -> RoundRecord:
    updates = [item for item in execution.results if isinstance(item, ClientUpdate)]
    failures = sorted(
        (
            {
                "client_id": item.client_id,
                "stage": item.stage,
                "error_code": item.error_type,
                "error_message": item.message,
            }
            for item in execution.results
            if isinstance(item, ClientFailure)
        ),
        key=lambda item: (str(item["client_id"]), str(item["stage"])),
    )
    total = sum(item.sample_count for item in updates)
    weights = {item.client_id: item.sample_count / total for item in updates} if total else {}
    completed = [
        records[item.client_id] for item in updates if records[item.client_id].status == "completed"
    ]
    evaluation_total = sum(item.evaluation_sample_count or 0 for item in completed)

    def weighted(name: str) -> float | None:
        if not evaluation_total:
            return None
        return (
            sum(
                float(getattr(item.record, name)) * int(item.evaluation_sample_count or 0)
                for item in completed
            )
            / evaluation_total
        )

    loss_values = [
        item.stats.get("train_loss") for item in updates if item.stats.get("train_loss") is not None
    ]
    loss = (
        sum(float(value) * item.sample_count for value, item in zip(loss_values, updates)) / total
        if total and len(loss_values) == len(updates)
        else None
    )
    if execution.aggregation is None:
        return RoundRecord(
            round_index=execution.selection.round_index,
            status="failed",
            error_code="NoSuccessfulUpdates",
            error_message="all selected clients failed",
            selected_client_ids=execution.selection.client_ids,
            successful_client_ids=(),
            failures=tuple(failures),
            input_global_state_id=input_state_id,
            total_seconds=execution.elapsed_seconds,
        )
    return RoundRecord(
        round_index=execution.selection.round_index,
        status="completed",
        loss=loss,
        ade=weighted("ade"),
        fde=weighted("fde"),
        selected_client_ids=execution.selection.client_ids,
        successful_client_ids=tuple(item.client_id for item in updates),
        failures=tuple(failures),
        total_train_sample_count=total,
        aggregation_weights=weights,
        input_global_state_id=input_state_id,
        output_global_state_id=execution.aggregation.global_state_id,
        sample_visits=sum(item.sample_count * int(item.stats["epoch_count"]) for item in updates),
        completed_local_epochs=local_epochs,
        evaluation_sample_count=evaluation_total,
        evaluation_loss=(
            sum(
                float(item.evaluation_loss) * int(item.evaluation_sample_count or 0)
                for item in completed
            )
            / evaluation_total
            if evaluation_total
            else None
        ),
        total_seconds=execution.elapsed_seconds,
    )


def _open_run(
    *,
    mode: str,
    output_dir: str | Path,
    run_id: str,
    identity: IdentityDict,
    initial_state: ModelState,
    planned_budget: BudgetDict,
    config_snapshot: Mapping[str, object],
    resume_checkpoint: str | Path | None,
) -> tuple[Path, _Recovered]:
    root = Path(output_dir)
    if resume_checkpoint is None:
        root.mkdir(parents=True, exist_ok=False)
        _write_json(root / "config_snapshot.json", dict(config_snapshot))
        _write_state(root / "checkpoints" / "initial_state.pt", initial_state)
        artifacts = {
            "config": "config_snapshot.json",
            "initial_state": "checkpoints/initial_state.pt",
            "recovery": "recovery.json",
            "manifest": "manifest.json",
            "results_csv": "results.csv",
        }
        recovery = {
            "resume_schema_version": RESUME_SCHEMA_VERSION,
            "mode": mode,
            "run_id": run_id,
            "identity": identity,
            "config_digest": _snapshot_digest(config_snapshot),
            "initial_state_id": identity["initial_state_id"],
            "current_state_id": identity["initial_state_id"],
            "current_state_path": "checkpoints/initial_state.pt",
            "selector_state": {},
            "rng_state_paths": {},
            "completed_client_ids": [],
            "completed_round_indices": [],
            "planned_budget": planned_budget,
            "actual_budget": _zero_budget(planned_budget),
            "retry_sample_visits": 0,
            "artifact_paths": artifacts,
            "resume_from": None,
        }
        manifest = _manifest_payload(
            run_id, mode, "interrupted", None, identity, None, {}, [], None, artifacts
        )
        _write_json(root / "recovery.json", recovery)
        _write_json(root / "manifest.json", manifest)
        return root, _Recovered(
            manifest,
            recovery,
            clone_model_state(initial_state),
            str(identity["initial_state_id"]),
            (),
            (),
            artifacts,
            None,
        )

    checkpoint = Path(resume_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint
    checkpoint = checkpoint.resolve()
    if not checkpoint.is_relative_to(root.resolve()) or checkpoint.name != "recovery.json":
        raise ExperimentInputError("resume_checkpoint must be this run's recovery.json")
    recovery = _read_json(checkpoint, "recovery checkpoint")
    manifest = _read_json(root / "manifest.json", "run manifest")
    if manifest.get("status") == "completed":
        raise ExperimentInputError("completed runs cannot be resumed or overwritten")
    if manifest.get("status") not in ("failed", "interrupted"):
        raise ExperimentInputError("resume requires a failed or interrupted run")
    if recovery.get("resume_schema_version") != RESUME_SCHEMA_VERSION:
        raise ExperimentInputError("unsupported or damaged recovery schema")
    if recovery.get("mode") != mode or recovery.get("run_id") != run_id:
        raise ExperimentInputError("recovery mode/run_id does not match request")
    if recovery.get("identity") != identity or manifest.get("identity") != identity:
        raise ExperimentInputError("recovery identity does not match request")
    if recovery.get("config_digest") != _snapshot_digest(config_snapshot):
        raise ExperimentInputError("recovery config snapshot does not match request")
    if recovery.get("planned_budget") != planned_budget:
        raise ExperimentInputError("recovery planned budget does not match request")
    state_relative = recovery.get("current_state_path")
    if not isinstance(state_relative, str):
        raise ExperimentInputError("recovery current_state_path is missing")
    state_path = (root / state_relative).resolve()
    if not state_path.is_relative_to(root.resolve()):
        raise ExperimentInputError("recovery state path escapes run directory")
    state = _read_state(state_path)
    state_id = model_state_id(state)
    if state_id != recovery.get("current_state_id"):
        raise ExperimentInputError("recovery state checkpoint is damaged or stale")
    completed_clients = _string_tuple(recovery.get("completed_client_ids"), "completed_client_ids")
    completed_rounds = recovery.get("completed_round_indices")
    if not isinstance(completed_rounds, list) or completed_rounds != list(
        range(len(completed_rounds))
    ):
        raise ExperimentInputError("completed rounds are not a contiguous boundary")
    artifacts = recovery.get("artifact_paths")
    if not isinstance(artifacts, Mapping) or any(
        not isinstance(k, str) or not isinstance(v, str) for k, v in artifacts.items()
    ):
        raise ExperimentInputError("recovery artifact index is damaged")
    failures = (
        manifest.get("error", {}).get("terminal_failures", [])
        if isinstance(manifest.get("error"), Mapping)
        else []
    )
    if not isinstance(failures, list):
        raise ExperimentInputError("manifest terminal failures are damaged")
    return root, _Recovered(
        manifest,
        recovery,
        state,
        state_id,
        completed_clients,
        tuple(item for item in failures if isinstance(item, Mapping)),
        {str(k): str(v) for k, v in artifacts.items()},
        checkpoint.relative_to(root).as_posix(),
    )


def _finalize(
    *,
    root: Path,
    run_id: str,
    mode: str,
    identity: IdentityDict,
    planned_budget: BudgetDict,
    actual_budget: BudgetDict,
    records: Mapping[str, ClientResultRecord],
    rounds: Sequence[RoundRecord],
    completed_client_ids: Sequence[str],
    completed_round_indices: Sequence[int],
    failures: Sequence[Mapping[str, object]],
    artifacts: Mapping[str, str],
    initial_state: ModelState,
    current_state: ModelState,
    selector_state: Mapping[str, object],
    bundle: ClientDataBundle,
    resume_from: str | None,
) -> ModeRunResult:
    completed = [record for record in records.values() if record.status == "completed"]
    fairness = FairnessRecord(
        **identity,
        planned_budget=planned_budget,
        actual_budget=actual_budget,
        comparable=planned_budget == actual_budget,
        reason=None if planned_budget == actual_budget else "planned_budget != actual_budget",
    )
    summary = summarize_client_results(completed)[1] if completed else None
    all_rounds_completed = all(item.status == "completed" for item in rounds)
    status: Literal["completed", "failed"] = (
        "completed"
        if completed and fairness.comparable and all_rounds_completed and not failures
        else "failed"
    )
    exit_code = 0 if status == "completed" else 1
    error = (
        None
        if status == "completed"
        else {
            "code": "NoComparableEvaluation" if not completed else "IncompleteOrIncomparableRun",
            "message": "run did not produce a complete comparable result",
            "terminal_failures": list(failures),
        }
    )
    artifact_index = dict(artifacts)
    _write_results_csv(root / "results.csv", records.values())
    _commit_recovery(
        root,
        mode=mode,
        identity=identity,
        initial_state=initial_state,
        current_state=current_state,
        completed_client_ids=completed_client_ids,
        completed_round_indices=completed_round_indices,
        planned_budget=planned_budget,
        actual_budget=actual_budget,
        records=records,
        rounds=rounds,
        artifacts=artifact_index,
        failures=failures,
        selector_state=selector_state,
        bundle=bundle,
        resume_from=resume_from,
        status=status,
        fairness=fairness,
        summary=summary,
        error=error,
    )
    return ModeRunResult(
        run_id=run_id,
        mode=mode,
        status=status,
        exit_code=exit_code,
        client_records=tuple(records[key] for key in sorted(records)),
        round_records=tuple(sorted(rounds, key=lambda item: item.round_index)),
        completed_client_ids=tuple(sorted(completed_client_ids)),
        completed_round_indices=tuple(completed_round_indices),
        actual_budget=actual_budget,
        fairness=fairness,
        summary=summary,
        output_dir=root,
        manifest_path=root / "manifest.json",
        recovery_path=root / "recovery.json",
        terminal_failures=tuple(failures),
    )


def _commit_recovery(
    root: Path,
    *,
    mode: str,
    identity: IdentityDict,
    initial_state: ModelState,
    current_state: ModelState,
    completed_client_ids: Sequence[str],
    completed_round_indices: Sequence[int],
    planned_budget: BudgetDict,
    actual_budget: BudgetDict,
    records: Mapping[str, ClientResultRecord],
    rounds: Sequence[RoundRecord],
    artifacts: Mapping[str, str],
    failures: Sequence[Mapping[str, object]],
    selector_state: Mapping[str, object],
    bundle: ClientDataBundle,
    resume_from: str | None,
    status: str = "interrupted",
    fairness: FairnessRecord | None = None,
    summary: ResultRecord | None = None,
    error: Mapping[str, object] | None = None,
) -> None:
    state_path = "checkpoints/current_state.pt"
    _write_state(root / state_path, current_state)
    rng_paths = _save_rng_state(root)
    loader_paths = _save_loader_states(bundle, root)
    artifact_index = dict(artifacts)
    artifact_index["current_state"] = state_path
    artifact_index.update(rng_paths)
    artifact_index.update(loader_paths)
    recovery = {
        "resume_schema_version": RESUME_SCHEMA_VERSION,
        "mode": mode,
        "run_id": next(iter(records.values())).record.run_id.split(":", 1)[0]
        if records
        else root.name,
        "identity": identity,
        "config_digest": _snapshot_digest(
            _read_json(root / "config_snapshot.json", "config snapshot")
        ),
        "initial_state_id": model_state_id(initial_state),
        "current_state_id": model_state_id(current_state),
        "current_state_path": state_path,
        "selector_state": dict(selector_state),
        "rng_state_paths": {**rng_paths, **loader_paths},
        "completed_client_ids": list(completed_client_ids),
        "completed_round_indices": list(completed_round_indices),
        "planned_budget": planned_budget,
        "actual_budget": actual_budget,
        "retry_sample_visits": 0,
        "artifact_paths": dict(sorted(artifact_index.items())),
        "resume_from": resume_from,
    }
    manifest_error = error
    if manifest_error is None and status == "interrupted":
        manifest_error = {
            "code": "Interrupted",
            "message": "run is resumable from the last committed boundary",
            "terminal_failures": list(failures),
        }
    manifest = _manifest_payload(
        recovery["run_id"],
        mode,
        status,
        manifest_error,
        identity,
        fairness,
        records,
        rounds,
        summary,
        artifact_index,
    )
    _write_json(root / "recovery.json", recovery)
    _write_json(root / "manifest.json", manifest)


def _manifest_payload(
    run_id: str,
    mode: str,
    status: str,
    error: Mapping[str, object] | None,
    identity: IdentityDict,
    fairness: FairnessRecord | None,
    records: Mapping[str, ClientResultRecord],
    rounds: Sequence[RoundRecord],
    summary: ResultRecord | None,
    artifacts: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": MODE_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "mode": mode,
        "status": status,
        "error": dict(error) if error else None,
        "identity": identity,
        "fairness": fairness.to_dict() if fairness else None,
        "clients": [records[key].to_dict() for key in sorted(records)],
        "rounds": [item.to_dict() for item in sorted(rounds, key=lambda item: item.round_index)],
        "summary": dict(summary.__dict__) if summary else None,
        "artifacts": dict(sorted(artifacts.items())),
    }


def _records_from_manifest(
    manifest: Mapping[str, object], *, identity: Mapping[str, str | int], code_sha: str
) -> dict[str, ClientResultRecord]:
    # Recovery needs accounting facts, not rehydrated ResultRecord objects.  The
    # durable client JSON files retain full records and are loaded here.
    root_clients = manifest.get("clients", [])
    if not isinstance(root_clients, list):
        raise ExperimentInputError("manifest clients are damaged")
    records: dict[str, ClientResultRecord] = {}
    for payload in root_clients:
        if not isinstance(payload, Mapping):
            raise ExperimentInputError("manifest client record is damaged")
        record = _client_record_from_dict(payload, identity=identity, code_sha=code_sha)
        if record.client_id in records:
            raise ExperimentInputError("manifest contains duplicate client records")
        records[record.client_id] = record
    return records


def _client_record_from_dict(
    payload: Mapping[str, object], *, identity: Mapping[str, str | int], code_sha: str
) -> ClientResultRecord:
    status = payload.get("status")
    if status not in ("completed", "failed", "skipped"):
        raise ExperimentInputError("client record status is damaged")
    completed = status == "completed"
    result = ResultRecord(
        run_id=str(payload["run_id"]),
        code_sha=code_sha,
        seed=int(identity["seed"]),
        split_id=str(identity["split_id"]),
        mode=str(payload["mode"]),  # type: ignore[arg-type]
        sample_count=int(payload.get("evaluation_sample_count", 0)) if completed else 0,
        ade=float(payload["ade"]) if completed else 0.0,
        fde=float(payload["fde"]) if completed else 0.0,
        total_seconds=float(payload.get("total_seconds", 0.0)),
        status="completed" if completed else "failed",
        error=None if completed else str(payload.get("error_message") or "recovered failure"),
    )
    return ClientResultRecord(
        result,
        str(payload["client_id"]),
        int(payload.get("train_sample_count", 0)),
        int(payload.get("sample_visits", 0)),
        status=status,  # type: ignore[arg-type]
        error=None if completed else str(payload.get("error_message") or "recovered failure"),
        error_code=payload.get("error_code")
        if isinstance(payload.get("error_code"), str)
        else None,
        error_message=payload.get("error_message")
        if isinstance(payload.get("error_message"), str)
        else None,
        evaluation_sample_count=int(payload.get("evaluation_sample_count", 0)),
        train_loss=_optional_float(payload.get("train_loss")),
        validation_loss=_optional_float(payload.get("validation_loss")),
        evaluation_loss=_optional_float(payload.get("evaluation_loss")),
        initial_state_id=payload.get("initial_state_id")
        if isinstance(payload.get("initial_state_id"), str)
        else None,
        final_state_id=payload.get("final_state_id")
        if isinstance(payload.get("final_state_id"), str)
        else None,
        client_profile=payload.get("client_profile")
        if isinstance(payload.get("client_profile"), Mapping)
        else {},
        artifact_paths=payload.get("artifact_paths")
        if isinstance(payload.get("artifact_paths"), Mapping)
        else {},
    )


def _rounds_from_manifest(manifest: Mapping[str, object]) -> list[RoundRecord]:
    raw = manifest.get("rounds", [])
    if not isinstance(raw, list):
        raise ExperimentInputError("manifest rounds are damaged")
    rounds: list[RoundRecord] = []
    for payload in raw:
        if not isinstance(payload, Mapping):
            raise ExperimentInputError("manifest round is damaged")
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
        rounds.append(
            RoundRecord(
                round_index=int(payload["round_index"]),
                status=str(payload["status"]),  # type: ignore[arg-type]
                loss=_optional_float(metrics.get("train_loss")),
                ade=_optional_float(metrics.get("ade")),
                fde=_optional_float(metrics.get("fde")),
                error_code=payload.get("error_code")
                if isinstance(payload.get("error_code"), str)
                else None,
                error_message=payload.get("error_message")
                if isinstance(payload.get("error_message"), str)
                else None,
                selected_client_ids=tuple(payload.get("selected_client_ids", [])),
                successful_client_ids=tuple(payload.get("successful_client_ids", [])),
                failures=tuple(payload.get("failures", [])),
                total_train_sample_count=int(payload.get("total_train_sample_count", 0)),
                aggregation_weights=dict(payload.get("aggregation_weights", {})),
                input_global_state_id=payload.get("input_global_state_id"),
                output_global_state_id=payload.get("output_global_state_id"),
                sample_visits=int(payload.get("sample_visits", 0)),
                completed_local_epochs=int(payload.get("completed_local_epochs", 0)),
                evaluation_sample_count=int(metrics.get("evaluation_sample_count", 0)),
                validation_loss=_optional_float(metrics.get("validation_loss")),
                evaluation_loss=_optional_float(metrics.get("evaluation_loss")),
                total_seconds=float(payload.get("total_seconds", 0.0)),
                artifact_paths=dict(payload.get("artifact_paths", {})),
            )
        )
    if [item.round_index for item in rounds] != list(range(len(rounds))):
        raise ExperimentInputError("manifest rounds are not contiguous")
    return rounds


def _local_actual_budget(records: Mapping[str, ClientResultRecord], epochs: int) -> BudgetDict:
    completed_training = [item for item in records.values() if item.sample_visits > 0]
    return {
        "sample_visits": sum(item.sample_visits for item in completed_training),
        "local_epochs": epochs,
        "rounds": 1,
        "selected_clients": sorted(item.client_id for item in completed_training),
    }


def _federated_actual_budget(rounds: Sequence[RoundRecord], epochs: int) -> BudgetDict:
    completed = [item for item in rounds if item.status == "completed"]
    return {
        "sample_visits": sum(item.sample_visits for item in completed),
        "local_epochs": epochs,
        "rounds": len(completed),
        "selected_clients": [
            client_id for item in completed for client_id in item.successful_client_ids
        ],
    }


def _zero_budget(planned: Mapping[str, object]) -> BudgetDict:
    return {
        "sample_visits": 0,
        "local_epochs": planned["local_epochs"],
        "rounds": 0,
        "selected_clients": [],
    }


def _failure_fact(
    record: ClientResultRecord, *, stage: str, round_index: int = 0
) -> dict[str, object]:
    return {
        "client_id": record.client_id,
        "round_index": round_index,
        "stage": stage,
        "error_code": record.error_code,
        "error_message": record.error_message,
    }


def _save_rng_state(root: Path) -> dict[str, str]:
    torch = require_torch()
    random_path = Path("checkpoints") / "rng.json"
    torch_path = Path("checkpoints") / "torch_rng.pt"
    numpy_state = np.random.get_state()
    _write_json(
        root / random_path,
        {
            "python": _jsonable_state(random.getstate()),
            "numpy": {
                "kind": numpy_state[0],
                "keys": numpy_state[1].tolist(),
                "position": numpy_state[2],
                "has_gauss": numpy_state[3],
                "cached_gaussian": numpy_state[4],
            },
        },
    )
    _write_state(root / torch_path, {"cpu_rng_state": torch.get_rng_state()})
    return {"rng": random_path.as_posix(), "torch_rng": torch_path.as_posix()}


def _restore_rng_state(root: Path, recovery: Mapping[str, object]) -> None:
    paths = recovery.get("rng_state_paths")
    if not isinstance(paths, Mapping) or not paths:
        return
    rng_path = paths.get("rng")
    torch_path = paths.get("torch_rng")
    if not isinstance(rng_path, str) or not isinstance(torch_path, str):
        raise ExperimentInputError("recovery RNG paths are damaged")
    payload = _read_json(root / rng_path, "RNG checkpoint")
    python_state = payload.get("python")
    numpy_state = payload.get("numpy")
    if not isinstance(python_state, list) or not isinstance(numpy_state, Mapping):
        raise ExperimentInputError("recovery RNG state is damaged")
    random.setstate(_tuple_state(python_state))
    np.random.set_state(
        (
            str(numpy_state["kind"]),
            np.asarray(numpy_state["keys"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch_state = _read_state(root / torch_path)
    require_torch().set_rng_state(torch_state["cpu_rng_state"])


def _save_loader_states(bundle: ClientDataBundle, root: Path) -> dict[str, str]:
    states: dict[str, Any] = {}
    for client_id, client in bundle.clients.items():
        for split, loader in client.loaders.items():
            generator = getattr(loader, "generator", None)
            if generator is not None:
                states[f"{client_id}:{split}"] = generator.get_state()
    if not states:
        return {}
    path = Path("checkpoints") / "loader_rng.pt"
    _write_state(root / path, states)
    return {"loader_rng": path.as_posix()}


def _restore_loader_states(
    bundle: ClientDataBundle, root: Path, recovery: Mapping[str, object]
) -> None:
    paths = recovery.get("rng_state_paths")
    path = paths.get("loader_rng") if isinstance(paths, Mapping) else None
    if not isinstance(path, str):
        return
    states = _read_state(root / path)
    for key, state in states.items():
        client_id, split = key.split(":", 1)
        if client_id not in bundle.clients or split not in bundle.clients[client_id].loaders:
            raise ExperimentInputError("loader RNG checkpoint does not match client bundle")
        generator = getattr(bundle.clients[client_id].loaders[split], "generator", None)
        if generator is None:
            raise ExperimentInputError("loader RNG checkpoint targets a loader without generator")
        generator.set_state(state)


def _write_state(path: Path, state: Mapping[str, Any]) -> None:
    torch = require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        torch.save(dict(state), temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_state(path: Path) -> dict[str, Any]:
    torch = require_torch()
    if not path.is_file():
        raise ExperimentInputError(f"state checkpoint does not exist: {path}")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError, ValueError, IndexError) as exc:
        raise ExperimentInputError(f"cannot load state checkpoint: {exc}") from exc
    if not isinstance(value, Mapping) or not value:
        raise ExperimentInputError("state checkpoint root must be a non-empty mapping")
    return dict(value)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        temporary.write_text(f"{text}\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, name: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentInputError(f"cannot read {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentInputError(f"{name} root must be an object")
    return value


def _write_results_csv(path: Path, records: Sequence[ClientResultRecord] | Any) -> None:
    rows = [record.to_dict() for record in sorted(records, key=lambda item: item.client_id)]
    fields = [
        "run_id",
        "mode",
        "client_id",
        "status",
        "error_code",
        "error_message",
        "train_sample_count",
        "sample_visits",
        "evaluation_sample_count",
        "train_loss",
        "validation_loss",
        "evaluation_loss",
        "ade",
        "fde",
        "total_seconds",
        "initial_state_id",
        "final_state_id",
        "client_profile",
        "artifact_paths",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        key: json.dumps(row[key], ensure_ascii=False, sort_keys=True)
                        if key in ("client_profile", "artifact_paths")
                        else row[key]
                        for key in fields
                    }
                )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _jsonable_state(value: object) -> object:
    if isinstance(value, tuple):
        return [_jsonable_state(item) for item in value]
    if isinstance(value, list):
        return [_jsonable_state(item) for item in value]
    return value


def _tuple_state(value: object) -> Any:
    if isinstance(value, list):
        return tuple(_tuple_state(item) for item in value)
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ExperimentInputError(f"{name} is damaged")
    if len(set(value)) != len(value):
        raise ExperimentInputError(f"{name} contains duplicates")
    return tuple(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)
