"""S3 structured client, round, and three-mode result consumers."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Literal, Mapping, Sequence

from src.evaluation.result_store import ResultRecord
from src.evaluation.visualization import plot_convergence, plot_mode_comparison


@dataclass(frozen=True)
class ClientResultRecord:
    """One structured client outcome; metrics remain the canonical ResultRecord."""

    record: ResultRecord
    client_id: str
    train_sample_count: int
    sample_visits: int
    status: Literal["completed", "failed", "skipped"] = "completed"
    error: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    evaluation_sample_count: int | None = None
    train_loss: float | None = None
    validation_loss: float | None = None
    evaluation_loss: float | None = None
    total_seconds: float | None = None
    initial_state_id: str | None = None
    final_state_id: str | None = None
    client_profile: Mapping[str, object] = field(default_factory=dict)
    artifact_paths: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.client_id, str) or not self.client_id.strip():
            raise ValueError("client_id must be a non-empty string")
        if self.train_sample_count < 0 or self.sample_visits < 0:
            raise ValueError("client counts must be non-negative")
        evaluation_count = (
            self.record.sample_count
            if self.evaluation_sample_count is None and self.status == "completed"
            else self.evaluation_sample_count
        )
        if evaluation_count is None:
            evaluation_count = 0
        if isinstance(evaluation_count, bool) or evaluation_count < 0:
            raise ValueError("evaluation_sample_count must be non-negative")
        object.__setattr__(self, "evaluation_sample_count", evaluation_count)
        for name in ("train_loss", "validation_loss", "evaluation_loss", "total_seconds"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError(f"{name} must be finite when present")
        if self.total_seconds is None:
            object.__setattr__(self, "total_seconds", self.record.total_seconds)
        if self.status == "completed":
            if (
                self.record.status != "completed"
                or self.error is not None
                or self.error_code is not None
                or self.error_message is not None
                or evaluation_count <= 0
            ):
                raise ValueError("completed client records require a completed ResultRecord")
        elif not any(
            isinstance(value, str) and value.strip()
            for value in (self.error, self.error_code, self.error_message)
        ):
            raise ValueError("failed/skipped client records require an error")
        elif self.record.status != "failed":
            raise ValueError("failed/skipped client records require a failed ResultRecord")
        if self.status != "completed" and (self.record.ade != 0.0 or self.record.fde != 0.0):
            raise ValueError("failed/skipped client metrics must use null/empty export semantics")

    def to_dict(self) -> dict[str, object]:
        """Return the schema-v2 client fact used by JSON manifests."""

        error_code = self.error_code
        error_message = self.error_message or self.error
        return {
            "run_id": self.record.run_id,
            "mode": self.record.mode,
            "client_id": self.client_id,
            "status": self.status,
            "error_code": error_code,
            "error_message": error_message,
            "train_sample_count": self.train_sample_count,
            "sample_visits": self.sample_visits,
            "evaluation_sample_count": self.evaluation_sample_count,
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "evaluation_loss": self.evaluation_loss,
            "ade": self.record.ade if self.status == "completed" else None,
            "fde": self.record.fde if self.status == "completed" else None,
            "total_seconds": self.total_seconds,
            "initial_state_id": self.initial_state_id,
            "final_state_id": self.final_state_id,
            "client_profile": dict(self.client_profile),
            "artifact_paths": dict(sorted(self.artifact_paths.items())),
        }


@dataclass(frozen=True)
class RoundRecord:
    """A finite, ordered federated round summary without aggregation logic."""

    round_index: int
    status: Literal["completed", "failed"] = "completed"
    loss: float | None = None
    ade: float | None = None
    fde: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    selected_client_ids: tuple[str, ...] = ()
    successful_client_ids: tuple[str, ...] = ()
    failures: tuple[Mapping[str, object], ...] = ()
    total_train_sample_count: int = 0
    aggregation_weights: Mapping[str, float] = field(default_factory=dict)
    input_global_state_id: str | None = None
    output_global_state_id: str | None = None
    sample_visits: int = 0
    completed_local_epochs: int = 0
    evaluation_sample_count: int = 0
    validation_loss: float | None = None
    evaluation_loss: float | None = None
    total_seconds: float = 0.0
    artifact_paths: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.round_index < 0:
            raise ValueError("round_index must be non-negative")
        for value in (self.loss, self.ade, self.fde):
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError("round metrics must be finite non-negative values")
        if self.total_train_sample_count < 0 or self.sample_visits < 0:
            raise ValueError("round counts must be non-negative")
        if len(set(self.selected_client_ids)) != len(self.selected_client_ids):
            raise ValueError("selected_client_ids must be unique")
        if not set(self.successful_client_ids).issubset(self.selected_client_ids):
            raise ValueError("successful clients must be selected")
        weight_sum = sum(self.aggregation_weights.values())
        if self.aggregation_weights and abs(weight_sum - 1.0) > 1e-9:
            raise ValueError("aggregation weights must sum to one")
        if any(not isfinite(value) or value <= 0 for value in self.aggregation_weights.values()):
            raise ValueError("aggregation weights must be positive finite values")
        if self.status == "completed":
            if self.selected_client_ids and not self.successful_client_ids:
                raise ValueError("completed rounds require successful clients")
            if self.selected_client_ids and self.output_global_state_id is None:
                raise ValueError("completed rounds require output_global_state_id")
        elif not self.error_code or not self.error_message:
            raise ValueError("failed rounds require error_code and error_message")

    @property
    def metrics(self) -> dict[str, float | int | None]:
        return {
            "train_loss": self.loss,
            "validation_loss": self.validation_loss,
            "evaluation_loss": self.evaluation_loss,
            "ade": self.ade,
            "fde": self.fde,
            "evaluation_sample_count": self.evaluation_sample_count,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "round_index": self.round_index,
            "status": self.status,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "selected_client_ids": list(self.selected_client_ids),
            "successful_client_ids": list(self.successful_client_ids),
            "failures": [dict(item) for item in self.failures],
            "total_train_sample_count": self.total_train_sample_count,
            "aggregation_weights": dict(sorted(self.aggregation_weights.items())),
            "input_global_state_id": self.input_global_state_id,
            "output_global_state_id": self.output_global_state_id,
            "sample_visits": self.sample_visits,
            "completed_local_epochs": self.completed_local_epochs,
            "metrics": self.metrics,
            "total_seconds": self.total_seconds,
            "artifact_paths": dict(sorted(self.artifact_paths.items())),
        }


@dataclass(frozen=True)
class ComparisonIdentity:
    """All frozen controls required before a multi-mode comparison."""

    partition_id: str
    scaler_id: str
    model_config_digest: str
    initial_state_id: str
    metric_schema: str
    budget_id: str = "legacy"
    data_version: str = "legacy"
    split_id: str = "legacy"
    seed: int = 0

    def __post_init__(self) -> None:
        string_values = [value for name, value in self.__dict__.items() if name != "seed"]
        if any(not isinstance(value, str) or not value.strip() for value in string_values):
            raise ValueError("comparison identity fields must be non-empty strings")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

    def to_identity_dict(self) -> dict[str, str | int]:
        return {
            "data_version": self.data_version,
            "split_id": self.split_id,
            "partition_id": self.partition_id,
            "scaler_id": self.scaler_id,
            "model_config_digest": self.model_config_digest,
            "seed": self.seed,
            "initial_state_id": self.initial_state_id,
            "metric_schema": self.metric_schema,
        }


@dataclass(frozen=True)
class FairnessRecord:
    """Strict identity and planned/actual budget comparison for one mode."""

    data_version: str
    split_id: str
    partition_id: str
    scaler_id: str
    model_config_digest: str
    seed: int
    initial_state_id: str
    metric_schema: str
    planned_budget: Mapping[str, object]
    actual_budget: Mapping[str, object]
    comparable: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        identity = self.identity
        if any(
            not isinstance(value, str) or not value.strip()
            for key, value in identity.items()
            if key != "seed"
        ):
            raise ValueError("fairness identity fields must be non-empty strings")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        planned = _validate_budget(self.planned_budget, "planned_budget")
        actual = _validate_budget(self.actual_budget, "actual_budget")
        object.__setattr__(self, "planned_budget", planned)
        object.__setattr__(self, "actual_budget", actual)
        equal = planned == actual
        if self.comparable != equal:
            raise ValueError("comparable must exactly reflect planned/actual budget equality")
        if not self.comparable and (not isinstance(self.reason, str) or not self.reason.strip()):
            raise ValueError("incomparable fairness requires a reason")
        if self.comparable and self.reason is not None:
            raise ValueError("comparable fairness must not include a reason")

    @property
    def identity(self) -> dict[str, str | int]:
        return {
            "data_version": self.data_version,
            "split_id": self.split_id,
            "partition_id": self.partition_id,
            "scaler_id": self.scaler_id,
            "model_config_digest": self.model_config_digest,
            "seed": self.seed,
            "initial_state_id": self.initial_state_id,
            "metric_schema": self.metric_schema,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.identity,
            "planned_budget": dict(self.planned_budget),
            "actual_budget": dict(self.actual_budget),
            "comparable": self.comparable,
            "reason": self.reason,
        }


def _validate_budget(value: Mapping[str, object], name: str) -> dict[str, object]:
    required = {"sample_visits", "local_epochs", "rounds", "selected_clients"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"{name} must contain exactly {sorted(required)}")
    normalized = dict(value)
    for key in ("sample_visits", "local_epochs", "rounds"):
        item = normalized[key]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{name}.{key} must be a non-negative integer")
    selected = normalized["selected_clients"]
    if not isinstance(selected, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in selected
    ):
        raise ValueError(f"{name}.selected_clients must be a list of client IDs")
    normalized["selected_clients"] = list(selected)
    return normalized


def summarize_client_results(
    records: Sequence[ClientResultRecord],
) -> tuple[ResultRecord, ResultRecord]:
    """Return macro and evaluation-sample-weighted summaries from successful clients."""

    completed = [item for item in records if item.status == "completed"]
    ids = [item.client_id for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("client result records must have unique client IDs")
    if not completed:
        raise ValueError("at least one completed client result is required")
    reference = completed[0].record
    _same_identity([item.record for item in completed])
    total = sum(item.record.sample_count for item in completed)
    macro_ade = sum(item.record.ade for item in completed) / len(completed)
    macro_fde = sum(item.record.fde for item in completed) / len(completed)
    weighted_ade = sum(item.record.ade * item.record.sample_count for item in completed) / total
    weighted_fde = sum(item.record.fde * item.record.sample_count for item in completed) / total
    base = dict(reference.__dict__)
    total_seconds = sum(item.record.total_seconds for item in completed)
    macro = ResultRecord(
        **{
            **base,
            "run_id": f"{reference.run_id}:macro",
            "sample_count": total,
            "ade": macro_ade,
            "fde": macro_fde,
            "total_seconds": total_seconds,
            "artifact_paths": {},
        }
    )
    weighted = ResultRecord(
        **{
            **base,
            "run_id": f"{reference.run_id}:weighted",
            "sample_count": total,
            "ade": weighted_ade,
            "fde": weighted_fde,
            "total_seconds": total_seconds,
            "artifact_paths": {},
        }
    )
    return macro, weighted


def plot_round_metrics(
    rounds: Sequence[RoundRecord], output_dir: str | Path
) -> tuple[Path, Path, Path]:
    """Plot only completed, contiguous round metrics in stable round order."""

    ordered = sorted(rounds, key=lambda item: item.round_index)
    if len({item.round_index for item in ordered}) != len(ordered):
        raise ValueError("round records must have unique indices")
    if any(item.status != "completed" for item in ordered):
        raise ValueError("failed rounds cannot be plotted as a continuous curve")
    if [item.round_index for item in ordered] != list(range(len(ordered))):
        raise ValueError("round records must be contiguous from zero")
    completed = [item for item in ordered if item.status == "completed"]
    if not completed:
        raise ValueError("at least one completed round is required for plots")
    indices = [item.round_index for item in completed]
    root = Path(output_dir)
    return (
        plot_convergence(
            indices, [item.loss for item in completed], root / "round_loss.png", label="loss"
        ),
        plot_convergence(
            indices, [item.ade for item in completed], root / "round_ade_m.png", label="ADE (m)"
        ),
        plot_convergence(
            indices, [item.fde for item in completed], root / "round_fde_m.png", label="FDE (m)"
        ),
    )


def compare_modes(
    records: Sequence[ResultRecord],
    output_path: str | Path,
    *,
    identities: Sequence[ComparisonIdentity],
) -> Path:
    """Reject incomparable/failed/duplicate modes before using the shared bar plot."""

    if len(records) < 2:
        raise ValueError("at least two mode records are required for comparison")
    if any(record.status != "completed" for record in records):
        raise ValueError("failed records cannot enter a mode comparison")
    if len(identities) != len(records) or len(set(identities)) != 1:
        raise ValueError("records do not share initial-state and budget identity")
    _same_identity(records)
    return plot_mode_comparison(sorted(records, key=lambda item: item.mode), output_path)


def _same_identity(records: Sequence[ResultRecord]) -> None:
    first = records[0]
    for record in records[1:]:
        if (record.seed, record.split_id, record.code_sha, record.dataset, record.model) != (
            first.seed,
            first.split_id,
            first.code_sha,
            first.dataset,
            first.model,
        ):
            raise ValueError("records do not share comparison identity")
