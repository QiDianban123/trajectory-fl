"""S3 structured client, round, and three-mode result consumers."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal, Sequence

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

    def __post_init__(self) -> None:
        if not isinstance(self.client_id, str) or not self.client_id.strip():
            raise ValueError("client_id must be a non-empty string")
        if self.train_sample_count < 0 or self.sample_visits < 0:
            raise ValueError("client counts must be non-negative")
        if self.status == "completed":
            if self.record.status != "completed" or self.error is not None:
                raise ValueError("completed client records require a completed ResultRecord")
        elif not isinstance(self.error, str) or not self.error.strip():
            raise ValueError("failed/skipped client records require an error")
        elif self.record.status != "failed":
            raise ValueError("failed/skipped client records require a failed ResultRecord")


@dataclass(frozen=True)
class RoundRecord:
    """A finite, ordered federated round summary without aggregation logic."""

    round_index: int
    status: Literal["completed", "failed"]
    loss: float | None
    ade: float | None
    fde: float | None

    def __post_init__(self) -> None:
        if self.round_index < 0:
            raise ValueError("round_index must be non-negative")
        for value in (self.loss, self.ade, self.fde):
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError("round metrics must be finite non-negative values")
        if self.status == "completed" and None in (self.loss, self.ade, self.fde):
            raise ValueError("completed rounds require loss, ADE, and FDE")


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
    macro = ResultRecord(
        **{**base, "sample_count": len(completed), "ade": macro_ade, "fde": macro_fde}
    )
    weighted = ResultRecord(
        **{**base, "sample_count": total, "ade": weighted_ade, "fde": weighted_fde}
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
    records: Sequence[ResultRecord], output_path: str | Path, *, identities: Sequence[tuple[str, str]]
) -> Path:
    """Reject incomparable/failed/duplicate modes before using the shared bar plot."""

    if len(records) < 2:
        raise ValueError("at least two mode records are required for comparison")
    if any(record.status != "completed" for record in records):
        raise ValueError("failed records cannot enter a mode comparison")
    _same_identity(records)
    return plot_mode_comparison(sorted(records, key=lambda item: item.mode), output_path)


def _same_identity(records: Sequence[ResultRecord]) -> None:
    first = records[0]
    for record in records[1:]:
        if (record.seed, record.split_id, record.code_sha, record.dataset, record.model) != (
            first.seed, first.split_id, first.code_sha, first.dataset, first.model
        ):
            raise ValueError("records do not share comparison identity")
    if len(identities) != len(records) or len(set(identities)) != 1:
        raise ValueError("records do not share initial-state and budget identity")
