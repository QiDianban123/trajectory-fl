from pathlib import Path

import pytest

from src.evaluation.federated_results import (
    ClientResultRecord,
    ComparisonIdentity,
    RoundRecord,
    compare_modes,
    plot_round_metrics,
    summarize_client_results,
)
from src.evaluation.result_store import ResultRecord


def _record(mode="local_only", samples=2, ade=1.0, fde=2.0, seed=7):
    return ResultRecord(
        run_id="run", code_sha="sha", seed=seed, split_id="split", mode=mode,
        sample_count=samples, ade=ade, fde=fde, total_seconds=1.0,
    )


def test_client_macro_and_weighted_use_evaluation_counts() -> None:
    macro, weighted = summarize_client_results(
        [
            ClientResultRecord(_record(samples=2, ade=1, fde=2), "rsu_01", 5, 15),
            ClientResultRecord(_record(samples=8, ade=3, fde=6), "rsu_02", 5, 15),
        ]
    )
    assert (macro.ade, macro.fde, macro.sample_count) == (2.0, 4.0, 10)
    assert macro.run_id == "run:macro"
    assert weighted.run_id == "run:weighted"
    assert weighted.ade == pytest.approx(2.6)
    assert weighted.fde == pytest.approx(5.2)


def test_round_and_mode_plot_reject_failures_and_identity_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="failed rounds"):
        plot_round_metrics([RoundRecord(1, "failed", None, None, None)], tmp_path)
    paths = plot_round_metrics([RoundRecord(0, "completed", 1, 2, 3)], tmp_path)
    assert all(path.is_file() for path in paths)
    with pytest.raises(ValueError, match="comparison identity"):
        compare_modes(
            [_record("centralized"), _record("federated", seed=8)],
            tmp_path / "x.png",
            identities=[_identity(), _identity()],
        )
    with pytest.raises(ValueError, match="initial-state and budget"):
        compare_modes(
            [_record("centralized"), _record("federated")],
            tmp_path / "x.png",
            identities=[_identity(), _identity(initial_state_id="other")],
        )
    failed = ResultRecord(
        **{**_record("federated").__dict__, "status": "failed", "error": "client failed"}
    )
    with pytest.raises(ValueError, match="failed records"):
        compare_modes(
            [_record("centralized"), failed],
            tmp_path / "x.png",
            identities=[_identity(), _identity()],
        )
    with pytest.raises(ValueError, match="at most one"):
        compare_modes(
            [_record("centralized"), _record("centralized", ade=2)],
            tmp_path / "x.png",
            identities=[_identity(), _identity()],
        )


def _identity(**overrides: str) -> ComparisonIdentity:
    values = {
        "partition_id": "partition",
        "scaler_id": "scaler",
        "model_config_digest": "model",
        "initial_state_id": "initial",
        "metric_schema": "metric-v1",
        "budget_id": "budget",
    }
    values.update(overrides)
    return ComparisonIdentity(**values)
