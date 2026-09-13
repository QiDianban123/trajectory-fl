"""S2 UI command whitelist and path-boundary tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ui.capabilities import (
    UiCommandError,
    build_centralized_train_command,
    build_s3_train_command,
    build_smoke_command,
    preflight_s3_fairness,
    s2_capabilities,
    s3_capabilities,
)


def _project(tmp_path: Path) -> Path:
    for relative_path in (
        "configs/data.yaml",
        "configs/model.yaml",
        "configs/experiments/smoke.yaml",
        "outputs/processed/split/split_manifest.json",
    ):
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    return tmp_path


def test_s2_only_enables_centralized_capabilities() -> None:
    capabilities = {capability.key: capability for capability in s2_capabilities()}
    assert capabilities["centralized_smoke"].enabled
    assert capabilities["centralized_train"].enabled
    assert not capabilities["local_only"].enabled
    assert not capabilities["federated"].enabled


def test_train_command_is_argument_array_and_rejects_unsafe_inputs(tmp_path: Path) -> None:
    root = _project(tmp_path)
    spec = build_centralized_train_command(
        root,
        data_config="configs/data.yaml",
        model_config="configs/model.yaml",
        experiment_config="configs/experiments/smoke.yaml",
        processed_dir="outputs/processed/split",
        output_root="outputs",
        run_id="ui-run-42",
        seed=42,
        epochs=2,
        batch_size=4,
    )
    assert spec.argv[:4] == (spec.argv[0], "-m", "src.cli", "train")
    assert "--mode" in spec.argv and "centralized" in spec.argv
    assert ";" not in spec.preview

    with pytest.raises(UiCommandError, match="unsafe"):
        build_centralized_train_command(
            root,
            data_config="../data.yaml",
            model_config="configs/model.yaml",
            experiment_config="configs/experiments/smoke.yaml",
            processed_dir="outputs/processed/split",
            output_root="outputs",
            run_id="ui-run-42",
            seed=42,
            epochs=1,
            batch_size=1,
        )
    with pytest.raises(UiCommandError, match="run_id"):
        build_centralized_train_command(
            root,
            data_config="configs/data.yaml",
            model_config="configs/model.yaml",
            experiment_config="configs/experiments/smoke.yaml",
            processed_dir="outputs/processed/split",
            output_root="outputs",
            run_id="../escape",
            seed=42,
            epochs=1,
            batch_size=1,
        )


def test_smoke_rejects_existing_workspace(tmp_path: Path) -> None:
    root = _project(tmp_path)
    with pytest.raises(UiCommandError, match="already exists"):
        build_smoke_command(root, "outputs/processed")


def test_s3_capabilities_and_preflight_reject_inconsistent_config(tmp_path: Path) -> None:
    root = _project(tmp_path)
    path = root / "configs/experiments/s3_federated_smoke.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "run: {mode: federated, seed: 1}\n"
        "three_mode: {mode_matrix: [centralized, local_only, federated], rounds: 1, "
        "local_epochs: 1, clients_per_round: 5, metric_schema: ade_fde_meter_v1, "
        "recovery: {boundary: complete_client_or_round}}\n",
        encoding="utf-8",
    )
    assert {item.key for item in s3_capabilities()} >= {
        "local_only_train",
        "federated_train",
        "resume",
    }
    assert preflight_s3_fairness(
        root, mode="federated", experiment_config="configs/experiments/s3_federated_smoke.yaml"
    ).allowed
    assert not preflight_s3_fairness(
        root, mode="local_only", experiment_config="configs/experiments/s3_federated_smoke.yaml"
    ).allowed


def test_s3_train_rejects_escape_and_resume_without_recovery(tmp_path: Path) -> None:
    root = _project(tmp_path)
    for mode in ("local_only", "federated", "centralized"):
        path = root / f"configs/experiments/s3_{mode}_smoke.yaml"
        path.write_text(
            f"run: {{mode: {mode}, seed: 1}}\n"
            "three_mode: {mode_matrix: [centralized, local_only, federated], rounds: 1, "
            "local_epochs: 1, clients_per_round: 5, metric_schema: ade_fde_meter_v1, "
            "recovery: {boundary: complete_client_or_round}}\n",
            encoding="utf-8",
        )
    with pytest.raises(UiCommandError):
        build_s3_train_command(root, mode="federated", processed_dir="../escape", run_id="safe")
    with pytest.raises(UiCommandError, match="recovery"):
        build_s3_train_command(
            root,
            mode="federated",
            processed_dir="outputs/processed/split",
            run_id="safe",
            resume=True,
        )
