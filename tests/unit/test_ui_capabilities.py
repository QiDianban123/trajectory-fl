"""S2 UI command whitelist and path-boundary tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.ui.capabilities import (
    UiCommandError,
    build_centralized_train_command,
    build_smoke_command,
    s2_capabilities,
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
