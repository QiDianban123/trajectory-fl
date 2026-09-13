"""Tests for A's command-line contract."""

from pathlib import Path
from types import SimpleNamespace

from src.cli import main

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_status_command(capsys: object) -> None:
    assert main(["status"]) == 0
    assert "S3 three-mode CLI candidate" in capsys.readouterr().out  # type: ignore[attr-defined]


def test_validate_default_config_bundle(capsys: object) -> None:
    assert main(["validate-config"]) == 0
    assert "Configuration valid" in capsys.readouterr().out  # type: ignore[attr-defined]


def test_validate_missing_config_returns_error(capsys: object) -> None:
    assert main(["validate-config", "--data", "missing.yaml"]) == 2
    assert "Configuration error" in capsys.readouterr().err  # type: ignore[attr-defined]


def test_train_rejects_unknown_mode(capsys: object) -> None:
    assert main(["train", "--mode", "unknown"]) == 2
    assert "unsupported mode" in capsys.readouterr().err  # type: ignore[attr-defined]


def test_train_reports_missing_processed_data(
    tmp_path: Path, capsys: object, monkeypatch: object
) -> None:
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    assert (
        main(
            [
                "train",
                "--mode",
                "centralized",
                "--data",
                str(PROJECT_ROOT / "configs/data.yaml"),
                "--model",
                str(PROJECT_ROOT / "configs/model.yaml"),
                "--experiment",
                str(PROJECT_ROOT / "configs/experiments/smoke.yaml"),
                "--processed-dir",
                "missing-processed",
                "--output-root",
                "outputs",
                "--run-id",
                "missing-data-run",
            ]
        )
        == 2
    )
    error = capsys.readouterr().err  # type: ignore[attr-defined]
    assert "processed dataset directory does not exist" in error


def test_train_reports_corrupt_checkpoint(capsys: object, monkeypatch: object) -> None:
    def fail_run(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ValueError("cannot load checkpoint corrupt.pt")

    monkeypatch.setattr(  # type: ignore[attr-defined]
        "src.cli.CentralizedExperiment.run", fail_run
    )
    assert main(["train", "--mode", "centralized", "--resume-checkpoint", "corrupt.pt"]) == 2
    assert "cannot load checkpoint" in capsys.readouterr().err  # type: ignore[attr-defined]


def test_s3_mode_failure_returns_nonzero(capsys: object, monkeypatch: object) -> None:
    failed = SimpleNamespace(
        result=SimpleNamespace(
            run_id="failed", status="failed", exit_code=1, manifest_path=Path("manifest.json")
        )
    )
    monkeypatch.setattr("src.cli.run_mode", lambda *args, **kwargs: failed)  # type: ignore[attr-defined]
    assert (
        main(
            [
                "train",
                "--mode",
                "local_only",
                "--experiment",
                "configs/experiments/s3_local_only_smoke.yaml",
            ]
        )
        == 1
    )
    assert "status=failed" in capsys.readouterr().out  # type: ignore[attr-defined]
