"""Tests for A's Day 2 command-line contract."""

from src.cli import main


def test_status_command(capsys: object) -> None:
    assert main(["status"]) == 0
    assert "D2 design review complete" in capsys.readouterr().out  # type: ignore[attr-defined]


def test_validate_default_config_bundle(capsys: object) -> None:
    assert main(["validate-config"]) == 0
    assert "Configuration valid" in capsys.readouterr().out  # type: ignore[attr-defined]


def test_validate_missing_config_returns_error(capsys: object) -> None:
    assert main(["validate-config", "--data", "missing.yaml"]) == 2
    assert "Configuration error" in capsys.readouterr().err  # type: ignore[attr-defined]


def test_train_rejects_unknown_mode(capsys: object) -> None:
    assert main(["train", "--mode", "federated"]) == 2
    assert "unsupported mode" in capsys.readouterr().err  # type: ignore[attr-defined]


def test_train_reports_missing_processed_data(capsys: object) -> None:
    assert (
        main(
            [
                "train",
                "--mode",
                "centralized",
                "--processed-dir",
                "missing-processed",
                "--run-id",
                "missing-data-run",
            ]
        )
        == 2
    )
    assert "Train error" in capsys.readouterr().err  # type: ignore[attr-defined]
