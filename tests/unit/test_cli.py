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


def test_unimplemented_command_is_not_silent_success(capsys: object) -> None:
    assert main(["train"]) == 2
    assert "not implemented yet" in capsys.readouterr().err  # type: ignore[attr-defined]
