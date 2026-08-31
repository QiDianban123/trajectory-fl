"""Smoke test for A's D1 command entry."""

from src.cli import main


def test_status_command(capsys: object) -> None:
    assert main(["status"]) == 0
    assert "D1 baseline ready" in capsys.readouterr().out  # type: ignore[attr-defined]
