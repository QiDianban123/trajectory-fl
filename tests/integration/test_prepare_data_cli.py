"""Prepare-data CLI integration tests for S1-F-01 (AT-04).

Covers REQ-PREP-01 (prepare-data returns exit code 2 with a not-implemented
message) and REQ-PREP-02 (the --config argument is accepted and parsed).
"""

from __future__ import annotations

from pathlib import Path

from src.cli import build_parser, main

# ---------------------------------------------------------------------------
# REQ-PREP-01: prepare-data is not yet implemented
# ---------------------------------------------------------------------------


def test_prepare_data_returns_exit_code_2(capsys: object) -> None:
    """The prepare-data command must fail loudly with exit code 2."""

    exit_code = main(["prepare-data"])
    assert exit_code == 2
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "not implemented" in captured.err


# ---------------------------------------------------------------------------
# REQ-PREP-02: --config argument is accepted
# ---------------------------------------------------------------------------


def test_prepare_data_accepts_config_argument() -> None:
    """The --config argument must be parsed into a Path value."""

    custom_args = build_parser().parse_args(["prepare-data", "--config", "custom.yaml"])
    assert custom_args.config == Path("custom.yaml")

    default_args = build_parser().parse_args(["prepare-data"])
    assert default_args.config == Path("configs/experiments/smoke.yaml")
