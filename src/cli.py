"""Day 2 command-line contract and configuration validation entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils.config import ConfigError, validate_config_bundle

DEFAULT_DATA_CONFIG = Path("configs/data.yaml")
DEFAULT_MODEL_CONFIG = Path("configs/model.yaml")
DEFAULT_EXPERIMENT_CONFIG = Path("configs/experiments/smoke.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trajectory-FL experiment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status", help="show the current project baseline status")

    validate_parser = subparsers.add_parser(
        "validate-config", help="load and validate the Day 2 YAML configuration bundle"
    )
    validate_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_CONFIG)
    validate_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_CONFIG)
    validate_parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)

    for name, help_text in (
        ("prepare-data", "prepare the selected trajectory dataset (scheduled for D3)"),
        ("train", "run one training mode (scheduled for D6-D10)"),
        ("compare", "summarize three-mode results (scheduled for D11-D13)"),
    ):
        command_parser = subparsers.add_parser(name, help=help_text)
        command_parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        print("D2 architecture baseline ready: highD selected; YAML contracts are available.")
        return 0
    if args.command == "validate-config":
        try:
            bundle = validate_config_bundle(args.data, args.model, args.experiment)
        except ConfigError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        run = bundle["experiment"]["run"]
        print(f"Configuration valid: run={run['name']} mode={run['mode']} seed={run['seed']}")
        return 0
    if args.command in {"prepare-data", "train", "compare"}:
        print(
            f"Command '{args.command}' is defined by the D2 interface but is not implemented yet; "
            "see the project schedule.",
            file=sys.stderr,
        )
        return 2
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
