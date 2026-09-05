"""Day 2 command-line contract and configuration validation entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.data.pipeline import DataPreparationError, prepare_data
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

    prepare_parser = subparsers.add_parser(
        "prepare-data", help="prepare the selected trajectory dataset (S1-A-01 entry point)"
    )
    prepare_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_CONFIG)
    prepare_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="processed output root; defaults to dataset.processed_dir from the data config",
    )

    for name, help_text in (
        ("train", "run one training mode (scheduled for D6-D10)"),
        ("compare", "summarize three-mode results (scheduled for D11-D13)"),
    ):
        command_parser = subparsers.add_parser(name, help=help_text)
        command_parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "status":
        print("D2 design review complete: highD selected; core interfaces are frozen for D3.")
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
    if args.command == "prepare-data":
        try:
            result = prepare_data(args.data, output_root=args.output)
        except ConfigError as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 2
        except DataPreparationError as exc:
            print(f"Data error: {exc}", file=sys.stderr)
            return 2
        print(
            "prepare-data entry point ready:\n"
            f"  dataset={result.dataset_name} config={result.config_path}\n"
            f"  raw input: {result.raw_dir}\n"
            f"  processed output root: {result.processed_dir}\n"
            f"  window: history={result.history_steps} future={result.future_steps} "
            f"stride={result.stride}\n"
            f"  split: train={result.split_ratios[0]} validation={result.split_ratios[1]} "
            f"test={result.split_ratios[2]} seed={result.split_seed}"
        )
        return 0
    if args.command in {"train", "compare"}:
        print(
            f"Command '{args.command}' is defined by the D2 interface but is not implemented yet; "
            "see the project schedule.",
            file=sys.stderr,
        )
        return 2
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
