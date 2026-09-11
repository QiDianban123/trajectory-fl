"""Day 2 command-line contract and configuration validation entry point."""

from __future__ import annotations

import argparse
import sys
import uuid
from copy import deepcopy
from pathlib import Path

from src.data.prepare import PrepareDataError, prepare_data
from src.experiments.centralized import CentralizedExperiment, CentralizedExperimentRequest
from src.utils.config import ConfigError, load_and_validate, validate_config_bundle

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
        "prepare-data", help="prepare the selected trajectory dataset"
    )
    prepare_parser.add_argument("--data", type=Path, default=DEFAULT_DATA_CONFIG)
    prepare_parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_CONFIG)
    prepare_parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    prepare_parser.add_argument("--raw-dir", type=Path)
    prepare_parser.add_argument("--processed-dir", type=Path)
    prepare_parser.add_argument("--output-root", type=Path)
    prepare_parser.add_argument("--run-id")

    train_parser = subparsers.add_parser("train", help="run one supported training mode")
    train_parser.add_argument("--mode", required=True)
    train_parser.add_argument("--data", type=Path)
    train_parser.add_argument("--model", type=Path)
    train_parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
    train_parser.add_argument("--processed-dir", type=Path)
    train_parser.add_argument("--output-root", type=Path)
    train_parser.add_argument("--run-id")
    train_parser.add_argument("--data-version")
    train_parser.add_argument("--split-id")
    train_parser.add_argument("--resume-checkpoint", type=Path)

    compare_parser = subparsers.add_parser("compare", help="summarize three-mode results")
    compare_parser.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_CONFIG)
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
            bundle = validate_config_bundle(args.data, args.model, args.experiment)
            prepared = prepare_data(
                bundle,
                project_root=Path.cwd(),
                raw_dir=args.raw_dir,
                processed_dir=args.processed_dir,
                output_root=args.output_root,
                run_id=args.run_id,
            )
        except (ConfigError, PrepareDataError) as exc:
            print(f"Prepare-data error: {exc}", file=sys.stderr)
            return 2
        print(f"Prepared highD data: split_id={prepared.split_id}")
        print(f"Samples: {dict(prepared.sample_counts)}")
        print(f"Processed data: {prepared.processed_dir}")
        print(f"Split manifest: {prepared.split_manifest_path}")
        print(f"Partition manifest: {prepared.partition_manifest_path}")
        print(f"Processed index: {prepared.processed_index_path}")
        print(f"Run manifest: {prepared.run_manifest_path}")
        return 0
    if args.command == "train":
        if args.mode != "centralized":
            print(
                f"Train error: unsupported mode {args.mode!r}; expected 'centralized'",
                file=sys.stderr,
            )
            return 2
        try:
            bundle = _load_train_bundle(args.data, args.model, args.experiment)
            dataset = bundle["data"]["dataset"]
            run = bundle["experiment"]["run"]
            processed_dir = (
                args.processed_dir if args.processed_dir is not None else dataset["processed_dir"]
            )
            output_root = args.output_root if args.output_root is not None else run["output_root"]
            run_id = args.run_id or f"centralized-{uuid.uuid4().hex[:12]}"
            effective_bundle = _effective_train_bundle(
                bundle,
                processed_dir=processed_dir,
                output_root=output_root,
                run_id=run_id,
                resume_checkpoint=args.resume_checkpoint,
            )
            result = CentralizedExperiment().run(
                CentralizedExperimentRequest(
                    config_bundle=effective_bundle,
                    processed_dir=processed_dir,
                    project_root=Path.cwd(),
                    run_id=run_id,
                    output_root=output_root,
                    expected_data_version=args.data_version,
                    expected_split_id=args.split_id,
                    resume_checkpoint=args.resume_checkpoint,
                )
            )
        except (
            ConfigError,
            FileNotFoundError,
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as exc:
            print(f"Train error: {exc}", file=sys.stderr)
            return 2
        print(f"run_id={result.record.run_id}")
        print(f"best_epoch={result.best_epoch} sample_count={result.record.sample_count}")
        print(f"ADE={result.record.ade:.6f}m FDE={result.record.fde:.6f}m")
        print(f"checkpoint={result.checkpoint_path}")
        print(f"metrics={result.output_dir / 'metrics.json'}")
        print(f"figures={result.output_dir / 'figures'}")
        return 0
    if args.command == "compare":
        print(
            f"Command '{args.command}' is defined by the D2 interface but is not implemented yet; "
            "see the project schedule.",
            file=sys.stderr,
        )
        return 2
    raise ValueError(f"Unsupported command: {args.command}")


def _load_train_bundle(
    data_path: Path | None, model_path: Path | None, experiment_path: Path
) -> dict[str, dict[str, object]]:
    experiment = load_and_validate(experiment_path, "experiment")
    references = experiment["configs"]
    resolved_data = data_path if data_path is not None else Path(references["data"])
    resolved_model = model_path if model_path is not None else Path(references["model"])
    return validate_config_bundle(resolved_data, resolved_model, experiment_path)


def _effective_train_bundle(
    bundle: dict[str, dict[str, object]],
    *,
    processed_dir: object,
    output_root: object,
    run_id: str,
    resume_checkpoint: Path | None,
) -> dict[str, dict[str, object]]:
    effective = deepcopy(bundle)
    dataset = effective["data"]["dataset"]
    run = effective["experiment"]["run"]
    assert isinstance(dataset, dict)
    assert isinstance(run, dict)
    dataset["processed_dir"] = str(processed_dir)
    run["mode"] = "centralized"
    run["output_root"] = str(output_root)
    effective["experiment"]["runtime"] = {
        "run_id": run_id,
        "processed_dir": str(processed_dir),
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
    }
    return effective


if __name__ == "__main__":
    raise SystemExit(main())
