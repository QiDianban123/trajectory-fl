"""Run one reproducible centralized experiment against processed data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.experiments.centralized import CentralizedExperiment, CentralizedExperimentRequest
from src.utils.config import validate_config_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--data-config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model.yaml"))
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("configs/experiments/smoke.yaml"),
    )
    args = parser.parse_args(argv)
    bundle = validate_config_bundle(
        args.data_config,
        args.model_config,
        args.experiment_config,
    )
    output = CentralizedExperiment().run(
        CentralizedExperimentRequest(
            config_bundle=bundle,
            processed_dir=args.processed_dir,
            project_root=args.project_root,
            run_id=args.run_id,
            output_root=args.output_root,
        )
    )
    print(
        json.dumps(
            {
                "run_id": output.record.run_id,
                "loss": output.loss,
                "ade": output.record.ade,
                "fde": output.record.fde,
                "baseline_ade": output.baseline_record.ade,
                "baseline_fde": output.baseline_record.fde,
                "output_dir": str(output.output_dir),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
