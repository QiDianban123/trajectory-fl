"""Prepare anonymous data and run the centralized CLI in one command."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("outputs/s2-a-smoke"))
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    workspace = (root / args.workspace).resolve()
    if workspace.exists():
        print(f"Centralized smoke workspace already exists: {workspace}", file=sys.stderr)
        return 2
    prepared = subprocess.run(
        [sys.executable, "scripts/run_s1_smoke.py", "--workspace", str(workspace)],
        cwd=root,
        check=False,
    )
    if prepared.returncode != 0:
        return prepared.returncode
    model_config = _write_smoke_model(root, workspace)
    split_directories = [path for path in (workspace / "processed").iterdir() if path.is_dir()]
    if len(split_directories) != 1:
        print("Centralized smoke could not locate one processed split", file=sys.stderr)
        return 1
    command = [
        sys.executable,
        "-m",
        "src.cli",
        "train",
        "--mode",
        "centralized",
        "--processed-dir",
        str(split_directories[0].relative_to(root)),
        "--model",
        str(model_config.relative_to(root)),
        "--output-root",
        str((workspace / "centralized-runs").relative_to(root)),
        "--run-id",
        "centralized-smoke",
    ]
    completed = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        print(completed.stderr, file=sys.stderr, end="")
        return completed.returncode
    print(completed.stdout, end="")
    run_dir = workspace / "centralized-runs" / "centralized-smoke"
    required = [
        run_dir / "config_snapshot.json",
        run_dir / "metadata.json",
        run_dir / "train.log",
        run_dir / "checkpoints" / "best.pt",
        run_dir / "training_history.json",
        run_dir / "metrics.json",
        run_dir / "metrics.csv",
        run_dir / "figures" / "loss_curve.png",
        run_dir / "figures" / "prediction_trajectory.png",
    ]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        print("Centralized smoke artifact verification failed", file=sys.stderr)
        return 1
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    history = json.loads((run_dir / "training_history.json").read_text(encoding="utf-8"))
    losses = [
        entry["validation_loss"]
        if entry["validation_loss"] is not None
        else entry["train_loss"]
        for entry in history["epochs"]
    ]
    if min(losses) >= losses[0]:
        print("Centralized smoke loss did not decrease", file=sys.stderr)
        return 1
    print(f"Centralized smoke loss: {losses[0]:.6f} -> {min(losses):.6f}")
    print(f"Centralized smoke passed: ADE={metrics['metrics']['ade']:.6f}m")
    return 0


def _write_smoke_model(root: Path, workspace: Path) -> Path:
    config = yaml.safe_load((root / "configs/model.yaml").read_text(encoding="utf-8"))
    config["model"]["hidden_size"] = 8
    config["training"]["epochs"] = 8
    path = workspace / "model-smoke.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
