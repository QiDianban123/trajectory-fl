"""Build and verify the complete S1 data pipeline with an anonymous sample."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace",
        type=Path,
        help="Artifact workspace inside the repository; defaults to outputs/s1-smoke-<UTC time>.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    workspace = _resolve_workspace(project_root, args.workspace)
    if workspace.exists():
        print(f"S1 smoke workspace already exists: {workspace}", file=sys.stderr)
        return 2

    raw_dir = workspace / "raw"
    processed_dir = workspace / "processed"
    run_root = workspace / "runs"
    raw_dir.mkdir(parents=True)
    _write_anonymous_tracks(raw_dir / "anonymous_tracks.csv")
    data_config = _write_data_config(project_root, workspace, raw_dir, processed_dir)

    command = [
        sys.executable,
        "-m",
        "src.cli",
        "prepare-data",
        "--data",
        str(data_config),
        "--raw-dir",
        _relative(raw_dir, project_root),
        "--processed-dir",
        _relative(processed_dir, project_root),
        "--output-root",
        _relative(run_root, project_root),
        "--run-id",
        "s1-smoke",
    ]
    completed = subprocess.run(command, cwd=project_root, check=False)
    if completed.returncode != 0:
        return completed.returncode

    try:
        summary = _verify_outputs(processed_dir, run_root / "s1-smoke")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"S1 smoke verification failed: {exc}", file=sys.stderr)
        return 1

    print(f"S1 smoke passed: workspace={_relative(workspace, project_root)}")
    print(f"Clients: {', '.join(summary['client_ids'])}")
    print(f"Samples: {summary['sample_counts']}")
    print("Split groups: disjoint")
    return 0


def _resolve_workspace(project_root: Path, requested: Path | None) -> Path:
    if requested is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        candidate = project_root / "outputs" / f"s1-smoke-{timestamp}"
    else:
        candidate = requested if requested.is_absolute() else project_root / requested
    resolved = candidate.resolve()
    if not resolved.is_relative_to(project_root):
        raise ValueError("S1 smoke workspace must be inside the repository")
    return resolved


def _write_anonymous_tracks(path: Path) -> None:
    rows: list[dict[str, float | int]] = []
    for region in range(5):
        for member in range(5):
            vehicle_id = region * 100 + member + 1
            for frame in range(200):
                rows.append(
                    {
                        "Track ID": vehicle_id,
                        "Frame ID": frame,
                        "x Position": float(region * 1000 + member * 10 + frame / 10),
                        "y Position": float(region * 4 + member / 10),
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_data_config(
    project_root: Path,
    workspace: Path,
    raw_dir: Path,
    processed_dir: Path,
) -> Path:
    config = yaml.safe_load((project_root / "configs" / "data.yaml").read_text(encoding="utf-8"))
    config["dataset"]["raw_dir"] = _relative(raw_dir, project_root)
    config["dataset"]["processed_dir"] = _relative(processed_dir, project_root)
    if config["partition"]["num_clients"] != 5:
        raise ValueError("S1 smoke requires the default five-client partition config")
    path = workspace / "data.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _verify_outputs(processed_root: Path, run_dir: Path) -> dict[str, object]:
    split_directories = [path for path in processed_root.iterdir() if path.is_dir()]
    if len(split_directories) != 1:
        raise ValueError("prepare-data must produce exactly one split directory")
    split_dir = split_directories[0]
    partition = json.loads((split_dir / "partition_manifest.json").read_text(encoding="utf-8"))
    client_ids = [client["client_id"] for client in partition["clients"]]
    if client_ids != ["rsu_01", "rsu_02", "rsu_03", "rsu_04", "rsu_05"]:
        raise ValueError(f"expected five stable RSU clients, got {client_ids}")

    split_groups: dict[str, set[tuple[str, str]]] = {}
    sample_counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        manifest = json.loads((split_dir / split / "manifest.json").read_text(encoding="utf-8"))
        metadata = manifest["metadata"]
        split_groups[split] = {
            (str(item["recording_id"]), str(item["vehicle_id"])) for item in metadata
        }
        sample_counts[split] = int(manifest["sample_count"])
        if not metadata:
            raise ValueError(f"{split} split contains no samples")
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    if not all(split_groups[left].isdisjoint(split_groups[right]) for left, right in pairs):
        raise ValueError("vehicle groups overlap across splits")

    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    processed_index = json.loads((run_dir / "processed_index.json").read_text(encoding="utf-8"))
    if run_manifest["artifacts"].get("processed_index") != "processed_index.json":
        raise ValueError("run manifest does not index processed artifacts")
    required_index = {"processed_dir", "split_manifest", "partition_manifest", "scaler", "splits"}
    if set(processed_index) != required_index:
        raise ValueError("processed index is incomplete")
    return {"client_ids": client_ids, "sample_counts": sample_counts}


def _relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
