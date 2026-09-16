"""Generate anonymous highD-style data and run the production three-mode matrix."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd
import yaml


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from src.experiments.mode_runner import run_three_mode_matrix
    from src.utils.config import validate_config_bundle

    token = uuid.uuid4().hex[:12]
    workspace = root / "outputs" / f"s3-three-mode-input-{token}"
    raw, processed = workspace / "raw", workspace / "processed"
    raw.mkdir(parents=True)
    _tracks(raw / "anonymous_tracks.csv")
    data_path = _data_config(root, workspace, raw, processed)
    prepared = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "prepare-data",
            "--data",
            str(data_path),
            "--raw-dir",
            str(raw.relative_to(root)),
            "--processed-dir",
            str(processed.relative_to(root)),
            "--output-root",
            str((workspace / "prepare-run").relative_to(root)),
            "--run-id",
            "prepare",
        ],
        cwd=root,
        check=False,
    )
    if prepared.returncode:
        return prepared.returncode
    splits = [path for path in processed.iterdir() if path.is_dir()]
    if len(splits) != 1:
        return 1
    try:
        bundle = validate_config_bundle(
            data_path,
            root / "configs/model.yaml",
            root / "configs/experiments/s3_three_mode_smoke.yaml",
        )
        result = run_three_mode_matrix(
            bundle,
            project_root=root,
            processed_dir=splits[0],
            output_root=root / "outputs",
            run_id=f"s3-smoke-{token}",
        )
    except Exception as exc:
        print(f"Three-mode smoke error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    manifests = {
        mode: root
        / "outputs"
        / f"s3-smoke-{token}-{mode}"
        / ("s3_manifest.json" if mode == "centralized" else "manifest.json")
        for mode in ("centralized", "local_only", "federated")
    }
    valid = result.exit_code == 0 and all(path.is_file() for path in manifests.values())
    if valid:
        payloads = [json.loads(path.read_text(encoding="utf-8")) for path in manifests.values()]
        valid = all(
            item["status"] == "completed" and item["fairness"]["comparable"] and item["summary"]
            for item in payloads
        )
    print(
        json.dumps(
            {
                "run_id": token,
                "status": result.status,
                "manifests": {
                    key: str(value.relative_to(root)) for key, value in manifests.items()
                },
            },
            sort_keys=True,
        )
    )
    return 0 if valid else 1


def _tracks(path: Path) -> None:
    rows = []
    for region in range(5):
        for member in range(6):
            vehicle = region * 100 + member + 1
            for frame in range(200):
                rows.append(
                    {
                        "Track ID": vehicle,
                        "Frame ID": frame,
                        "x Position": float(region * 1000 + frame / 10),
                        "y Position": float(region * 4 + member / 10),
                    }
                )
    pd.DataFrame(rows).to_csv(path, index=False)


def _data_config(root: Path, workspace: Path, raw: Path, processed: Path) -> Path:
    config = yaml.safe_load((root / "configs/data.yaml").read_text(encoding="utf-8"))
    config["dataset"]["raw_dir"] = raw.relative_to(root).as_posix()
    config["dataset"]["processed_dir"] = processed.relative_to(root).as_posix()
    path = workspace / "data.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
