"""Orchestration layer for the ``prepare-data`` entry point (S1-A-01).

The CLI owns argument parsing and process exit codes; this module owns the
ordered preparation steps and only composes frozen public interfaces
(config validation, and on D4 the adapter/batching/partition/manifest
boundaries). It must never implement highD parsing, cleaning, windowing, or
partitioning algorithms itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.utils.config import Config, load_and_validate


class DataPreparationError(ValueError):
    """A locatable data preparation failure, such as a missing input directory."""


@dataclass(frozen=True)
class PrepareDataResult:
    """Verified inputs and created artifact paths for one prepare-data run.

    D4 extends this with sample statistics and split/manifest artifact paths.
    """

    config_path: Path
    dataset_name: str
    raw_dir: Path
    processed_dir: Path
    history_steps: int
    future_steps: int
    stride: int
    split_ratios: tuple[float, float, float]
    split_seed: int


def prepare_data(
    config_path: str | Path, *, output_root: str | Path | None = None
) -> PrepareDataResult:
    """Validate the data config, verify inputs, and create the output root.

    Configuration problems raise :class:`ConfigError`; locatable data problems
    raise :class:`DataPreparationError`. Both are mapped to exit code 2 by the
    CLI, while unexpected errors are allowed to propagate as failures.
    """

    config: Config = load_and_validate(config_path, "data")
    dataset = config["dataset"]

    raw_dir = Path(dataset["raw_dir"])
    if not raw_dir.is_dir():
        raise DataPreparationError(
            f"raw input directory does not exist or is not a directory: {raw_dir}"
        )

    processed_dir = (
        Path(output_root)
        if output_root is not None
        else Path(dataset["processed_dir"])
    )
    processed_dir.mkdir(parents=True, exist_ok=True)

    sequence = config["sequence"]
    split = config["split"]
    return PrepareDataResult(
        config_path=Path(config_path),
        dataset_name=dataset["name"],
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        history_steps=sequence["history_steps"],
        future_steps=sequence["future_steps"],
        stride=sequence["stride"],
        split_ratios=(split["train"], split["validation"], split["test"]),
        split_seed=split["seed"],
    )
