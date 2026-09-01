"""Load and validate the frozen Day 2 YAML configuration contracts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

Config = dict[str, Any]
SUPPORTED_KINDS = ("data", "model", "experiment")
SUPPORTED_RUN_MODES = ("smoke", "centralized", "local_only", "federated")


class ConfigError(ValueError):
    """A configuration file is missing, malformed, or violates its schema."""


def load_yaml(path: str | Path) -> Config:
    """Load a YAML mapping and reject missing, empty, or non-mapping documents."""

    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Configuration file does not exist: {config_path}")

    try:
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot read YAML configuration {config_path}: {exc}") from exc

    if not isinstance(loaded, Mapping):
        raise ConfigError(f"Configuration root must be a mapping: {config_path}")
    return dict(loaded)


def validate_config(config: Mapping[str, Any], kind: str) -> Config:
    """Validate one configuration mapping and return a shallow normalized copy."""

    if kind not in SUPPORTED_KINDS:
        raise ConfigError(f"Unknown configuration kind {kind!r}; expected {SUPPORTED_KINDS}")

    validated = dict(config)
    _require_schema_version(validated)
    if kind == "data":
        _validate_data(validated)
    elif kind == "model":
        _validate_model(validated)
    else:
        _validate_experiment(validated)
    return validated


def load_and_validate(path: str | Path, kind: str) -> Config:
    """Load and validate one YAML file."""

    return validate_config(load_yaml(path), kind)


def validate_config_bundle(
    data_path: str | Path,
    model_path: str | Path,
    experiment_path: str | Path,
) -> dict[str, Config]:
    """Validate the three config files and their shared tensor dimensions."""

    bundle = {
        "data": load_and_validate(data_path, "data"),
        "model": load_and_validate(model_path, "model"),
        "experiment": load_and_validate(experiment_path, "experiment"),
    }
    sequence = bundle["data"]["sequence"]
    model = bundle["model"]["model"]
    for key in ("history_steps", "future_steps"):
        if sequence[key] != model[key]:
            raise ConfigError(
                f"data.sequence.{key} ({sequence[key]}) must equal "
                f"model.model.{key} ({model[key]})"
            )
    if sequence["coordinate_dimension"] != model["input_size"]:
        raise ConfigError("data coordinate dimension must equal model input_size")
    if model["input_size"] != model["output_size"]:
        raise ConfigError("trajectory model input_size and output_size must match")
    return bundle


def _require_schema_version(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != 1:
        raise ConfigError("schema_version must be integer 1")


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError(f"{name} must be a mapping")
    return value


def _require_keys(
    section: Mapping[str, Any], section_name: str, keys: tuple[str, ...]
) -> None:
    missing = [key for key in keys if key not in section]
    if missing:
        raise ConfigError(f"{section_name} is missing required keys: {', '.join(missing)}")


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> None:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{name} must be a {qualifier} integer")


def _positive_number(value: Any, name: str, *, allow_zero: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        valid = False
    else:
        valid = value >= 0 if allow_zero else value > 0
    if not valid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{name} must be a {qualifier} number")


def _non_empty_string(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")


def _validate_data(config: Mapping[str, Any]) -> None:
    dataset = _section(config, "dataset")
    sequence = _section(config, "sequence")
    split = _section(config, "split")
    normalization = _section(config, "normalization")

    _require_keys(
        dataset,
        "dataset",
        (
            "name",
            "raw_dir",
            "processed_dir",
            "sample_dir",
            "frame_rate_hz",
            "coordinate_system",
            "coordinate_unit",
            "required_columns",
        ),
    )
    if dataset["name"] != "highd":
        raise ConfigError("dataset.name must be 'highd' for the frozen P0 dataset")
    for key in ("raw_dir", "processed_dir", "sample_dir"):
        _non_empty_string(dataset[key], f"dataset.{key}")
    if dataset["coordinate_system"] != "highd_road_local":
        raise ConfigError("dataset.coordinate_system must be highd_road_local")
    if dataset["coordinate_unit"] != "meter":
        raise ConfigError("dataset.coordinate_unit must be meter")
    _positive_number(dataset["frame_rate_hz"], "dataset.frame_rate_hz")
    columns = dataset["required_columns"]
    if not isinstance(columns, list) or not all(isinstance(item, str) for item in columns):
        raise ConfigError("dataset.required_columns must be a list of strings")
    if not {"id", "frame", "x", "y"}.issubset(columns):
        raise ConfigError("dataset.required_columns must contain id, frame, x, and y")

    _require_keys(
        sequence,
        "sequence",
        ("history_steps", "future_steps", "stride", "coordinate_dimension"),
    )
    for key in ("history_steps", "future_steps", "stride", "coordinate_dimension"):
        _positive_int(sequence[key], f"sequence.{key}")
    if sequence["coordinate_dimension"] != 2:
        raise ConfigError("sequence.coordinate_dimension must be 2")

    _require_keys(
        split,
        "split",
        (
            "strategy",
            "group_by",
            "require_group_disjointness",
            "train",
            "validation",
            "test",
            "seed",
        ),
    )
    if split["strategy"] != "group_then_window":
        raise ConfigError("split.strategy must be group_then_window")
    if split["group_by"] not in ("vehicle_id", "scenario_id"):
        raise ConfigError("split.group_by must be vehicle_id or scenario_id")
    if split["require_group_disjointness"] is not True:
        raise ConfigError("split.require_group_disjointness must be true")
    _positive_int(split["seed"], "split.seed", allow_zero=True)
    ratios = [split[name] for name in ("train", "validation", "test")]
    for name, value in zip(("train", "validation", "test"), ratios, strict=True):
        _positive_number(value, f"split.{name}")
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ConfigError("split train/validation/test ratios must sum to 1")

    _require_keys(
        normalization,
        "normalization",
        ("method", "fit_split", "per_axis", "statistics_artifact"),
    )
    if normalization["method"] != "standard":
        raise ConfigError("normalization.method must be standard")
    if normalization["fit_split"] != "train":
        raise ConfigError("normalization.fit_split must be train to prevent leakage")
    if not isinstance(normalization["per_axis"], bool):
        raise ConfigError("normalization.per_axis must be boolean")
    _non_empty_string(normalization["statistics_artifact"], "normalization.statistics_artifact")

    preprocessing = _section(config, "preprocessing")
    _require_keys(
        preprocessing,
        "preprocessing",
        (
            "time_order",
            "missing_required_policy",
            "nonfinite_coordinate_policy",
            "duplicate_frame_policy",
            "minimum_track_frames",
        ),
    )
    expected_policies = {
        "time_order": "strict_increasing",
        "missing_required_policy": "reject_sample",
        "nonfinite_coordinate_policy": "reject_sample",
        "duplicate_frame_policy": "reject_track",
    }
    for key, expected in expected_policies.items():
        if preprocessing[key] != expected:
            raise ConfigError(f"preprocessing.{key} must be {expected}")
    _positive_int(preprocessing["minimum_track_frames"], "preprocessing.minimum_track_frames")


def _validate_model(config: Mapping[str, Any]) -> None:
    model = _section(config, "model")
    training = _section(config, "training")
    _require_keys(
        model,
        "model",
        (
            "name",
            "history_steps",
            "future_steps",
            "input_size",
            "output_size",
            "hidden_size",
            "num_layers",
            "dropout",
            "dtype",
            "coordinate_representation",
            "device_policy",
        ),
    )
    if model["name"] != "lstm_encoder_decoder":
        raise ConfigError("model.name must be lstm_encoder_decoder for the P0 baseline")
    integer_keys = (
        "history_steps",
        "future_steps",
        "input_size",
        "output_size",
        "hidden_size",
        "num_layers",
    )
    for key in integer_keys:
        _positive_int(model[key], f"model.{key}")
    if model["input_size"] != 2 or model["output_size"] != 2:
        raise ConfigError("model input_size and output_size must both be 2")
    _positive_number(model["dropout"], "model.dropout", allow_zero=True)
    if not 0 <= model["dropout"] < 1:
        raise ConfigError("model.dropout must be in [0, 1)")
    if model["dtype"] != "float32":
        raise ConfigError("model.dtype must be float32")
    if model["coordinate_representation"] != "absolute_position":
        raise ConfigError("model.coordinate_representation must be absolute_position")
    if model["device_policy"] != "trainer_managed":
        raise ConfigError("model.device_policy must be trainer_managed")

    _require_keys(
        training,
        "training",
        (
            "loss",
            "optimizer",
            "learning_rate",
            "batch_size",
            "epochs",
            "initialization",
            "checkpoint",
        ),
    )
    if training["loss"] != "mse" or training["optimizer"] != "adam":
        raise ConfigError("P0 training loss/optimizer must be mse/adam")
    _positive_number(training["learning_rate"], "training.learning_rate")
    _positive_int(training["batch_size"], "training.batch_size")
    _positive_int(training["epochs"], "training.epochs")

    initialization = _section(training, "initialization")
    _require_keys(
        initialization,
        "training.initialization",
        ("owner", "strategy", "share_initial_state"),
    )
    if initialization["owner"] != "experiment_runner":
        raise ConfigError("training.initialization.owner must be experiment_runner")
    if initialization["strategy"] != "xavier_uniform":
        raise ConfigError("training.initialization.strategy must be xavier_uniform")
    if initialization["share_initial_state"] is not True:
        raise ConfigError("training.initialization.share_initial_state must be true")

    checkpoint = _section(training, "checkpoint")
    _require_keys(checkpoint, "training.checkpoint", ("schema_version", "required_keys"))
    if checkpoint["schema_version"] != 1:
        raise ConfigError("training.checkpoint.schema_version must be 1")
    required_keys = checkpoint["required_keys"]
    expected_keys = {
        "schema_version",
        "model_state",
        "model_config",
        "seed",
        "epoch",
        "split_id",
        "metrics",
    }
    if not isinstance(required_keys, list) or set(required_keys) != expected_keys:
        raise ConfigError("training.checkpoint.required_keys must match the checkpoint contract")


def _validate_experiment(config: Mapping[str, Any]) -> None:
    run = _section(config, "run")
    configs = _section(config, "configs")
    execution = _section(config, "execution")
    _require_keys(run, "run", ("name", "mode", "seed", "output_root"))
    _non_empty_string(run["name"], "run.name")
    if run["mode"] not in SUPPORTED_RUN_MODES:
        raise ConfigError(f"run.mode must be one of {SUPPORTED_RUN_MODES}")
    _positive_int(run["seed"], "run.seed", allow_zero=True)
    _non_empty_string(run["output_root"], "run.output_root")

    _require_keys(configs, "configs", ("data", "model"))
    _non_empty_string(configs["data"], "configs.data")
    _non_empty_string(configs["model"], "configs.model")

    _require_keys(execution, "execution", ("device", "num_workers"))
    if execution["device"] not in ("cpu", "cuda", "auto"):
        raise ConfigError("execution.device must be cpu, cuda, or auto")
    _positive_int(execution["num_workers"], "execution.num_workers", allow_zero=True)
