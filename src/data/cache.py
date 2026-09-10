"""Deterministic identity helpers for processed trajectory data."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping


class DataCacheIdentityError(ValueError):
    """The data configuration cannot produce a stable cache identity."""


_SEMANTIC_DATASET_KEYS = (
    "name",
    "frame_rate_hz",
    "coordinate_system",
    "coordinate_unit",
    "required_columns",
)
_SEMANTIC_SECTIONS = ("sequence", "split", "partition", "normalization", "preprocessing")


def semantic_data_config(config: Mapping[str, object]) -> dict[str, object]:
    """Return only config values that can change processed sample semantics."""

    dataset = _mapping(config, "dataset")
    missing_dataset = [key for key in _SEMANTIC_DATASET_KEYS if key not in dataset]
    if missing_dataset:
        raise DataCacheIdentityError(
            "data.dataset is missing semantic keys: " + ", ".join(missing_dataset)
        )
    result: dict[str, object] = {
        "schema_version": config.get("schema_version"),
        "dataset": {key: dataset[key] for key in _SEMANTIC_DATASET_KEYS},
    }
    for section_name in _SEMANTIC_SECTIONS:
        result[section_name] = dict(_mapping(config, section_name))
    _canonical_json(result, "semantic data configuration")
    return result


def semantic_config_digest(config: Mapping[str, object]) -> str:
    """Hash the canonical semantic subset of a validated data config."""

    payload = _canonical_json(semantic_data_config(config), "semantic data configuration")
    return hashlib.sha256(payload).hexdigest()


def processed_cache_key(
    *, data_version: str, split_id: str, data_config: Mapping[str, object]
) -> str:
    """Build the stable processed-data cache key used by training readers."""

    for name, value in (("data_version", data_version), ("split_id", split_id)):
        if not isinstance(value, str) or not value.strip():
            raise DataCacheIdentityError(f"{name} must be a non-empty string")
    payload = _canonical_json(
        {
            "schema_version": 1,
            "data_version": data_version,
            "split_id": split_id,
            "semantic_config_digest": semantic_config_digest(data_config),
        },
        "processed cache identity",
    )
    return "processed-v1-" + hashlib.sha256(payload).hexdigest()


def _mapping(config: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = config.get(name)
    if not isinstance(value, Mapping):
        raise DataCacheIdentityError(f"data.{name} must be a mapping")
    return value


def _canonical_json(value: object, name: str) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DataCacheIdentityError(f"{name} must be JSON serializable: {exc}") from exc
    return text.encode("utf-8")
