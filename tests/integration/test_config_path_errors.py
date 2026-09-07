"""Configuration path error integration tests for S1-F-01 (AT-04).

Covers REQ-PATH-01 (missing file) and REQ-PATH-02 (empty / non-mapping YAML).
All file I/O uses the pytest ``tmp_path`` fixture so no real config files are
mutated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import ConfigError, load_and_validate, load_yaml

# ---------------------------------------------------------------------------
# REQ-PATH-01: missing configuration file
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["data", "model", "experiment"])
def test_missing_config_file_is_rejected(kind: str, tmp_path: Path) -> None:
    """A non-existent config path must raise ConfigError, not silently pass."""

    missing_path = tmp_path / "nonexistent.yaml"
    with pytest.raises(ConfigError, match="does not exist"):
        load_and_validate(missing_path, kind)


# ---------------------------------------------------------------------------
# REQ-PATH-02: empty or non-mapping YAML
# ---------------------------------------------------------------------------


def test_empty_yaml_is_rejected(tmp_path: Path) -> None:
    """An empty YAML document must raise ConfigError with a mapping message."""

    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_yaml(path)


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    """A YAML list root must raise ConfigError with a mapping message."""

    path = tmp_path / "list.yaml"
    path.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_yaml(path)
