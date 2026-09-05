"""Tests for A's Day 2 command-line contract and the D3 prepare-data entry point."""

from pathlib import Path
from typing import Any

import yaml

from src.cli import main


def test_status_command(capsys: object) -> None:
    assert main(["status"]) == 0
    assert "D2 design review complete" in capsys.readouterr().out  # type: ignore[attr-defined]


def test_validate_default_config_bundle(capsys: object) -> None:
    assert main(["validate-config"]) == 0
    assert "Configuration valid" in capsys.readouterr().out  # type: ignore[attr-defined]


def test_validate_missing_config_returns_error(capsys: object) -> None:
    assert main(["validate-config", "--data", "missing.yaml"]) == 2
    assert "Configuration error" in capsys.readouterr().err  # type: ignore[attr-defined]


def test_unimplemented_command_is_not_silent_success(capsys: object) -> None:
    assert main(["train"]) == 2
    assert "not implemented yet" in capsys.readouterr().err  # type: ignore[attr-defined]


def _write_data_config(
    tmp_path: Path, config_bundle: dict[str, dict[str, Any]], **overrides: str
) -> Path:
    """Dump a validated data config copy with raw/processed paths redirected to tmp_path."""

    data_config = config_bundle["data"]
    data_config["dataset"]["raw_dir"] = overrides.get("raw_dir", str(tmp_path / "raw"))
    data_config["dataset"]["processed_dir"] = overrides.get(
        "processed_dir", str(tmp_path / "processed")
    )
    config_path = tmp_path / "data.yaml"
    config_path.write_text(yaml.safe_dump(data_config), encoding="utf-8")
    return config_path


def test_prepare_data_success_creates_output_root(
    tmp_path: Path, capsys: object, config_bundle: dict[str, dict[str, Any]]
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    processed_dir = tmp_path / "processed"
    config_path = _write_data_config(
        tmp_path, config_bundle, raw_dir=str(raw_dir), processed_dir=str(processed_dir)
    )

    assert main(["prepare-data", "--data", str(config_path)]) == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]

    assert processed_dir.is_dir()
    assert "prepare-data entry point ready" in captured.out
    assert str(processed_dir) in captured.out
    assert "highd" in captured.out


def test_prepare_data_output_flag_overrides_processed_dir(
    tmp_path: Path, capsys: object, config_bundle: dict[str, dict[str, Any]]
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    config_path = _write_data_config(tmp_path, config_bundle, raw_dir=str(raw_dir))
    override_output = tmp_path / "custom-output"

    assert (
        main(
            [
                "prepare-data",
                "--data",
                str(config_path),
                "--output",
                str(override_output),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()  # type: ignore[attr-defined]

    assert override_output.is_dir()
    assert str(override_output) in captured.out


def test_prepare_data_missing_config_file_returns_error(capsys: object) -> None:
    assert main(["prepare-data", "--data", "missing-data.yaml"]) == 2
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "Configuration error" in captured.err


def test_prepare_data_malformed_config_returns_error(
    tmp_path: Path, capsys: object
) -> None:
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    assert main(["prepare-data", "--data", str(config_path)]) == 2
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "Configuration error" in captured.err


def test_prepare_data_missing_input_directory_returns_error(
    tmp_path: Path, capsys: object, config_bundle: dict[str, dict[str, Any]]
) -> None:
    missing_raw = tmp_path / "does-not-exist"
    config_path = _write_data_config(tmp_path, config_bundle, raw_dir=str(missing_raw))

    assert main(["prepare-data", "--data", str(config_path)]) == 2
    captured = capsys.readouterr()  # type: ignore[attr-defined]

    assert "Data error" in captured.err
    assert "raw input directory" in captured.err
    assert str(missing_raw) in captured.err
    assert not (tmp_path / "processed").exists()
