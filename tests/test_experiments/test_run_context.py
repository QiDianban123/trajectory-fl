import json
import pytest
from pathlib import Path
from src.experiments import RunContext


def test_run_context_creates_output_dir(tmp_path):
    config = {"seed": 42, "data": "highD"}
    ctx = RunContext(config=config, project_root=tmp_path)
    
    assert ctx.output_dir.exists()
    assert (ctx.output_dir / "config_snapshot.json").exists()
    assert ctx.git_sha is not None
    assert len(ctx.git_sha) == 40


def test_invalid_run_id_rejected(tmp_path):
    with pytest.raises(ValueError):
        RunContext(config={}, project_root=tmp_path, run_id="")
    
    with pytest.raises(ValueError):
        RunContext(config={}, project_root=tmp_path, run_id="a/b")
    
    with pytest.raises(ValueError):
        RunContext(config={}, project_root=tmp_path, run_id="a..b")


def test_duplicate_run_id_raises(tmp_path):
    config = {"seed": 42}
    RunContext(config=config, project_root=tmp_path, run_id="test_dup")
    
    with pytest.raises(FileExistsError):
        RunContext(config=config, project_root=tmp_path, run_id="test_dup")


def test_path_escape_detected(tmp_path):
    with pytest.raises(ValueError):
        RunContext(
            config={},
            project_root=tmp_path,
            run_id="../../../tmp",
        )


def test_non_serializable_config_raises(tmp_path):
    with pytest.raises(ValueError):
        RunContext(config={"func": lambda x: x}, project_root=tmp_path)


def test_manifest_export(tmp_path):
    config = {"seed": 42}
    ctx = RunContext(config=config, project_root=tmp_path)
    
    ctx.add_data_file("data/train.csv", "abc123")
    ctx.add_split_manifest("train", [{"file": "train.csv", "checksum": "abc123"}])
    
    manifest_path = ctx.export_manifest()
    assert manifest_path.exists()
    
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert data["run_id"] == ctx.run_id
    assert data["git_sha"] == ctx.git_sha
    assert data["config"] == config
    assert "data_files" in data
    assert "splits" in data