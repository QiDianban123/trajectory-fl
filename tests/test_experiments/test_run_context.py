import json
import pytest
from pathlib import Path
from src.experiments import RunContext, file_checksum


# ==================== D3 测试 ====================

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


# ==================== D4 测试 ====================

def test_file_checksum(tmp_path):
    test_file = tmp_path / "test.txt"
    test_file.write_text("hello", encoding="utf-8")
    
    cs1 = file_checksum(test_file)
    assert len(cs1) == 64
    
    test_file.write_text("world", encoding="utf-8")
    cs2 = file_checksum(test_file)
    assert cs1 != cs2


def test_record_data_file(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    fake_file = data_dir / "train.csv"
    fake_file.write_text("x,y\n1,2\n3,4\n")
    
    ctx = RunContext(config={"seed": 42}, project_root=tmp_path)
    ctx.record_data_file(fake_file)
    
    manifest = ctx.get_manifest()
    assert "data/train.csv" in manifest["data_files"]
    assert len(manifest["data_files"]["data/train.csv"]) == 64


def test_record_missing_file_raises(tmp_path):
    ctx = RunContext(config={"seed": 42}, project_root=tmp_path)
    missing = tmp_path / "data" / "missing.csv"
    
    with pytest.raises(FileNotFoundError):
        ctx.record_data_file(missing)


def test_manifest_deterministic_ordering(tmp_path):
    ctx = RunContext(config={"seed": 42}, project_root=tmp_path)
    
    (tmp_path / "a.csv").write_text("a")
    (tmp_path / "b.csv").write_text("b")
    
    ctx.record_data_file(tmp_path / "a.csv")
    ctx.record_data_file(tmp_path / "b.csv")
    
    manifest = ctx.get_manifest()
    keys = list(manifest["data_files"].keys())
    assert keys == ["a.csv", "b.csv"]


def test_export_data_profile(tmp_path):
    ctx = RunContext(config={"seed": 42}, project_root=tmp_path)
    
    profile = {
        "total_samples": 1000,
        "train_samples": 700,
        "val_samples": 200,
        "test_samples": 100,
    }
    
    path = ctx.export_data_profile(profile)
    assert path.exists()
    
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["total_samples"] == 1000