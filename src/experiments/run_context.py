"""
RunContext: 实验运行上下文，记录配置、代码版本、输出目录和 manifest。
"""

import hashlib
import json
import re
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def file_checksum(filepath: Path) -> str:
    """计算文件的 SHA256 校验值。"""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class RunContext:
    """
    为每次数据准备或实验创建统一的运行上下文。
    
    输出目录结构：
        outputs/<run_id>/
            config_snapshot.json
            manifest.json
            data_profile.json
            ... (其他产物由调用方写入)
    """

    _RUN_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
    _TIMESTAMP_FMT = "%Y%m%d_%H%M%S"

    def __init__(
        self,
        config: Dict[str, Any],
        project_root: Path,
        run_id: Optional[str] = None,
    ):
        """
        Args:
            config: 本次运行的配置字典，必须可 JSON 序列化。
            project_root: 项目根目录路径。
            run_id: 自定义 run_id；None 则自动生成。
        
        Raises:
            ValueError: run_id 非法、配置不可序列化、或路径逃逸。
            FileExistsError: run_id 已存在，防止覆盖。
            RuntimeError: 无法获取 git SHA。
        """
        self.project_root = Path(project_root).resolve()
        self.config = config

        # 1. 生成/验证 run_id
        if run_id is None:
            self.run_id = self._generate_run_id()
        else:
            self.run_id = self._validate_run_id(run_id)

        # 2. 路径逃逸检查
        self.output_dir = self.project_root / "outputs" / self.run_id
        self._check_path_escape(self.output_dir)

        # 3. 检查重复 run
        if self.output_dir.exists():
            raise FileExistsError(
                f"Run '{self.run_id}' already exists at {self.output_dir}. "
                "Use a different run_id or delete the existing directory."
            )

        # 4. 获取 git SHA
        self.git_sha = self._get_git_sha()

        # 5. 配置 JSON 序列化检查
        try:
            self._config_json = json.dumps(config, indent=2, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Config is not JSON serializable: {exc}") from exc

        # 6. 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=False)

        # 7. 保存配置快照
        self._save_config_snapshot()

        # 8. 初始化 manifest
        self._manifest: Dict[str, Any] = {
            "run_id": self.run_id,
            "git_sha": self.git_sha,
            "created_at": datetime.now().isoformat(),
            "config": config,
            "output_dir": str(self.output_dir.relative_to(self.project_root)),
            "data_files": {},
            "splits": {},
        }

    # ------------------------------------------------------------------ #
    # 内部方法
    # ------------------------------------------------------------------ #
    def _generate_run_id(self) -> str:
        timestamp = datetime.now().strftime(self._TIMESTAMP_FMT)
        short_uuid = uuid.uuid4().hex[:6]
        return f"{timestamp}_{short_uuid}"

    def _validate_run_id(self, run_id: str) -> str:
        if not run_id:
            raise ValueError("run_id cannot be empty.")
        if not self._RUN_ID_PATTERN.match(run_id):
            raise ValueError(
                f"run_id '{run_id}' contains illegal characters. "
                "Only alphanumeric, underscore, and hyphen are allowed."
            )
        return run_id

    def _check_path_escape(self, path: Path) -> None:
        try:
            path.resolve().relative_to(self.project_root.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Path '{path}' escapes project root '{self.project_root}'."
            ) from exc

    def _get_git_sha(self) -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            raise RuntimeError(
                f"Failed to retrieve git SHA from {self.project_root}: {exc}"
            ) from exc

    def _save_config_snapshot(self) -> None:
        config_path = self.output_dir / "config_snapshot.json"
        config_path.write_text(self._config_json, encoding="utf-8")

    # ------------------------------------------------------------------ #
    # D3 公共接口
    # ------------------------------------------------------------------ #
    def add_data_file(self, relative_path: str, checksum: str) -> None:
        """记录一个数据文件及其 SHA256 校验值。"""
        self._manifest["data_files"][relative_path] = checksum

    def add_split_manifest(self, split_id: str, file_manifests: list) -> None:
        """记录某个 split（如 train/val/test 或 rsu_01）的文件清单。"""
        self._manifest["splits"][split_id] = file_manifests

    def export_manifest(self) -> Path:
        """将 manifest 写入 outputs/<run_id>/manifest.json，返回路径。"""
        manifest_path = self.output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(self._manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return manifest_path

    def get_manifest(self) -> Dict[str, Any]:
        """返回当前 manifest 字典（不写入文件）。"""
        return self._manifest.copy()

    # ------------------------------------------------------------------ #
    # D4 新增公共接口
    # ------------------------------------------------------------------ #
    def record_data_file(self, absolute_path: Path) -> None:
        """记录一个数据文件（自动计算 checksum）。"""
        absolute_path = Path(absolute_path)
        if not absolute_path.exists():
            raise FileNotFoundError(f"Data file not found: {absolute_path}")
        checksum = file_checksum(absolute_path)
        relative_path = str(absolute_path.relative_to(self.project_root)).replace("\\", "/")
        self.add_data_file(relative_path, checksum)

    def export_data_profile(self, profile: dict) -> Path:
        """导出数据画像 JSON。"""
        profile_path = self.output_dir / "data_profile.json"
        profile_path.write_text(
            json.dumps(profile, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return profile_path