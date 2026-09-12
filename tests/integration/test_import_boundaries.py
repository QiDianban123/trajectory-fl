"""Fresh-process import checks for package boundaries."""

from __future__ import annotations

import subprocess
import sys


def test_torch_trainer_imports_without_prior_data_package_import() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", "from src.training.torch_trainer import TorchTrainer"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
