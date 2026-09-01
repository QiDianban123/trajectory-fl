"""Small, dependency-light fixtures shared by Day 2 unit and integration tests."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from src.utils.config import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tiny_trajectory_pair() -> tuple[np.ndarray, np.ndarray]:
    truth = np.array([[[0.0, 0.0], [3.0, 4.0]]])
    prediction = np.array([[[0.0, 0.0], [0.0, 0.0]]])
    return prediction, truth


@pytest.fixture
def tiny_trajectory_triplet() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Physical-coordinate history, truth, and prediction suitable for plotting."""

    history = np.array([[-1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    future = np.array([[1.0, 0.0], [2.0, 1.0]], dtype=np.float32)
    prediction = np.array([[1.0, 0.2], [2.0, 1.2]], dtype=np.float32)
    return history, future, prediction


@pytest.fixture
def config_bundle() -> dict[str, dict[str, object]]:
    """Independent copies of the repository's validated configuration sources."""

    paths = {
        "data": PROJECT_ROOT / "configs/data.yaml",
        "model": PROJECT_ROOT / "configs/model.yaml",
        "experiment": PROJECT_ROOT / "configs/experiments/smoke.yaml",
    }
    return {name: deepcopy(load_yaml(path)) for name, path in paths.items()}


@pytest.fixture
def temporary_output_root(tmp_path: Path) -> Path:
    """An isolated output root for path, logging, and result-store tests."""

    return tmp_path / "outputs"
