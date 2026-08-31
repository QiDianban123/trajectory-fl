"""Shared pytest fixtures will be added as real dataset samples arrive on D3."""

import numpy as np
import pytest


@pytest.fixture
def tiny_trajectory_pair() -> tuple[np.ndarray, np.ndarray]:
    truth = np.array([[[0.0, 0.0], [3.0, 4.0]]])
    prediction = np.array([[[0.0, 0.0], [0.0, 0.0]]])
    return prediction, truth
