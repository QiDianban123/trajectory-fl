"""D1 numerical contract tests owned by E, maintained with F's test skeleton."""

import numpy as np
import pytest

from src.evaluation.metrics import ade, fde


def test_ade_and_fde_on_known_coordinates(tiny_trajectory_pair: tuple[np.ndarray, np.ndarray]) -> None:
    prediction, truth = tiny_trajectory_pair
    assert ade(prediction, truth) == pytest.approx(2.5)
    assert fde(prediction, truth) == pytest.approx(5.0)


def test_metrics_reject_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="does not match"):
        ade(np.zeros((2, 2)), np.zeros((3, 2)))


def test_metrics_reject_nan() -> None:
    with pytest.raises(ValueError, match="finite"):
        fde(np.array([[float("nan"), 0.0]]), np.zeros((1, 2)))
