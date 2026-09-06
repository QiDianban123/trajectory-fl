"""Data pipeline integration gate tests for S1-F-01 (AT-04).

Covers split group disjointness (REQ-SPLIT-01) and TrainingCoordinateScaler
leakage prevention, reversibility, and unfitted-use rejection
(REQ-SCALER-01/02/03). Every assertion maps to a data-pipeline invariant that
must fail loudly instead of silently dropping or corrupting samples.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.data.preprocess import TrainingCoordinateScaler, validate_split_assignments

# ---------------------------------------------------------------------------
# REQ-SPLIT-01: split group disjointness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("group_key", ["vehicle_id", "scenario_id"])
def test_valid_split_has_no_group_leakage(group_key: str) -> None:
    """Disjoint group assignments resolve to the expected split mapping."""

    assignments = [(101, "train"), (202, "validation"), (303, "test")]
    resolved = validate_split_assignments(assignments, group_key=group_key)
    assert resolved == {101: "train", 202: "validation", 303: "test"}


@pytest.mark.parametrize("group_key", ["vehicle_id", "scenario_id"])
def test_group_in_multiple_splits_is_rejected(group_key: str) -> None:
    """A group appearing in two splits must raise before window construction."""

    assignments = [(101, "train"), (101, "test")]
    with pytest.raises(ValueError, match="split groups before building windows"):
        validate_split_assignments(assignments, group_key=group_key)


# ---------------------------------------------------------------------------
# REQ-SCALER-01: scaler may only fit the train split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("split", ["validation", "test"])
def test_scaler_rejects_non_train_split(split: str) -> None:
    """Fitting the scaler on a non-train split must raise to prevent leakage."""

    coordinates = np.array([[[0.0, 2.0], [2.0, 6.0]]], dtype=np.float32)
    scaler = TrainingCoordinateScaler()
    with pytest.raises(ValueError, match="only be fitted on the train split"):
        scaler.fit(coordinates, split=split)


# ---------------------------------------------------------------------------
# REQ-SCALER-02: transform/inverse_transform round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("zero_variance", [True, False], ids=["zero_var", "nonzero_var"])
def test_scaler_transform_inverse_is_reversible(zero_variance: bool) -> None:
    """Physical coordinates must be recovered exactly after transform."""

    if zero_variance:
        coordinates = np.full((2, 2, 2), 5.0, dtype=np.float32)
    else:
        coordinates = np.array(
            [[[0.0, 2.0], [2.0, 6.0]], [[4.0, 8.0], [6.0, 10.0]]], dtype=np.float32
        )
    scaler = TrainingCoordinateScaler().fit(coordinates, split="train")
    normalized = scaler.transform(coordinates)
    recovered = scaler.inverse_transform(normalized)
    # float32 round-trip tolerance: atol covers values near zero where rtol is
    # ineffective; rtol covers the larger coordinates.
    assert np.allclose(recovered, coordinates, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# REQ-SCALER-03: unfitted scaler rejects use
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["transform", "inverse_transform"])
def test_unfitted_scaler_rejects_use(method: str) -> None:
    """Calling transform/inverse_transform before fit must raise RuntimeError."""

    scaler = TrainingCoordinateScaler()
    coordinates = np.array([[[0.0, 2.0], [2.0, 6.0]]], dtype=np.float32)
    with pytest.raises(RuntimeError, match="scaler must be fitted"):
        getattr(scaler, method)(coordinates)
