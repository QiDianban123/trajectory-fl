"""S1-C-01: conversion boundaries and failures before model execution."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from src.data.adapters import TrajectorySample
from src.models import base
from src.models.base import ModelContract
from src.training import TrajectoryBatch, collate_trajectory_samples, sample_to_tensor

CONTRACT = ModelContract(history_steps=2, future_steps=3)


@pytest.fixture
def sample() -> TrajectorySample:
    return TrajectorySample(
        history=np.arange(4, dtype=np.float32).reshape(2, 2),
        future=np.arange(4, 10, dtype=np.float32).reshape(3, 2),
        meta={
            "dataset_name": "highd",
            "data_version": "synthetic-s1-c",
            "recording_id": 1,
            "vehicle_id": 101,
            "history_start_frame": 0,
            "history_end_frame": 1,
            "future_start_frame": 2,
            "future_end_frame": 4,
            "split_id": "synthetic-split",
            "split": "train",
            "client_id": "rsu_00",
            "extra": {"tags": ["normalized"]},
        },
    )


@pytest.mark.parametrize("size", [1, 3])
def test_conversion_preserves_values_dtype_and_metadata(sample, size):
    batch = collate_trajectory_samples([sample] * size, contract=CONTRACT)
    batch.validate(CONTRACT)
    assert batch.history.shape == (size, 2, 2)
    assert batch.future.shape == (size, 3, 2)
    assert batch.history.dtype == batch.future.dtype == torch.float32
    assert batch.history.device == batch.future.device == torch.device("cpu")
    np.testing.assert_array_equal(batch.history.numpy(), np.stack([sample.history] * size))
    np.testing.assert_array_equal(batch.future.numpy(), np.stack([sample.future] * size))
    assert batch.meta == (sample.meta,) * size


def test_single_sample_and_batch_storage_are_independent(sample):
    batch = sample_to_tensor(sample, contract=CONTRACT)
    batch.history[0, 0, 0] = -999
    batch.future[0, 0, 0] = -999
    batch.meta[0]["extra"]["tags"].append("changed")
    assert sample.history[0, 0] == 0
    assert sample.future[0, 0] == 4
    assert sample.meta["extra"]["tags"] == ["normalized"]
    assert batch.history.shape == (1, 2, 2)


@pytest.mark.parametrize("field", ["history", "future"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_mutated_nonfinite_sample_is_rejected(sample, field, value):
    getattr(sample, field)[0, 0] = value
    with pytest.raises(ValueError, match=f"samples\\[0\\].*{field}.*finite"):
        sample_to_tensor(sample, contract=CONTRACT)


@pytest.mark.parametrize("field", ["history", "future"])
@pytest.mark.parametrize("dtype", [np.float64, np.int32, np.bool_])
def test_mutated_wrong_dtype_is_rejected_without_casting(sample, field, dtype):
    object.__setattr__(sample, field, getattr(sample, field).astype(dtype))
    with pytest.raises(TypeError, match=f"{field}.*float32"):
        sample_to_tensor(sample, contract=CONTRACT)


@pytest.mark.parametrize("field", ["history", "future"])
@pytest.mark.parametrize("shape", [(0, 2), (2,), (2, 3), (1, 2, 2), (4, 2)])
def test_wrong_shape_is_rejected_before_stacking(sample, field, shape):
    object.__setattr__(sample, field, np.zeros(shape, dtype=np.float32))
    with pytest.raises(ValueError, match=f"{field}.*shape"):
        sample_to_tensor(sample, contract=CONTRACT)


def test_empty_batch_and_wrong_sample_type_are_rejected():
    with pytest.raises(ValueError, match="empty batch"):
        collate_trajectory_samples([], contract=CONTRACT)
    with pytest.raises(TypeError, match="TrajectorySample"):
        collate_trajectory_samples([{}], contract=CONTRACT)


@pytest.mark.parametrize("field", ["history", "future"])
def test_non_numpy_coordinates_are_rejected(sample, field):
    object.__setattr__(sample, field, getattr(sample, field).tolist())
    with pytest.raises(TypeError, match="numpy.ndarray"):
        sample_to_tensor(sample, contract=CONTRACT)


@pytest.mark.parametrize("key,value", [("split", "test"), ("split_id", "other")])
def test_mixed_split_provenance_is_rejected(sample, key, value):
    other = replace(sample, meta={**sample.meta, key: value})
    with pytest.raises(ValueError, match=f"samples\\[1\\] {key} must match"):
        collate_trajectory_samples([sample, other], contract=CONTRACT)


def test_mutated_required_metadata_is_revalidated(sample):
    sample.meta.pop("split_id")
    with pytest.raises(ValueError, match="missing required keys: split_id"):
        sample_to_tensor(sample, contract=CONTRACT)


def test_readonly_negative_stride_arrays_are_supported(sample):
    history = sample.history[:, ::-1]
    future = sample.future[:, ::-1]
    history.setflags(write=False)
    future.setflags(write=False)
    batch = sample_to_tensor(replace(sample, history=history, future=future), contract=CONTRACT)
    np.testing.assert_array_equal(batch.history[0].numpy(), history)
    np.testing.assert_array_equal(batch.future[0].numpy(), future)


def test_default_device_does_not_move_collated_tensors(sample):
    with torch.device("meta"):
        batch = sample_to_tensor(sample, contract=CONTRACT)
    assert batch.history.device == batch.future.device == torch.device("cpu")


def test_missing_torch_has_dependency_install_guidance(sample, monkeypatch):
    monkeypatch.setattr(base, "torch", None)
    with pytest.raises(RuntimeError, match="requirements.txt"):
        sample_to_tensor(sample, contract=CONTRACT)


@pytest.mark.parametrize("future", [None, 0, torch.tensor(1.0)])
def test_invalid_future_has_contract_error_instead_of_index_error(sample, future):
    batch = sample_to_tensor(sample, contract=CONTRACT)
    with pytest.raises(ValueError, match="future"):
        replace(batch, future=future).validate(CONTRACT)


def test_device_mismatch_is_rejected_before_tensor_value_checks(sample):
    batch = sample_to_tensor(sample, contract=CONTRACT)
    with pytest.raises(ValueError, match="same device"):
        replace(batch, future=torch.empty((1, 3, 2), device="meta")).validate(CONTRACT)
    with pytest.raises(ValueError, match="materialized values"):
        TrajectoryBatch(
            torch.empty((1, 2, 2), device="meta"), torch.empty((1, 3, 2), device="meta")
        ).validate(CONTRACT)


@pytest.mark.parametrize("meta", [({}, {}), ("invalid",)])
def test_invalid_batch_metadata_is_rejected(sample, meta):
    batch = sample_to_tensor(sample, contract=CONTRACT)
    with pytest.raises(ValueError, match="one mapping per sample"):
        replace(batch, meta=meta).validate(CONTRACT)
