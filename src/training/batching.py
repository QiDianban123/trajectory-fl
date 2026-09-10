"""The explicit NumPy sample to CPU Torch batch boundary for training."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy

import numpy as np

from src.data.adapters import TrajectorySample
from src.models.base import ModelContract, require_torch
from src.training.trainer import TrajectoryBatch


def sample_to_tensor(sample: TrajectorySample, *, contract: ModelContract) -> TrajectoryBatch:
    """Convert one sample to a batch of size one using the shared collate boundary."""

    return collate_trajectory_samples([sample], contract=contract)


def collate_trajectory_samples(
    samples: Sequence[TrajectorySample], *, contract: ModelContract
) -> TrajectoryBatch:
    """Build a finite, batch-first float32 CPU batch without changing coordinates.

    Bind ``contract`` with ``functools.partial`` for a DataLoader ``collate_fn``.
    All samples must share split and split_id. Arrays and nested metadata are
    copied so training cannot mutate the Dataset through the resulting batch.
    Device transfer remains the Trainer's responsibility, including with workers.
    """

    torch = require_torch()
    if not samples:
        raise ValueError("cannot collate an empty batch")
    for index, sample in enumerate(samples):
        if not isinstance(sample, TrajectorySample):
            raise TypeError(f"samples[{index}] must be a TrajectorySample")
        # Frozen dataclasses still contain mutable NumPy arrays and metadata.
        # Reuse the data contract to catch mutations since sample construction.
        try:
            TrajectorySample(sample.history, sample.future, sample.meta)
        except (TypeError, ValueError) as error:
            raise type(error)(f"samples[{index}]: {error}") from error
        if sample.history.shape != (contract.history_steps, contract.input_size):
            raise ValueError(f"samples[{index}] history shape does not match ModelContract")
        if sample.future.shape != (contract.future_steps, contract.output_size):
            raise ValueError(f"samples[{index}] future shape does not match ModelContract")
        for key in ("split", "split_id"):
            if sample.meta[key] != samples[0].meta[key]:
                raise ValueError(f"samples[{index}] {key} must match the batch {key}")

    # Stacking copies even read-only / negative-stride inputs into owned storage;
    # from_numpy always creates CPU tensors, regardless of Torch's default device.
    batch = TrajectoryBatch(
        history=torch.from_numpy(np.stack([sample.history for sample in samples])),
        future=torch.from_numpy(np.stack([sample.future for sample in samples])),
        meta=tuple(deepcopy(dict(sample.meta)) for sample in samples),
    )
    batch.validate(contract)
    return batch
