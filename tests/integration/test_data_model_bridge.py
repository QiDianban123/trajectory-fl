"""S1-C-01 / AT-01: existing Dataset -> DataLoader -> frozen model contract."""

from functools import partial

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.adapters import TrajectorySample
from src.data.dataset import TrajectoryDataset
from src.data.preprocess import WindowSpec
from src.models.base import ModelContract, validate_history_tensor, validate_prediction_tensor
from src.training import collate_trajectory_samples


@pytest.mark.parametrize("split", ["train", "validation", "test"])
@pytest.mark.parametrize("num_workers", [0, 1])
def test_dataset_loader_preserves_contract_order_and_partial_batch(
    config_bundle, split, num_workers
):
    contract = ModelContract.from_model_config(config_bundle["model"]["model"])
    assert (contract.history_steps, contract.future_steps) == (75, 125)
    samples = [
        TrajectorySample(
            history=np.full((75, 2), index + 0.25, dtype=np.float32),
            future=np.full((125, 2), index + 0.75, dtype=np.float32),
            meta={
                "dataset_name": "highd",
                "data_version": "synthetic-s1-c",
                "recording_id": 1,
                "vehicle_id": index,
                "history_start_frame": 0,
                "history_end_frame": 74,
                "future_start_frame": 75,
                "future_end_frame": 199,
                "split_id": "synthetic-split",
                "split": split,
                "client_id": f"rsu_{index:02d}",
            },
        )
        for index in range(3)
    ]
    dataset = TrajectoryDataset(
        samples,
        split=split,
        split_id="synthetic-split",
        window_spec=WindowSpec(history_steps=75, future_steps=125, stride=1),
    )
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=partial(collate_trajectory_samples, contract=contract),
        **({"multiprocessing_context": "spawn"} if num_workers else {}),
    )
    batches = list(loader)
    assert [batch.history.shape[0] for batch in batches] == [2, 1]
    for batch in batches:
        batch.validate(contract)
        validate_history_tensor(batch.history, contract)
        # Shape-only output double; S1 does not implement the LSTM or train it.
        prediction = torch.zeros_like(batch.future)
        validate_prediction_tensor(prediction, batch.history, contract)
        assert batch.history.dtype == batch.future.dtype == torch.float32
        assert batch.history.device == batch.future.device == torch.device("cpu")
    np.testing.assert_array_equal(
        torch.cat([batch.history for batch in batches]).numpy(),
        np.stack([sample.history for sample in samples]),
    )
    np.testing.assert_array_equal(
        torch.cat([batch.future for batch in batches]).numpy(),
        np.stack([sample.future for sample in samples]),
    )
    assert [meta for batch in batches for meta in batch.meta] == [sample.meta for sample in samples]
