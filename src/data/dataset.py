"""Dataset container that enforces the Day 2 standard-sample contract."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from src.data.adapters import SplitName, TrajectorySample
from src.data.preprocess import WindowSpec


class TrajectoryDataset(Sequence[TrajectorySample]):
    """An immutable sequence of one split's fixed-length trajectory samples.

    D4 may add a PyTorch-specific wrapper; this dependency-free container keeps
    split, window, and metadata validation available to every data consumer.
    """

    def __init__(
        self,
        samples: Sequence[TrajectorySample],
        *,
        split: SplitName,
        split_id: str,
        window_spec: WindowSpec,
    ) -> None:
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation, or test")
        if not split_id.strip():
            raise ValueError("split_id must be a non-empty string")
        self._samples = tuple(samples)
        self.split = split
        self.split_id = split_id
        self.window_spec = window_spec
        self._validate_samples()

    def __getitem__(self, index: int) -> TrajectorySample:
        return self._samples[index]

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[TrajectorySample]:
        return iter(self._samples)

    def _validate_samples(self) -> None:
        for sample in self._samples:
            if sample.meta["split"] != self.split:
                raise ValueError("all samples must belong to the dataset split")
            if sample.meta["split_id"] != self.split_id:
                raise ValueError("all samples must use the dataset split_id")
            if sample.history.shape != (self.window_spec.history_steps, 2):
                raise ValueError("sample history shape does not match window_spec")
            if sample.future.shape != (self.window_spec.future_steps, 2):
                raise ValueError("sample future shape does not match window_spec")
