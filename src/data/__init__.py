"""Dataset contracts, preprocessing safeguards, and adapters."""

from src.data.adapters import DatasetAdapter, TrajectorySample
from src.data.dataset import TrajectoryDataset
from src.data.preprocess import TrainingCoordinateScaler, WindowSpec

__all__ = [
    "DatasetAdapter",
    "TrajectoryDataset",
    "TrajectorySample",
    "TrainingCoordinateScaler",
    "WindowSpec",
]
