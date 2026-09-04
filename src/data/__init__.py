"""Dataset contracts, preprocessing safeguards, and adapters."""

from src.data.adapters import DatasetAdapter, TrajectorySample
from src.data.dataset import TrajectoryDataset
from src.data.partition import (
    GroupExtent,
    PartitionConfig,
    PartitionError,
    RegionIndex,
    build_group_index,
    equal_width_edges,
    region_occupancy,
)
from src.data.preprocess import TrainingCoordinateScaler, WindowSpec

__all__ = [
    "DatasetAdapter",
    "GroupExtent",
    "PartitionConfig",
    "PartitionError",
    "RegionIndex",
    "TrajectoryDataset",
    "TrajectorySample",
    "TrainingCoordinateScaler",
    "WindowSpec",
    "build_group_index",
    "equal_width_edges",
    "region_occupancy",
]
