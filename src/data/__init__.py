"""Dataset contracts, preprocessing safeguards, and adapters."""

from src.data.adapters import DatasetAdapter, TrajectorySample
from src.data.dataset import TrajectoryDataset
from src.data.partition import (
    ClientPartition,
    GroupExtent,
    PartitionConfig,
    PartitionError,
    PartitionManifest,
    RegionIndex,
    build_group_index,
    check_partition_invariants,
    equal_width_edges,
    partition_train_groups,
    region_occupancy,
)
from src.data.preprocess import TrainingCoordinateScaler, WindowSpec

__all__ = [
    "ClientPartition",
    "DatasetAdapter",
    "GroupExtent",
    "PartitionConfig",
    "PartitionError",
    "PartitionManifest",
    "RegionIndex",
    "TrajectoryDataset",
    "TrajectorySample",
    "TrainingCoordinateScaler",
    "WindowSpec",
    "build_group_index",
    "check_partition_invariants",
    "equal_width_edges",
    "partition_train_groups",
    "region_occupancy",
]
