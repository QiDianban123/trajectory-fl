"""Dataset contracts, preprocessing safeguards, and adapters."""

from src.data.adapters import DatasetAdapter, TrajectorySample
from src.data.cache import processed_cache_key, semantic_config_digest, semantic_data_config
from src.data.client_loading import (
    ClientDataBundle,
    ClientDataError,
    ClientDataLoaders,
    ClientDataProfile,
    create_client_dataloaders,
)
from src.data.dataset import TrajectoryDataset
from src.data.loading import (
    DataLoaderConfig,
    ProcessedDataBundle,
    ProcessedDataError,
    ProcessedDatasetReader,
    create_dataloaders,
)
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
    "ClientDataBundle",
    "ClientDataError",
    "ClientDataLoaders",
    "ClientDataProfile",
    "DatasetAdapter",
    "DataLoaderConfig",
    "GroupExtent",
    "PartitionConfig",
    "PartitionError",
    "PartitionManifest",
    "ProcessedDataBundle",
    "ProcessedDataError",
    "ProcessedDatasetReader",
    "RegionIndex",
    "TrajectoryDataset",
    "TrajectorySample",
    "TrainingCoordinateScaler",
    "WindowSpec",
    "build_group_index",
    "check_partition_invariants",
    "equal_width_edges",
    "partition_train_groups",
    "processed_cache_key",
    "region_occupancy",
    "semantic_config_digest",
    "semantic_data_config",
    "create_dataloaders",
    "create_client_dataloaders",
]
