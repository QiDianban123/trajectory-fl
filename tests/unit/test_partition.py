"""Unit tests for the spatial RSU client partition (S1-D-01).

Day 3 scope: partition configuration, region boundaries, group index and
``rsu_<NN>`` identifiers. Day 4 assignment and merge tests are appended in
the same module so the acceptance view stays in one place.
"""

from __future__ import annotations

import pytest

from src.data.partition import (
    GroupExtent,
    PartitionConfig,
    PartitionError,
    RegionIndex,
    build_group_index,
    equal_width_edges,
    region_occupancy,
)


def _group(
    group_id: str | int,
    x_min: float,
    x_max: float,
    sample_count: int = 10,
) -> GroupExtent:
    return GroupExtent(group_id=group_id, x_min=x_min, x_max=x_max, sample_count=sample_count)


# --- PartitionConfig -------------------------------------------------------


def test_default_partition_config_matches_yaml_defaults() -> None:
    config = PartitionConfig()
    assert config.num_clients == 5
    assert config.axis == "x"
    assert config.client_id_prefix == "rsu_"
    assert config.region_edges is None
    assert config.min_samples_per_client == 1
    assert PartitionConfig.from_mapping({}) == config


def test_partition_config_from_mapping_converts_edge_list() -> None:
    config = PartitionConfig.from_mapping(
        {"num_clients": 3, "axis": "x", "region_edges": [0.0, 10.0, 20.0, 30.0]}
    )
    assert config.region_edges == (0.0, 10.0, 20.0, 30.0)


def test_partition_config_rejects_illegal_client_counts() -> None:
    for value in (0, -1, True, 2.5, "5", None):
        with pytest.raises(PartitionError, match="num_clients"):
            PartitionConfig(num_clients=value)  # type: ignore[arg-type]


def test_partition_config_rejects_illegal_minimum_samples() -> None:
    for value in (0, -1, True, 2.5, "1"):
        with pytest.raises(PartitionError, match="min_samples_per_client"):
            PartitionConfig(min_samples_per_client=value)  # type: ignore[arg-type]


def test_partition_config_requires_axis_prefix_and_edges_schema() -> None:
    with pytest.raises(PartitionError, match="axis"):
        PartitionConfig(axis="y")  # type: ignore[arg-type]
    with pytest.raises(PartitionError, match="client_id_prefix"):
        PartitionConfig(client_id_prefix="  ")
    with pytest.raises(PartitionError, match="num_clients \\+ 1"):
        PartitionConfig(num_clients=3, region_edges=(0.0, 10.0, 20.0))
    with pytest.raises(PartitionError, match="num_clients \\+ 1"):
        PartitionConfig(num_clients=3, region_edges=(0.0, 10.0, 20.0, 30.0, 40.0))


def test_partition_config_requires_strictly_increasing_edges() -> None:
    with pytest.raises(PartitionError, match="strictly increasing"):
        PartitionConfig(num_clients=3, region_edges=(0.0, 10.0, 10.0, 30.0))
    with pytest.raises(PartitionError, match="strictly increasing"):
        PartitionConfig(num_clients=3, region_edges=(30.0, 20.0, 10.0, 0.0))
    with pytest.raises(PartitionError, match="finite"):
        PartitionConfig(num_clients=3, region_edges=(0.0, 10.0, float("inf"), 30.0))


def test_partition_config_rejects_unknown_keys_and_text_edges() -> None:
    with pytest.raises(PartitionError, match="unknown partition config keys"):
        PartitionConfig.from_mapping({"num_clients": 5, "clients": 5})
    with pytest.raises(PartitionError, match="region_edges"):
        PartitionConfig.from_mapping({"num_clients": 5, "region_edges": "0,1,2,3,4,5"})


# --- GroupExtent and group index -------------------------------------------


def test_group_extent_validates_identity_span_and_sample_count() -> None:
    with pytest.raises(PartitionError, match="group_id"):
        _group(True, 0.0, 1.0)
    with pytest.raises(PartitionError, match="group_id"):
        _group("", 0.0, 1.0)
    with pytest.raises(PartitionError, match="x_min"):
        GroupExtent(group_id=1, x_min=5.0, x_max=2.0, sample_count=3)
    with pytest.raises(PartitionError, match="finite"):
        GroupExtent(group_id=1, x_min=float("nan"), x_max=2.0, sample_count=3)
    with pytest.raises(PartitionError, match="sample_count"):
        GroupExtent(group_id=1, x_min=0.0, x_max=2.0, sample_count=-1)


def test_group_extent_accepts_zero_width_and_zero_samples() -> None:
    group = _group(1, 5.0, 5.0, sample_count=0)
    assert group.midpoint == 5.0


def test_group_index_rejects_duplicate_groups() -> None:
    groups = [_group(101, 0.0, 5.0), _group(102, 6.0, 9.0), _group(101, 3.0, 7.0)]
    with pytest.raises(PartitionError, match="duplicate group"):
        build_group_index(groups)


def test_group_index_sort_is_independent_of_input_order() -> None:
    shuffled = [_group(3, 50.0, 60.0), _group(1, 0.0, 10.0), _group(2, 20.0, 30.0)]
    expected_ids = [1, 2, 3]
    assert [group.group_id for group in build_group_index(shuffled)] == expected_ids
    assert [group.group_id for group in build_group_index(list(reversed(shuffled)))] == expected_ids


def test_group_index_rejects_non_group_records() -> None:
    with pytest.raises(PartitionError, match="GroupExtent"):
        build_group_index([("101", 0.0, 1.0)])  # type: ignore[list-item]


# --- Equal-width edges and RegionIndex --------------------------------------


def test_equal_width_edges_split_extent_into_client_regions() -> None:
    edges = equal_width_edges(0.0, 100.0, 5)
    assert len(edges) == 6
    assert edges[0] == 0.0
    assert edges[-1] == 100.0
    assert all(edges[i + 1] - edges[i] == pytest.approx(20.0) for i in range(5))


def test_equal_width_edges_reject_degenerate_or_illegal_inputs() -> None:
    with pytest.raises(PartitionError, match="positive longitudinal extent"):
        equal_width_edges(5.0, 5.0, 5)
    with pytest.raises(PartitionError, match="positive longitudinal extent"):
        equal_width_edges(5.0, 4.0, 5)
    with pytest.raises(PartitionError, match="num_clients"):
        equal_width_edges(0.0, 100.0, 0)


def test_region_boundary_values_map_to_the_right_region() -> None:
    index = RegionIndex.from_edges([0.0, 10.0, 20.0, 30.0])
    assert index.num_regions == 3
    assert index.region_for_coordinate(0.0) == 0
    assert index.region_for_coordinate(5.0) == 0
    assert index.region_for_coordinate(9.999) == 0
    assert index.region_for_coordinate(10.0) == 1
    assert index.region_for_coordinate(19.999) == 1
    assert index.region_for_coordinate(20.0) == 2
    assert index.region_for_coordinate(29.0) == 2
    assert index.region_for_coordinate(30.0) == 2  # last region is closed on the right


def test_region_index_rejects_out_of_extent_or_non_finite_coordinates() -> None:
    index = RegionIndex.from_edges([0.0, 10.0, 20.0, 30.0])
    with pytest.raises(PartitionError, match="outside the region extent"):
        index.region_for_coordinate(-0.1)
    with pytest.raises(PartitionError, match="outside the region extent"):
        index.region_for_coordinate(30.1)
    with pytest.raises(PartitionError, match="finite"):
        index.region_for_coordinate(float("nan"))


def test_region_bounds_and_client_ids_use_rsu_prefix_with_width() -> None:
    index = RegionIndex.from_edges([0.0, 10.0, 20.0, 30.0])
    assert index.region_bounds(0) == (0.0, 10.0)
    assert index.region_bounds(2) == (20.0, 30.0)
    assert [index.client_id(i) for i in range(3)] == ["rsu_01", "rsu_02", "rsu_03"]
    assert index.client_id(1, prefix="rsu_") == "rsu_02"
    with pytest.raises(PartitionError, match="region_index"):
        index.region_bounds(-1)
    with pytest.raises(PartitionError, match="client_id_prefix"):
        index.client_id(0, prefix="")


def test_region_index_from_edges_requires_strict_order() -> None:
    with pytest.raises(PartitionError, match="strictly increasing"):
        RegionIndex.from_edges([0.0, 0.0, 10.0])


# --- Empty-region diagnostics ----------------------------------------------


def test_region_occupancy_reports_empty_regions_without_error() -> None:
    groups = [_group(101, 1.0, 2.0), _group(102, 9.0, 10.0)]
    index = RegionIndex.from_edges([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])
    assert region_occupancy(groups, index) == (1, 0, 0, 0, 1)
