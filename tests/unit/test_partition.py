"""Unit tests for the spatial RSU client partition (S1-D-01).

Day 3 scope: partition configuration, region boundaries, group index and
``rsu_<NN>`` identifiers. Day 4 assignment and merge tests are appended in
the same module so the acceptance view stays in one place.
"""

from __future__ import annotations

import pytest

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


# --- Day 4: Non-IID assignment, merging, statistics and manifest ------------


def _spread_groups() -> list[GroupExtent]:
    """Ten groups whose midpoints 5..95 fill five equal-width 20 m bands."""
    return [
        _group(group_id, midpoint - 5.0, midpoint + 5.0, sample_count=6)
        for group_id, midpoint in enumerate(range(5, 100, 10), start=1)
    ]


def test_partition_assigns_every_group_to_exactly_one_client() -> None:
    groups = _spread_groups()
    manifest = partition_train_groups(groups, PartitionConfig(num_clients=5))
    assert manifest.num_clients == 5
    assert [client.client_id for client in manifest.clients] == [
        "rsu_01",
        "rsu_02",
        "rsu_03",
        "rsu_04",
        "rsu_05",
    ]
    expected = {group_id: f"rsu_{((group_id - 1) // 2) + 1:02d}" for group_id in range(1, 11)}
    assert manifest.assignment == expected
    check_partition_invariants(manifest, groups)
    assert manifest.totals() == (10, 60)


def test_partition_merges_adjacent_small_region_into_smaller_neighbour() -> None:
    config = PartitionConfig(
        num_clients=3, region_edges=(0.0, 10.0, 20.0, 30.0), min_samples_per_client=30
    )
    groups = (
        [_group(group_id, group_id * 1.5, group_id * 1.5 + 1.0, sample_count=6)
         for group_id in range(1, 6)]
        + [_group(6, 11.0, 19.0, sample_count=5)]
        + [_group(7 + offset, 21.0 + offset, 22.0 + offset, sample_count=10)
           for offset in range(4)]
    )
    manifest = partition_train_groups(groups, config)
    assert manifest.num_clients == 2
    first, second = manifest.clients
    assert (first.client_id, first.x_min, first.x_max, first.sample_count) == (
        "rsu_01",
        0.0,
        20.0,
        35,
    )
    assert (second.client_id, second.x_min, second.x_max, second.sample_count) == (
        "rsu_02",
        20.0,
        30.0,
        40,
    )
    assert all(client.sample_count >= config.min_samples_per_client for client in manifest.clients)
    assert manifest.totals() == (10, 75)
    check_partition_invariants(manifest, groups)


def test_partition_dissolves_empty_regions_without_empty_clients() -> None:
    config = PartitionConfig(num_clients=3, region_edges=(0.0, 10.0, 20.0, 30.0))
    groups = [_group(1, 1.0, 2.0, 6), _group(2, 3.0, 4.0, 6), _group(3, 21.0, 22.0, 12)]
    manifest = partition_train_groups(groups, config)
    assert manifest.num_clients == 2  # empty middle band merged into rsu_01
    assert manifest.totals() == (3, 24)
    assert all(client.group_ids for client in manifest.clients)
    check_partition_invariants(manifest, groups)


def test_partition_rejects_groups_outside_explicit_region_edges() -> None:
    config = PartitionConfig(num_clients=4, region_edges=(0.0, 10.0, 20.0, 30.0, 40.0))
    with pytest.raises(PartitionError, match="outside the configured region edges"):
        partition_train_groups([_group(1, 50.0, 60.0)], config)


@pytest.mark.parametrize("short_end", [20.0, 80.0])
def test_overlapping_groups_use_the_full_longitudinal_extent(short_end: float) -> None:
    groups = [_group("long", 0.0, 100.0), _group("short", 10.0, short_end)]
    manifest = partition_train_groups(groups, PartitionConfig())
    assert manifest.region_edges == (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
    assert manifest == partition_train_groups(reversed(groups), PartitionConfig())
    assert manifest.totals() == (2, 20)
    check_partition_invariants(manifest, groups)


def test_nested_group_outside_explicit_edges_is_rejected_even_with_midpoint_inside() -> None:
    groups = [_group("long", 0.0, 100.0), _group("short", 10.0, 20.0)]
    config = PartitionConfig(num_clients=2, region_edges=(0.0, 30.0, 60.0))
    with pytest.raises(PartitionError, match="outside the configured region edges"):
        partition_train_groups(groups, config)


@pytest.mark.parametrize(
    "zero_width,explicit_edges", [(False, False), (True, False), (False, True)]
)
def test_zero_window_inputs_are_rejected_before_partitioning(zero_width, explicit_edges) -> None:
    groups = (
        [_group(1, 5.0, 5.0, 0), _group(2, 5.0, 5.0, 0)]
        if zero_width
        else [_group(1, 0.0, 5.0, 0), _group(2, 10.0, 20.0, 0)]
    )
    config = PartitionConfig(
        region_edges=(0.0, 4.0, 8.0, 12.0, 16.0, 20.0) if explicit_edges else None
    )
    with pytest.raises(PartitionError, match="at least one effective train sample"):
        partition_train_groups(groups, config)


def test_reserved_zero_window_groups_remain_when_effective_samples_exist() -> None:
    groups = [_group("reserved", 0.0, 5.0, 0), _group("active", 10.0, 20.0, 3)]
    manifest = partition_train_groups(groups, PartitionConfig(min_samples_per_client=10))
    assert manifest.totals() == (2, 3)
    assert manifest.assignment == {"reserved": "rsu_01", "active": "rsu_01"}
    assert all(client.sample_count > 0 for client in manifest.clients)
    check_partition_invariants(manifest, groups)


def test_partition_invariants_reject_zero_sample_clients() -> None:
    groups = [_group("reserved", 0.0, 5.0, 0), _group("active", 10.0, 20.0, 3)]
    manifest = PartitionManifest(
        region_edges=(0.0, 10.0, 20.0),
        num_clients_requested=2,
        clients=(
            ClientPartition("rsu_01", 0.0, 10.0, 0, ("reserved",)),
            ClientPartition("rsu_02", 10.0, 20.0, 3, ("active",)),
        ),
    )
    with pytest.raises(PartitionError, match="no effective train samples"):
        check_partition_invariants(manifest, groups)


def test_partition_handles_zero_width_extent_as_single_client() -> None:
    groups = [_group(1, 50.0, 50.0, 4), _group(2, 50.0, 50.0, 4), _group(3, 50.0, 50.0, 4)]
    manifest = partition_train_groups(groups, PartitionConfig(num_clients=5))
    assert manifest.num_clients == 1
    assert manifest.clients[0].client_id == "rsu_01"
    assert manifest.region_edges == (50.0, 50.0)
    assert manifest.totals() == (3, 12)
    check_partition_invariants(manifest, groups)


def test_vehicle_midpoint_on_region_boundary_belongs_to_right_client() -> None:
    config = PartitionConfig(num_clients=4, region_edges=(0.0, 10.0, 20.0, 30.0, 40.0))
    groups = [_group(1, 19.0, 21.0, 10), _group(2, 2.0, 3.0, 10), _group(3, 35.0, 36.0, 10)]
    manifest = partition_train_groups(groups, config)
    assert manifest.assignment[1] == "rsu_02"
    assert manifest.assignment[2] == "rsu_01"
    assert manifest.assignment[3] == "rsu_03"
    middle = manifest.clients[1]
    assert (middle.x_min, middle.x_max) == (20.0, 30.0)
    check_partition_invariants(manifest, groups)


def test_partition_rebuilds_identically_regardless_of_input_order() -> None:
    groups = _spread_groups()
    config = PartitionConfig(num_clients=5)
    first = partition_train_groups(groups, config)
    second = partition_train_groups(list(reversed(groups)), config)
    third = partition_train_groups(groups, config)
    assert first == second
    assert first == third
    assert first.to_mapping() == second.to_mapping()


def test_manifest_mapping_reports_clients_and_statistics() -> None:
    manifest = partition_train_groups(_spread_groups(), PartitionConfig(num_clients=5))
    payload = manifest.to_mapping()
    assert payload["schema_version"] == 1
    assert payload["num_clients_requested"] == 5
    assert payload["num_clients"] == 5
    assert len(payload["region_edges"]) == 6
    assert [client["vehicle_count"] for client in payload["clients"]] == [2] * 5
    assert sum(client["sample_count"] for client in payload["clients"]) == 60
    assert all(len(client["group_ids"]) == client["vehicle_count"] for client in payload["clients"])


def test_partition_rejects_empty_train_group_sets() -> None:
    with pytest.raises(PartitionError, match="at least one train group"):
        partition_train_groups([], PartitionConfig())


def test_partition_invariants_reject_incomplete_foreign_or_wrong_counts() -> None:
    groups = [_group(1, 0.0, 5.0, 4), _group(2, 6.0, 9.0, 5)]
    manifest = partition_train_groups(
        groups, PartitionConfig(num_clients=2, region_edges=(0.0, 5.0, 10.0))
    )
    check_partition_invariants(manifest, groups)

    with pytest.raises(PartitionError, match="incomplete"):
        # "incomplete" means an input train group is absent from the union;
        # a group present only in the manifest is caught as foreign instead.
        check_partition_invariants(manifest, groups + [_group(3, 20.0, 25.0, 3)])

    foreign = PartitionManifest(
        num_clients_requested=1,
        region_edges=(0.0, 10.0),
        clients=(ClientPartition("rsu_01", 0.0, 10.0, 9, (1, 2, 999)),),
    )
    with pytest.raises(PartitionError, match="not present"):
        check_partition_invariants(foreign, groups)

    wrong_count = PartitionManifest(
        num_clients_requested=1,
        region_edges=(0.0, 10.0),
        clients=(ClientPartition("rsu_01", 0.0, 10.0, 99, (1, 2)),),
    )
    with pytest.raises(PartitionError, match="does not match"):
        check_partition_invariants(wrong_count, groups)
