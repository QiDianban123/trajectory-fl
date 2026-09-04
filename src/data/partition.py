"""Spatial RSU client partition: schema, region index, and group index.

Day 3 (S1-D-01) production slice: partition configuration, region
boundaries, group extents, and the ``rsu_<NN>`` client identifier scheme.
Train/validation/test disjointness is guaranteed upstream (split before
windowing); this module only ever receives train groups. Spatial Non-IID
assignment, merging of adjacent small regions, and per-client statistics
are added on Day 4 via :func:`partition_train_groups`.
"""

from __future__ import annotations

import bisect
import math
import numbers
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

Axis = Literal["x"]

DEFAULT_CLIENT_ID_PREFIX = "rsu_"
MANIFEST_SCHEMA_VERSION = 1
_SUPPORTED_AXES = ("x",)
_CONFIG_KEYS = frozenset(
    {"num_clients", "axis", "client_id_prefix", "region_edges", "min_samples_per_client"}
)


class PartitionError(ValueError):
    """A client partition configuration, index, or dataset violates the schema."""


def _validate_finite_number(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise PartitionError(f"{name} must be a finite number")
    if not math.isfinite(float(value)):
        raise PartitionError(f"{name} must be a finite number")


def _validate_edges(edges: Sequence[float], *, name: str = "region_edges") -> None:
    """Require at least two finite edges ordered as ``e[i] < e[i + 1]``."""
    if len(edges) < 2:
        raise PartitionError(f"{name} must contain at least two edges")
    previous = None
    for index, edge in enumerate(edges):
        _validate_finite_number(edge, f"{name}[{index}]")
        current = float(edge)
        if previous is not None and current <= previous:
            raise PartitionError(f"{name} must be strictly increasing")
        previous = current


def _sorted_groups(groups: Iterable[GroupExtent]) -> tuple[GroupExtent, ...]:
    """Return groups in a deterministic total order independent of input order."""
    return tuple(sorted(groups, key=_group_sort_key))


def _group_sort_key(group: GroupExtent) -> tuple[float, float, str]:
    return (group.x_min, group.x_max, repr(group.group_id))


def _client_id_width(num_clients: int) -> int:
    return max(2, len(str(num_clients)))


@dataclass(frozen=True)
class PartitionConfig:
    """Validated schema for the spatial Non-IID partition (``data.yaml``).

    ``region_edges`` is ``None`` by default, which selects equal-width
    regions over the observed longitudinal extent of the train groups.
    Explicit edges must span the whole dataset extent and contain exactly
    ``num_clients + 1`` strictly increasing values.
    """

    num_clients: int = 5
    axis: Axis = "x"
    client_id_prefix: str = DEFAULT_CLIENT_ID_PREFIX
    region_edges: tuple[float, ...] | None = None
    min_samples_per_client: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.num_clients, bool) or not isinstance(self.num_clients, int):
            raise PartitionError("num_clients must be a positive integer")
        if self.num_clients <= 0:
            raise PartitionError("num_clients must be a positive integer")
        if self.axis not in _SUPPORTED_AXES:
            raise PartitionError(f"axis must be one of {_SUPPORTED_AXES}")
        if not isinstance(self.client_id_prefix, str) or not self.client_id_prefix.strip():
            raise PartitionError("client_id_prefix must be a non-empty string")
        if (
            isinstance(self.min_samples_per_client, bool)
            or not isinstance(self.min_samples_per_client, int)
            or self.min_samples_per_client < 1
        ):
            raise PartitionError("min_samples_per_client must be a positive integer")
        if self.region_edges is not None:
            if not isinstance(self.region_edges, tuple):
                raise PartitionError("region_edges must be a tuple or None")
            if len(self.region_edges) != self.num_clients + 1:
                raise PartitionError(
                    "region_edges must contain exactly num_clients + 1 edges "
                    f"({self.num_clients + 1}), got {len(self.region_edges)}"
                )
            _validate_edges(self.region_edges)

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "PartitionConfig":
        """Build a validated config from the ``data.yaml`` partition mapping."""
        if not isinstance(mapping, Mapping):
            raise PartitionError("partition config must be a mapping")
        unknown = sorted(set(mapping) - _CONFIG_KEYS)
        if unknown:
            raise PartitionError(f"unknown partition config keys: {', '.join(unknown)}")
        region_edges = mapping.get("region_edges")
        if region_edges is not None:
            if (
                isinstance(region_edges, bool)
                or isinstance(region_edges, (str, bytes))
                or not isinstance(region_edges, Sequence)
            ):
                raise PartitionError("region_edges must be a list of numbers or null")
            region_edges = tuple(float(edge) for edge in region_edges)
        return cls(
            num_clients=mapping.get("num_clients", 5),
            axis=mapping.get("axis", "x"),
            client_id_prefix=mapping.get("client_id_prefix", DEFAULT_CLIENT_ID_PREFIX),
            region_edges=region_edges,
            min_samples_per_client=mapping.get("min_samples_per_client", 1),
        )


@dataclass(frozen=True)
class GroupExtent:
    """One train group's longitudinal span and window count.

    ``x_min``/``x_max`` use physical meter coordinates along the road
    longitudinal axis; a group with zero width is valid (a track at a
    single coordinate). ``sample_count`` is the number of train windows
    the group contributes and may be zero for a reserved but windowless
    vehicle.
    """

    group_id: str | int
    x_min: float
    x_max: float
    sample_count: int

    def __post_init__(self) -> None:
        if isinstance(self.group_id, bool) or not isinstance(self.group_id, (str, int)):
            raise PartitionError("group_id must be a string or integer")
        if isinstance(self.group_id, str) and not self.group_id.strip():
            raise PartitionError("group_id must be a non-empty string")
        _validate_finite_number(self.x_min, "x_min")
        _validate_finite_number(self.x_max, "x_max")
        if self.x_min > self.x_max:
            raise PartitionError("x_min must be less than or equal to x_max")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 0
        ):
            raise PartitionError("sample_count must be a non-negative integer")

    @property
    def midpoint(self) -> float:
        """Longitudinal anchor used for deterministic region assignment."""
        return (self.x_min + self.x_max) / 2.0


def build_group_index(groups: Iterable[GroupExtent]) -> tuple[GroupExtent, ...]:
    """Sort train groups deterministically and reject duplicate group IDs.

    The returned order does not depend on caller iteration order, which is
    required for stable manifest reconstruction on Day 4.
    """

    materialized = list(groups)
    seen: set[str | int] = set()
    for group in materialized:
        if not isinstance(group, GroupExtent):
            raise PartitionError("group index entries must be GroupExtent records")
        if group.group_id in seen:
            raise PartitionError(f"duplicate group {group.group_id!r} in group index")
        seen.add(group.group_id)
    return _sorted_groups(materialized)


def equal_width_edges(x_lo: float, x_hi: float, num_clients: int) -> tuple[float, ...]:
    """Build ``num_clients + 1`` edges splitting ``[x_lo, x_hi]`` evenly."""

    if isinstance(num_clients, bool) or not isinstance(num_clients, int) or num_clients < 1:
        raise PartitionError("num_clients must be a positive integer")
    _validate_finite_number(x_lo, "x_lo")
    _validate_finite_number(x_hi, "x_hi")
    if x_hi <= x_lo:
        raise PartitionError("equal-width edges require a positive longitudinal extent")
    width = (x_hi - x_lo) / num_clients
    edges = [x_lo + step * width for step in range(num_clients + 1)]
    edges[-1] = float(x_hi)
    return tuple(edges)


@dataclass(frozen=True)
class RegionIndex:
    """One-dimensional spatial regions with deterministic coordinate lookup.

    Regions use right-open intervals ``[edges[i], edges[i + 1])`` for every
    region except the last, which is closed on the right. A coordinate equal
    to an interior edge therefore belongs to the region on its right.
    """

    axis: Axis
    edges: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.axis not in _SUPPORTED_AXES:
            raise PartitionError(f"axis must be one of {_SUPPORTED_AXES}")
        _validate_edges(self.edges)

    @classmethod
    def from_edges(cls, edges: Sequence[float], *, axis: Axis = "x") -> "RegionIndex":
        """Build a region index from full region edges."""
        return cls(axis=axis, edges=tuple(float(edge) for edge in edges))

    @property
    def num_regions(self) -> int:
        return len(self.edges) - 1

    def region_bounds(self, region_index: int) -> tuple[float, float]:
        """Return the half-open ``(x_min, x_max]`` coverage of one region."""
        if isinstance(region_index, bool) or not isinstance(region_index, int):
            raise PartitionError("region_index must be an integer")
        if not 0 <= region_index < self.num_regions:
            raise PartitionError(f"region_index must be in [0, {self.num_regions})")
        return float(self.edges[region_index]), float(self.edges[region_index + 1])

    def region_for_coordinate(self, coordinate: float) -> int:
        """Map one longitudinal coordinate to its region index (edge -> right)."""
        _validate_finite_number(coordinate, "coordinate")
        if coordinate < self.edges[0] or coordinate > self.edges[-1]:
            raise PartitionError(
                f"coordinate {coordinate} is outside the region extent "
                f"[{self.edges[0]}, {self.edges[-1]}]"
            )
        index = bisect.bisect_right(self.edges, coordinate) - 1
        if index >= self.num_regions:
            index = self.num_regions - 1
        return index

    def client_id(self, region_index: int, prefix: str = DEFAULT_CLIENT_ID_PREFIX) -> str:
        """Return the ``rsu_<NN>`` identifier for one region of this index."""
        self.region_bounds(region_index)
        if not isinstance(prefix, str) or not prefix.strip():
            raise PartitionError("client_id_prefix must be a non-empty string")
        width = _client_id_width(self.num_regions)
        return f"{prefix}{region_index + 1:0{width}d}"


def region_occupancy(
    groups: Sequence[GroupExtent], region_index: RegionIndex
) -> tuple[int, ...]:
    """Count groups per region; empty regions are visible as zero entries.

    Regions with zero groups are candidates for the adjacent-region merge on
    Day 4 and are never silently dropped before the assignment stage.
    """

    counts = [0] * region_index.num_regions
    for group in groups:
        counts[region_index.region_for_coordinate(group.midpoint)] += 1
    return tuple(counts)
