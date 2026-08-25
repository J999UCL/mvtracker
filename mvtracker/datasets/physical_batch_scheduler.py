"""Deterministic physical batching for planned multi-view scenes.

The scheduler is deliberately independent of dataset materialisation and model
execution.  It receives eight already-planned scene summaries and partitions
them into two synchronized rank waves.  A physical group contains at most two
logical scenes with matching view/frame/image dimensions; trajectory counts
are padded and masked after materialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence


Resolution = tuple[int, int]
ShapeKey = tuple[int, int, Resolution]


@dataclass(frozen=True)
class SceneSummary:
    """Metadata required to schedule one planned scene."""

    source: str
    scene: str
    cursor: int
    view_count: int
    frame_count: int
    resolution: Resolution
    track_count: int
    schedule_start: int = 0

    @property
    def shape_key(self) -> ShapeKey:
        return (
            self.view_count,
            self.frame_count,
            self.resolution,
        )


@dataclass(frozen=True)
class BatchCapacity:
    """Static physical-batching rules for one hardware configuration.

    The capacity names the profiled hardware and the maximum scene group size.
    ``pair_track_capacity_by_views`` records the profiled trajectory ceilings
    for observability; trajectory counts do not split otherwise-compatible
    samples into separate physical batches.  Views in ``singleton_only_views``
    can never be paired.
    """

    name: str
    rank_count: int
    logical_scenes_per_rank: int
    max_group_size: int
    pair_track_capacity_by_views: tuple[tuple[int, int], ...]
    singleton_only_views: frozenset[int]

    def pair_track_capacity(self, view_count: int) -> int | None:
        return dict(self.pair_track_capacity_by_views).get(view_count)


# These are explicit planning rules, rather than a runtime memory probe or an
# OOM/retry policy.  Keep the table visible so a new GPU profile can be added
# without changing the scheduler algorithm.
H100_BATCH_CAPACITY = BatchCapacity(
    name="h100",
    rank_count=2,
    logical_scenes_per_rank=4,
    max_group_size=2,
    pair_track_capacity_by_views=((1, 1024), (2, 1024), (3, 1280), (4, 1024)),
    singleton_only_views=frozenset({5, 6}),
)


@dataclass(frozen=True)
class PhysicalBatchGroup:
    """One physical model batch and its logical scene members."""

    scenes: tuple[SceneSummary, ...]

    def __post_init__(self) -> None:
        if not 1 <= len(self.scenes) <= 2:
            raise ValueError("physical groups must contain one or two scenes")
        if len({scene.shape_key for scene in self.scenes}) != 1:
            raise ValueError("all scenes in a physical group must share shape")

    @property
    def shape_key(self) -> ShapeKey:
        return self.scenes[0].shape_key

    @property
    def view_count(self) -> int:
        return self.shape_key[0]

    @property
    def frame_count(self) -> int:
        return self.shape_key[1]

    @property
    def resolution(self) -> Resolution:
        return self.shape_key[2]

    @property
    def max_track_count(self) -> int:
        return max(scene.track_count for scene in self.scenes)

    @property
    def padded_track_count(self) -> int:
        return sum(self.max_track_count - scene.track_count for scene in self.scenes)

    @property
    def work(self) -> int:
        """Stable work estimate used only to balance the two ranks.

        The estimate counts per-scene image volume plus trajectory work and
        charges trajectory padding to the physical group.  It is intentionally
        integer-valued and does not inspect tensors or query the GPU.
        """

        view_frames_pixels = self.view_count * self.frame_count
        height, width = self.resolution
        image_work = view_frames_pixels * height * width * len(self.scenes)
        track_work = view_frames_pixels * (
            self.max_track_count * len(self.scenes)
        )
        return image_work + track_work


@dataclass(frozen=True)
class RankWave:
    rank: int
    groups: tuple[PhysicalBatchGroup, ...]

    @property
    def logical_scene_count(self) -> int:
        return sum(len(group.scenes) for group in self.groups)

    @property
    def work(self) -> int:
        return sum(group.work for group in self.groups)


@dataclass(frozen=True)
class SynchronizedBatchWave:
    """The two rank-local physical groups for one synchronized update."""

    ranks: tuple[RankWave, ...]
    capacity_name: str

    @property
    def physical_group_count(self) -> int:
        return max(len(rank.groups) for rank in self.ranks)

    @property
    def pair_count(self) -> int:
        return sum(
            len(group.scenes) - 1
            for rank in self.ranks
            for group in rank.groups
        )

    @property
    def total_padding_tracks(self) -> int:
        return sum(
            group.padded_track_count
            for rank in self.ranks
            for group in rank.groups
        )

    @property
    def rank_work(self) -> tuple[int, ...]:
        return tuple(rank.work for rank in self.ranks)

    @property
    def wave_imbalance(self) -> int:
        return abs(self.ranks[0].work - self.ranks[1].work)

    @property
    def total_wave_imbalance(self) -> int:
        return self.wave_imbalance


def _validate_summaries(
    summaries: Sequence[SceneSummary], capacity: BatchCapacity
) -> None:
    expected = capacity.rank_count * capacity.logical_scenes_per_rank
    if len(summaries) != expected:
        raise ValueError(f"expected exactly {expected} scene summaries")
    if capacity.rank_count != 2:
        raise ValueError("synchronized scheduling currently requires two ranks")
    if capacity.max_group_size != 2:
        raise ValueError("this scheduler currently supports groups of at most two")
    identities = [(s.source, s.scene, s.cursor) for s in summaries]
    if len(set(identities)) != len(identities):
        raise ValueError("scene summaries must have unique source/scene/cursor identities")
    for scene in summaries:
        if not scene.source or not scene.scene:
            raise ValueError("source and scene must be non-empty")
        if scene.cursor < 0:
            raise ValueError("cursor must be non-negative")
        if not 1 <= scene.view_count <= 6:
            raise ValueError("view_count must be in [1, 6]")
        if scene.frame_count < 1:
            raise ValueError("frame_count must be positive")
        if (
            len(scene.resolution) != 2
            or any(int(value) != value or int(value) < 1 for value in scene.resolution)
        ):
            raise ValueError("resolution must contain two positive integers")
        if scene.track_count < 1:
            raise ValueError("track_count must be positive")


def _can_pair(
    first: SceneSummary, second: SceneSummary, capacity: BatchCapacity
) -> bool:
    if capacity.max_group_size < 2:
        return False
    if first.shape_key != second.shape_key:
        return False
    if first.view_count in capacity.singleton_only_views:
        return False
    # Trajectory counts are padded within the physical batch and masked by the
    # model; they are not a pairing compatibility constraint.
    return True


def _groupings(
    indices: tuple[int, ...],
    summaries: Sequence[SceneSummary],
    capacity: BatchCapacity,
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    """Enumerate every stable singleton/pair grouping for one rank."""

    if not indices:
        return ((),)
    first, *rest = indices
    result: list[tuple[tuple[int, ...], ...]] = []
    for suffix in _groupings(tuple(rest), summaries, capacity):
        result.append(((first,),) + suffix)
    for position, second in enumerate(rest):
        if not _can_pair(summaries[first], summaries[second], capacity):
            continue
        remaining = tuple(rest[:position] + rest[position + 1 :])
        for suffix in _groupings(remaining, summaries, capacity):
            result.append(((first, second),) + suffix)
    return tuple(result)


def _scene_index_signature(
    groups: tuple[PhysicalBatchGroup, ...], summaries: Sequence[SceneSummary]
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(summaries.index(scene) for scene in group.scenes)
        for group in groups
    )


def _order_groups(
    groups: tuple[PhysicalBatchGroup, ...],
    summaries: Sequence[SceneSummary],
) -> tuple[PhysicalBatchGroup, ...]:
    """Run larger groups first so their next decode can overlap model work."""

    return tuple(
        sorted(
            groups,
            key=lambda group: (
                -group.work,
                _scene_index_signature((group,), summaries),
            ),
        )
    )


def schedule_physical_batch(
    summaries: Sequence[SceneSummary],
    *,
    capacity: BatchCapacity = H100_BATCH_CAPACITY,
) -> SynchronizedBatchWave:
    """Return the deterministic two-rank physical batching plan.

    The objective is lexicographic: maximize safe pairs, minimize maximum rank
    work and rank-work imbalance, minimize trajectory padding, and finally
    choose the stable input-order tie-break.  Both ranks receive the same
    physical-group count so their final DDP backward collective is aligned.
    """

    summaries = tuple(summaries)
    _validate_summaries(summaries, capacity)
    all_indices = tuple(range(len(summaries)))
    candidates = []
    # Allow unequal logical-scene counts, but require synchronized physical
    # group counts so DDP's final backward collective is aligned.
    for rank_size in range(1, len(all_indices)):
        for rank_zero in combinations(all_indices[1:], rank_size - 1):
            rank_zero = (0,) + rank_zero
            rank_one = tuple(index for index in all_indices if index not in rank_zero)
            rank_zero_groupings = _groupings(rank_zero, summaries, capacity)
            rank_one_groupings = _groupings(rank_one, summaries, capacity)
            for raw_groups_zero in rank_zero_groupings:
                for raw_groups_one in rank_one_groupings:
                    groups_zero = _order_groups(tuple(
                        PhysicalBatchGroup(tuple(summaries[index] for index in group))
                        for group in raw_groups_zero
                    ), summaries)
                    groups_one = _order_groups(tuple(
                        PhysicalBatchGroup(tuple(summaries[index] for index in group))
                        for group in raw_groups_one
                    ), summaries)
                    rank_groups = tuple(
                        RankWave(
                            rank=rank,
                            groups=groups,
                        )
                        for rank, groups in enumerate((groups_zero, groups_one))
                    )
                    wave = SynchronizedBatchWave(
                        ranks=rank_groups,
                        capacity_name=capacity.name,
                    )
                    if len(groups_zero) == len(groups_one):
                        candidates.append(wave)

    if not candidates:
        raise ValueError("no synchronized physical batching plan exists")

    def objective(wave: SynchronizedBatchWave):
        rank_zero_signature = _scene_index_signature(
            wave.ranks[0].groups, summaries
        )
        rank_one_signature = _scene_index_signature(
            wave.ranks[1].groups, summaries
        )
        rank_work = wave.rank_work
        return (
            -wave.pair_count,
            max(rank_work),
            abs(rank_work[0] - rank_work[1]),
            wave.total_padding_tracks,
            rank_zero_signature,
            rank_one_signature,
        )

    return min(candidates, key=objective)


def schedule_rank_local_batch(
    summaries: Sequence[SceneSummary],
    *,
    capacity: BatchCapacity = H100_BATCH_CAPACITY,
) -> tuple[PhysicalBatchGroup, ...]:
    """Pair one rank's logical scenes without moving scenes between ranks.

    This is the rank-local counterpart to :func:`schedule_physical_batch`.
    It intentionally has no world-size assumption; DDP callers can use the
    resulting group count when coordinating their local accumulation wave.
    """
    summaries = tuple(summaries)
    if not summaries:
        raise ValueError("rank-local scheduling requires at least one scene")
    if len(summaries) > capacity.logical_scenes_per_rank:
        raise ValueError("rank-local batch exceeds logical scene capacity")
    for scene in summaries:
        if scene.track_count < 1:
            raise ValueError("track_count must be positive")
    candidates = _groupings(tuple(range(len(summaries))), summaries, capacity)
    if not candidates:
        raise ValueError("no rank-local physical batching plan exists")
    groups = []
    for raw in candidates:
        groups.append(tuple(
            PhysicalBatchGroup(tuple(summaries[index] for index in group))
            for group in raw
        ))
    return min(
        groups,
        key=lambda candidate: (
            len(candidate),
            sum(group.padded_track_count for group in candidate),
            sum(group.work for group in candidate),
            _scene_index_signature(candidate, summaries),
        ),
    )


__all__ = [
    "BatchCapacity",
    "H100_BATCH_CAPACITY",
    "PhysicalBatchGroup",
    "RankWave",
    "SceneSummary",
    "SynchronizedBatchWave",
    "schedule_rank_local_batch",
    "schedule_physical_batch",
]
