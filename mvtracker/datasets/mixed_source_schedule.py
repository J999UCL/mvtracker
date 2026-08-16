from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

@dataclass(frozen=True)
class ScheduledSampleRequest:
    virtual_index: int
    scene_index: int
    view_count: None = None


@dataclass(frozen=True)
class MixedSourceSample:
    source: str
    request: ScheduledSampleRequest


class BalancedMixedSourceSchedule:
    """Stateless rank-local schedule over balanced shuffled scene cycles."""

    def __init__(
        self,
        scene_counts: Mapping[str, int],
        source_pattern: Sequence[str],
        *,
        world_size: int,
        master_seed: int,
    ):
        if world_size < 1:
            raise ValueError("world_size must be at least one")
        if not source_pattern:
            raise ValueError("source_pattern must not be empty")
        if any(source not in scene_counts for source in source_pattern):
            raise ValueError("source_pattern contains an unknown source")
        if any(int(count) < 1 for count in scene_counts.values()):
            raise ValueError("every source must contain at least one scene")

        self.scene_counts = {source: int(count) for source, count in scene_counts.items()}
        self.source_pattern = tuple(source_pattern)
        self.world_size = int(world_size)
        self.master_seed = int(master_seed)
        self._source_indices = {
            source: index for index, source in enumerate(self.scene_counts)
        }
        self._positions = {
            source: tuple(
                index for index, candidate in enumerate(self.source_pattern)
                if candidate == source
            )
            for source in self.scene_counts
        }

    def sample(
        self,
        completed_step: int,
        microbatch: int,
        rank: int,
        attempt: int = 0,
    ) -> MixedSourceSample:
        if completed_step < 0:
            raise ValueError("completed_step must be non-negative")
        if not 0 <= microbatch < len(self.source_pattern):
            raise ValueError("microbatch is outside the source pattern")
        if not 0 <= rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        if attempt < 0:
            raise ValueError("attempt must be non-negative")

        source = self.source_pattern[microbatch]
        positions = self._positions[source]
        position_within_source = positions.index(microbatch)
        local_cursor = (
            completed_step * len(positions)
            + position_within_source
            + attempt
        )
        return self.sample_source(source, local_cursor, rank)

    def sample_source(
        self,
        source: str,
        local_cursor: int,
        rank: int,
    ) -> MixedSourceSample:
        """Resolve one rank-local source cursor to an explicit scene request."""
        if source not in self.scene_counts:
            raise ValueError(f"unknown source: {source}")
        if local_cursor < 0:
            raise ValueError("local_cursor must be non-negative")
        if not 0 <= rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")

        source_ordinal = local_cursor * self.world_size + rank
        scene_count = self.scene_counts[source]
        cycle, offset = divmod(source_ordinal, scene_count)
        permutation = np.random.default_rng(
            np.random.SeedSequence(
                [self.master_seed, self._source_indices[source], cycle]
            )
        ).permutation(scene_count)
        return MixedSourceSample(
            source=source,
            request=ScheduledSampleRequest(
                virtual_index=source_ordinal,
                scene_index=int(permutation[offset]),
            ),
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "scene_counts": dict(self.scene_counts),
            "source_pattern": list(self.source_pattern),
            "world_size": self.world_size,
            "master_seed": self.master_seed,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if state != self.state_dict():
            raise ValueError("mixed-source schedule state does not match this run")
