"""Pure configuration and search logic for Modal training profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


TRAJECTORY_CANDIDATES = (256, 512, 768, 1024, 1280, 1536, 1792, 2048)


@dataclass(frozen=True)
class ProfileCase:
    views: int
    batch_size: int
    accumulation: int

    @property
    def name(self) -> str:
        return f"views{self.views}-batch{self.batch_size}-accum{self.accumulation}"


PROFILE_CASES = (
    ProfileCase(views=1, batch_size=4, accumulation=1),
    ProfileCase(views=2, batch_size=4, accumulation=1),
    ProfileCase(views=3, batch_size=2, accumulation=2),
    ProfileCase(views=4, batch_size=2, accumulation=2),
)


@dataclass(frozen=True)
class TrialResult:
    trajectories: int
    status: str
    peak_memory_bytes: int | None = None
    total_memory_bytes: int | None = None
    result_path: str | None = None

    @property
    def safe(self) -> bool:
        return self.status == "safe"


@dataclass(frozen=True)
class SearchResult:
    selected_trajectories: int | None
    trials: tuple[TrialResult, ...]


def is_memory_safe(
    peak_bytes: int,
    total_bytes: int,
    *,
    max_fraction: float = 0.90,
) -> bool:
    if peak_bytes < 0 or total_bytes <= 0:
        raise ValueError("memory byte counts must be positive")
    if not 0.0 < max_fraction < 1.0:
        raise ValueError("max_fraction must be between zero and one")
    return peak_bytes <= total_bytes * max_fraction


def find_largest_safe(
    probe: Callable[[int], TrialResult],
    candidates: tuple[int, ...] = TRAJECTORY_CANDIDATES,
) -> SearchResult:
    """Probe the ceiling, then binary-search the largest safe candidate."""
    if not candidates or tuple(sorted(set(candidates))) != candidates:
        raise ValueError("candidates must be non-empty, unique, and increasing")

    trials: list[TrialResult] = []
    ceiling = probe(candidates[-1])
    trials.append(ceiling)
    if ceiling.trajectories != candidates[-1]:
        raise ValueError("probe returned a result for the wrong trajectory count")
    if ceiling.safe:
        return SearchResult(ceiling.trajectories, tuple(trials))

    low = 0
    high = len(candidates) - 2
    selected: int | None = None
    while low <= high:
        middle = (low + high) // 2
        requested = candidates[middle]
        result = probe(requested)
        trials.append(result)
        if result.trajectories != requested:
            raise ValueError("probe returned a result for the wrong trajectory count")
        if result.safe:
            selected = requested
            low = middle + 1
        else:
            high = middle - 1
    return SearchResult(selected, tuple(trials))


def validate_gpu_request(spec: str, *, max_gpus: int = 2) -> int:
    """Return the requested GPU count after enforcing the project ceiling."""
    if not spec or isinstance(spec, (list, tuple)):
        raise ValueError("GPU request must be one explicit Modal GPU string")
    name, separator, count_text = spec.rpartition(":")
    count = int(count_text) if separator else 1
    gpu_name = name if separator else spec
    if not gpu_name or count < 1:
        raise ValueError(f"invalid GPU request: {spec!r}")
    if count > max_gpus:
        raise ValueError(
            f"GPU request {spec!r} exceeds the hard limit of {max_gpus}"
        )
    return count
