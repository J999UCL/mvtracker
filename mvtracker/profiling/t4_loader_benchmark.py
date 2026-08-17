"""Deterministic case matrix and reporting helpers for the T4 loader profile."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import time
from typing import Callable, Mapping, Sequence


T4_GPU_REQUEST = "T4"
T4_MAX_CONTAINERS = 1
T4_WORKERS = 8
SIMULATED_COMPUTE_SECONDS = 1.25
SOURCE_SCHEDULE = ("diegesis", "mvkubric", "diegesis", "mvkubric")


@dataclass(frozen=True)
class LoaderCase:
    source: str
    views: int

    @property
    def name(self) -> str:
        return f"{self.source}-views{self.views}"


CASES = (
    LoaderCase("diegesis", 4),
    LoaderCase("mvkubric", 4),
    LoaderCase("mvkubric", 6),
)


def percentile(values: Sequence[float], fraction: float) -> float:
    """Return the nearest-rank percentile used by the existing profile."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between zero and one")
    ordered = sorted(float(value) for value in values)
    return ordered[min(len(ordered) - 1, max(0, int(len(ordered) * fraction) - 1))]


def validate_profile(profile: Mapping[str, object], *, case: LoaderCase) -> None:
    """Check the fields required for a comparable loader result."""
    if profile.get("view_count") != case.views:
        raise ValueError(f"{case.name}: profile view count does not match case")
    for key in (
        "samples_per_second",
        "sample_seconds_median",
        "sample_seconds_p95",
        "exposed_wait_seconds_p50",
        "exposed_wait_seconds_p95",
        "max_exposed_wait_seconds",
    ):
        if key not in profile:
            raise ValueError(f"{case.name}: profile is missing {key}")


def run_case_matrix(
    profile_loader: Callable[..., Mapping[str, object]],
    *,
    warmup: int,
    measured: int,
    workers: int = T4_WORKERS,
    simulated_compute_seconds: float = SIMULATED_COMPUTE_SECONDS,
    hardware_sampler: Callable[[], Mapping[str, object]] | None = None,
    progress_callback: Callable[[str, str, Mapping[str, object] | None], None] | None = None,
) -> dict[str, object]:
    """Run the short warm-only matrix and report each case as it completes."""
    profiles: dict[str, Mapping[str, object]] = {}
    for case in CASES:
        if progress_callback is not None:
            progress_callback("started", case.name, None)
        result = _run_profile(
            profile_loader, case, warmup=warmup, measured=measured,
            workers=workers, simulated_compute_seconds=0.0,
            hardware_sampler=hardware_sampler,
        )
        profiles[case.name] = result
        if progress_callback is not None:
            progress_callback("completed", case.name, result)

    schedule_name = "alternating-dkdk-views4"
    if progress_callback is not None:
        progress_callback("started", schedule_name, None)
    schedule_profile = profile_loader(
        source="diegesis", source_schedule=SOURCE_SCHEDULE, view_count=4,
        warmup=warmup, measured=measured, workers=workers, use_cuda=True,
        simulated_compute_seconds=simulated_compute_seconds,
        hardware_sampler=hardware_sampler,
    )
    if tuple(schedule_profile.get("source_schedule", ())) != SOURCE_SCHEDULE:
        raise ValueError("alternating profile did not preserve the production source schedule")
    if progress_callback is not None:
        progress_callback("completed", schedule_name, schedule_profile)
    return {
        "cases": profiles,
        "alternating_source_schedule": schedule_profile,
        "alternating_schedule_view_count": 4,
        "alternating_schedule_label": "representative-fixed-view4",
        "case_matrix": [case.name for case in CASES],
        "source_schedule": list(SOURCE_SCHEDULE),
        "warmup": warmup,
        "measured": measured,
        "workers": workers,
        "fixed_case_simulated_compute_seconds": 0.0,
        "alternating_simulated_compute_seconds": simulated_compute_seconds,
    }


def _run_profile(
    profile_loader: Callable[..., Mapping[str, object]],
    case: LoaderCase,
    *,
    warmup: int,
    measured: int,
    workers: int,
    simulated_compute_seconds: float,
    hardware_sampler: Callable[[], Mapping[str, object]] | None,
) -> Mapping[str, object]:
    result = profile_loader(
        source=case.source,
        view_count=case.views,
        warmup=warmup,
        measured=measured,
        workers=workers,
        use_cuda=True,
        simulated_compute_seconds=simulated_compute_seconds,
        hardware_sampler=hardware_sampler,
    )
    validate_profile(result, case=case)
    return result


class ContainerHardwareMonitor:
    """Read CPU and RAM usage from the Modal cgroup v1 limits."""

    def __init__(self, cgroup_root: Path = Path("/sys/fs/cgroup")):
        self.root = Path(cgroup_root)
        self._last_cpu_seconds = self._cpu_seconds()
        self._last_sample_time = time.monotonic()

    def _cpu_seconds(self) -> float:
        return int((self.root / "cpuacct/cpuacct.usage").read_text()) / 1_000_000_000.0

    def sample(self) -> dict[str, float]:
        now = time.monotonic()
        cpu_seconds = self._cpu_seconds()
        elapsed = now - self._last_sample_time
        cpu_cores = (cpu_seconds - self._last_cpu_seconds) / elapsed
        self._last_cpu_seconds = cpu_seconds
        self._last_sample_time = now
        quota = int((self.root / "cpu/cpu.cfs_quota_us").read_text())
        period = int((self.root / "cpu/cpu.cfs_period_us").read_text())
        cpu_limit = float(len(os.sched_getaffinity(0))) if quota < 0 else quota / period
        memory_used = int((self.root / "memory/memory.usage_in_bytes").read_text())
        memory_limit = int((self.root / "memory/memory.limit_in_bytes").read_text())
        return {
            "cpu_cores_used": cpu_cores,
            "cpu_utilization_percent": 100.0 * cpu_cores / cpu_limit,
            "ram_used_gib": memory_used / (1024 ** 3),
            "ram_limit_gib": memory_limit / (1024 ** 3),
            "ram_utilization_percent": 100.0 * memory_used / memory_limit,
        }


class GpuHardwareMonitor:
    """Read utilization and memory for the one assigned T4 through NVML."""

    def __init__(self, device_index: int = 0):
        import pynvml

        self._pynvml = pynvml
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(int(device_index))

    def sample(self) -> dict[str, float]:
        utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        return {
            "gpu_utilization_percent": float(utilization.gpu),
            "gpu_memory_used_gib": memory.used / (1024 ** 3),
            "gpu_memory_total_gib": memory.total / (1024 ** 3),
            "gpu_memory_utilization_percent": 100.0 * memory.used / memory.total,
        }

    def close(self) -> None:
        self._pynvml.nvmlShutdown()
