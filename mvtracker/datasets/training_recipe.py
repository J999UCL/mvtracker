"""Metadata-only planning and storage for deterministic mixed training recipes."""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

import numpy as np


LogFunction = Callable[[str], None]


def _print_log(message: str) -> None:
    print(message, flush=True)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raise TypeError(f"recipe value is not JSON serializable: {type(value).__name__}")


@dataclass(frozen=True)
class PhysicalAssignment:
    rank: int
    group: int
    position: int


@dataclass(frozen=True)
class RecipeRecord:
    step: int
    microbatch: int
    rank: int
    scheduled_rank: int
    source: str
    source_cursor: int
    retry_count: int
    request: dict[str, Any]
    seed: int
    scene_index: int
    scene: str
    frames: tuple[int, ...]
    views: tuple[int, ...]
    track_count: int
    tracks: tuple[int, ...]
    augmentation: dict[str, Any]
    depth_source: str
    physical: PhysicalAssignment

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecipeRecord":
        data = dict(value)
        data["frames"] = tuple(int(item) for item in data["frames"])
        data["views"] = tuple(int(item) for item in data["views"])
        data["tracks"] = tuple(int(item) for item in data["tracks"])
        data["physical"] = PhysicalAssignment(**data["physical"])
        return cls(**data)

    def replay_request(self, request_factory: Callable[..., Any]) -> Any:
        """Recreate the dataset request without coupling recipes to its class."""
        return request_factory(
            **{
                **self.request,
                "depth_source": self.depth_source,
                "expected_scene": self.scene,
            }
        )


class RecipeWriter:
    """Stream a recipe to disk and mark it complete only after finalization."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        manifest: Mapping[str, Any],
        world_size: int,
        step_count: int,
        records_per_step: int,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.world_size = int(world_size)
        self.expected_records = int(step_count) * int(records_per_step)
        self._counts = [0] * self.world_size
        self._files = [
            (self.output_dir / f"rank-{rank}.jsonl").open("w", encoding="utf-8")
            for rank in range(self.world_size)
        ]
        self._manifest = {
            **_jsonable(manifest),
            "world_size": self.world_size,
            "step_count": int(step_count),
            "records_per_step": int(records_per_step),
            "expected_records": self.expected_records,
            "complete": False,
        }
        self._write_json("manifest.json", self._manifest)

    def _write_json(self, name: str, value: Any) -> None:
        with (self.output_dir / name).open("w", encoding="utf-8") as handle:
            json.dump(_jsonable(value), handle, indent=2, sort_keys=True)
            handle.write("\n")

    def write(self, record: RecipeRecord) -> None:
        handle = self._files[record.rank]
        handle.write(json.dumps(record.to_dict(), separators=(",", ":")) + "\n")
        self._counts[record.rank] += 1

    def finalize(
        self,
        *,
        summary: Mapping[str, Any],
        estimated_depth_requests: Sequence[Mapping[str, Any]],
    ) -> None:
        self.close()
        actual_records = sum(self._counts)
        if actual_records != self.expected_records:
            raise ValueError(
                f"recipe has {actual_records} records; expected {self.expected_records}"
            )
        with (self.output_dir / "estimated-depth-requests.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for request in estimated_depth_requests:
                handle.write(json.dumps(_jsonable(request), separators=(",", ":")) + "\n")
        self._write_json("summary.json", summary)
        self._manifest.update(
            actual_records=actual_records,
            rank_record_counts=self._counts,
            complete=True,
        )
        self._write_json("manifest.json", self._manifest)

    def close(self) -> None:
        for handle in self._files:
            if not handle.closed:
                handle.flush()
                handle.close()


class RecipeReader:
    """Read completed rank-local recipe streams and perform cheap safety checks."""

    def __init__(self, recipe_dir: str | Path) -> None:
        self.recipe_dir = Path(recipe_dir)
        with (self.recipe_dir / "manifest.json").open(encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        if not self.manifest.get("complete"):
            raise ValueError("training recipe is incomplete")

    def records(self, rank: int) -> Iterator[RecipeRecord]:
        path = self.recipe_dir / f"rank-{rank}.jsonl"
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                yield RecipeRecord.from_dict(json.loads(line))

    def validate(self) -> None:
        positions: set[tuple[int, int, int]] = set()
        counts = []
        for rank in range(int(self.manifest["world_size"])):
            count = 0
            for record in self.records(rank):
                if record.rank != rank:
                    raise ValueError(f"rank-{rank}.jsonl contains rank {record.rank}")
                position = (record.step, record.microbatch, record.scheduled_rank)
                if position in positions:
                    raise ValueError(f"duplicate recipe position: {position}")
                positions.add(position)
                count += 1
            counts.append(count)
        if counts != self.manifest["rank_record_counts"]:
            raise ValueError(f"rank record counts differ: {counts}")
        if sum(counts) != int(self.manifest["expected_records"]):
            raise ValueError("recipe record count differs from manifest")


class _Heartbeat:
    def __init__(self, interval: float, log: LogFunction, status: Callable[[], str]):
        self.interval = float(interval)
        self.log = log
        self.status = status
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self.stop.wait(self.interval):
            self.log(self.status())

    def __enter__(self) -> "_Heartbeat":
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop.set()
        self.thread.join()


def _request_dict(request: Any) -> dict[str, Any]:
    if is_dataclass(request):
        names = (field.name for field in fields(request))
    elif hasattr(request, "__dict__"):
        names = (name for name in vars(request) if not name.startswith("_"))
    else:
        names = ("virtual_index", "scene_index", "view_count", "depth_source")
    return {name: _jsonable(getattr(request, name)) for name in names if hasattr(request, name)}


def _scene_summary(source: str, plan: Any) -> Any:
    from .physical_batch_scheduler import SceneSummary

    query_times = np.asarray(plan.query_points_3d)[:, 0]
    return SceneSummary(
        source=source,
        scene=plan.sequence,
        cursor=int(plan.virtual_index),
        view_count=len(plan.views),
        frame_count=len(plan.frame_indices),
        resolution=tuple(int(item) for item in plan.output_size),
        track_count=int(plan.track_count),
        schedule_start=int(query_times.min()),
    )


def plan_training_recipe(
    output_dir: str | Path,
    *,
    datasets: Mapping[str, Any],
    schedule: Any,
    step_count: int,
    manifest: Mapping[str, Any],
    source_cursors: Mapping[str, int] | None = None,
    heartbeat_seconds: float = 10.0,
    log: LogFunction = _print_log,
    physical_scheduler: Callable[[Sequence[Any]], Any] | None = None,
) -> dict[str, Any]:
    """Plan every rank without calling any dataset materialization method."""
    if step_count < 1:
        raise ValueError("step_count must be positive")
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    started = time.perf_counter()
    cursors = {
        source: int((source_cursors or {}).get(source, 0))
        for source in schedule.scene_counts
    }
    progress: dict[str, Any] = {
        "step": 0,
        "records": 0,
        "retries": 0,
        "source": "startup",
        "scene": "-",
    }
    source_counts: Counter[str] = Counter()
    view_counts: Counter[str] = Counter()
    track_counts: Counter[str] = Counter()
    depth_counts: Counter[str] = Counter()
    augmentation_counts: Counter[str] = Counter()
    source_planning_seconds: Counter[str] = Counter()
    source_plan_calls: Counter[str] = Counter()
    estimated: dict[tuple[str, str], set[str]] = {}
    records_per_step = len(schedule.source_pattern) * int(schedule.world_size)

    def status() -> str:
        elapsed = max(time.perf_counter() - started, 1e-9)
        rate = progress["records"] / elapsed
        remaining = records_per_step * step_count - progress["records"]
        eta = remaining / rate if rate else 0.0
        return (
            "recipe heartbeat "
            f"step={progress['step']}/{step_count} records={progress['records']} "
            f"source={progress['source']} scene={progress['scene']} "
            f"retries={progress['retries']} rate={rate:.1f}/s eta={eta:.1f}s "
            f"depth={dict(depth_counts)}"
        )

    log(
        "recipe planner start "
        f"steps={step_count} world_size={schedule.world_size} "
        f"records={step_count * records_per_step} output={output_dir}"
    )
    writer = RecipeWriter(
        output_dir,
        manifest={
            **manifest,
            "source_pattern": list(schedule.source_pattern),
            "initial_source_cursors": cursors,
        },
        world_size=schedule.world_size,
        step_count=step_count,
        records_per_step=records_per_step,
    )
    try:
        with _Heartbeat(heartbeat_seconds, log, status):
            for step in range(step_count):
                accepted: list[tuple[int, int, str, int, int, Any, Any]] = []
                for microbatch, source in enumerate(schedule.source_pattern):
                    cursor = cursors[source]
                    rejected = 0
                    while True:
                        candidates = []
                        for scheduled_rank in range(schedule.world_size):
                            request = schedule.sample_source(
                                source, cursor, scheduled_rank
                            ).request
                            resolver = getattr(
                                datasets[source], "resolve_recipe_request", None
                            )
                            if resolver is not None:
                                request = resolver(request)
                            progress.update(
                                source=source,
                                scene=f"scene-index-{request.scene_index}",
                            )
                            planning_started = time.perf_counter()
                            plan = datasets[source].plan_sample(request)
                            source_planning_seconds[source] += (
                                time.perf_counter() - planning_started
                            )
                            source_plan_calls[source] += 1
                            if plan is None:
                                break
                            candidates.append((scheduled_rank, request, plan))
                        if len(candidates) == schedule.world_size:
                            for scheduled_rank, request, plan in candidates:
                                accepted.append(
                                    (
                                        microbatch,
                                        scheduled_rank,
                                        source,
                                        cursor,
                                        rejected,
                                        request,
                                        plan,
                                    )
                                )
                            cursors[source] = cursor + 1
                            break
                        cursor += 1
                        rejected += 1
                        progress["retries"] += 1

                if physical_scheduler is None:
                    assignments = [
                        (
                            PhysicalAssignment(
                                rank=int(item[1]),
                                group=int(item[0]),
                                position=0,
                            ),
                            item,
                        )
                        for item in accepted
                    ]
                else:
                    summaries = tuple(
                        _scene_summary(source, plan)
                        for _, _, source, _, _, _, plan in accepted
                    )
                    physical = physical_scheduler(summaries)
                    by_identity = {
                        (source, plan.sequence, int(plan.virtual_index)): item
                        for item in accepted
                        for source, plan in ((item[2], item[6]),)
                    }
                    assignments = []
                    for rank_wave in physical.ranks:
                        for group_index, group in enumerate(rank_wave.groups):
                            for position, summary in enumerate(group.scenes):
                                assignments.append(
                                    (
                                        PhysicalAssignment(
                                            rank=int(rank_wave.rank),
                                            group=group_index,
                                            position=position,
                                        ),
                                        by_identity[
                                            (
                                                summary.source,
                                                summary.scene,
                                                summary.cursor,
                                            )
                                        ],
                                    )
                                )
                for physical_assignment, item in assignments:
                            microbatch, scheduled_rank, source, cursor, retries, request, plan = item
                            depth_source = str(plan.depth_source)
                            augmentation = {
                                "apply_rgb": bool(plan.apply_rgb_aug),
                                "rgb": _jsonable(plan.rgb_augmentation),
                                "apply_depth": bool(plan.apply_depth_aug),
                                "depth_patch_operations": _jsonable(
                                    plan.depth_patch_operations
                                ),
                                "seed": int(plan.augmentation_seed),
                            }
                            record = RecipeRecord(
                                step=step,
                                microbatch=microbatch,
                                rank=physical_assignment.rank,
                                scheduled_rank=scheduled_rank,
                                source=source,
                                source_cursor=cursor,
                                retry_count=retries,
                                request=_request_dict(request),
                                seed=int(plan.seed),
                                scene_index=int(plan.scene_index),
                                scene=str(plan.sequence),
                                frames=tuple(int(item) for item in plan.frame_indices),
                                views=tuple(int(item) for item in plan.views),
                                track_count=int(plan.track_count),
                                tracks=tuple(
                                    int(item)
                                    for item in plan.selected_global_track_indices
                                ),
                                augmentation=augmentation,
                                depth_source=depth_source,
                                physical=physical_assignment,
                            )
                            writer.write(record)
                            source_counts[source] += 1
                            view_counts[str(len(plan.views))] += 1
                            track_counts[str(plan.track_count)] += 1
                            depth_counts[depth_source] += 1
                            augmentation_counts[str(bool(plan.apply_rgb_aug))] += 1
                            if depth_source != "gt":
                                estimated.setdefault(
                                    (source, str(plan.sequence)), set()
                                ).add(depth_source)
                            progress.update(
                                records=progress["records"] + 1,
                                source=source,
                                scene=str(plan.sequence),
                            )
                progress["step"] = step + 1
                if progress["step"] % 25 == 0 or progress["step"] == step_count:
                    log(status())

        elapsed = time.perf_counter() - started
        summary = {
            "elapsed_seconds": elapsed,
            "records": progress["records"],
            "retries": progress["retries"],
            "final_source_cursors": cursors,
            "source_counts": dict(source_counts),
            "view_counts": dict(view_counts),
            "track_counts": dict(track_counts),
            "planned_depth_counts": dict(depth_counts),
            "rgb_augmentation_counts": dict(augmentation_counts),
            "source_plan_calls": dict(source_plan_calls),
            "source_planning_seconds": dict(source_planning_seconds),
            "unique_estimated_depth_scenes": len(estimated),
        }
        writer.finalize(
            summary=summary,
            estimated_depth_requests=[
                {
                    "source": source,
                    "scene": scene,
                    "planned_depth_sources": sorted(estimated[(source, scene)]),
                }
                for source, scene in sorted(estimated)
            ],
        )
        log(
            "recipe planner complete "
            f"steps={step_count} records={progress['records']} retries={progress['retries']} "
            f"elapsed={elapsed:.2f}s rate={progress['records'] / max(elapsed, 1e-9):.1f}/s"
        )
        return summary
    except BaseException:
        writer.close()
        log(f"recipe planner failed {status()}")
        raise


__all__ = [
    "PhysicalAssignment",
    "RecipeReader",
    "RecipeRecord",
    "RecipeWriter",
    "plan_training_recipe",
]
