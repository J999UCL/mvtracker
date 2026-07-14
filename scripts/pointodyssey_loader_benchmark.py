#!/usr/bin/env python3
"""Benchmark the prepared PointOdyssey MV-Tracker training loader."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import random
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence


EXPECTED_SCENES = 78
DEFAULT_WORKERS = (0, 2, 4, 8)
DEFAULT_SEED = 20260714
PREFETCH_FACTOR = 4
MIB = 1024 * 1024


class FixedIndexSampler:
    """Yield one fixed virtual-index schedule."""

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = tuple(int(index) for index in indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Parent containing PointOdyssey_MVTracker, e.g. /tmp/thakwani.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--worker-counts",
        nargs="+",
        type=int,
        default=list(DEFAULT_WORKERS),
    )
    parser.add_argument("--warmup-samples", type=int, default=32)
    parser.add_argument("--samples-per-worker", type=int, default=156)
    parser.add_argument("--skip-coverage", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--progress-every", type=int, default=8)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--wandb-project", default="mvtracker-loader-benchmark")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args(argv)
    if (
        not args.worker_counts
        or any(workers < 0 for workers in args.worker_counts)
        or len(set(args.worker_counts)) != len(args.worker_counts)
    ):
        parser.error("worker counts must be unique non-negative integers")
    if args.warmup_samples < 0 or args.samples_per_worker <= 0 or args.progress_every <= 0:
        parser.error("warmup must be non-negative; samples and progress must be positive")
    return args


def build_balanced_schedule(
    count: int,
    *,
    repeat_offset: int,
    seed: int,
    scene_count: int = EXPECTED_SCENES,
) -> list[int]:
    """Return distinct virtual indices whose scene IDs are near-balanced."""
    rng = random.Random(seed)
    result: list[int] = []
    repeat = repeat_offset
    while len(result) < count:
        scenes = list(range(scene_count))
        rng.shuffle(scenes)
        result.extend(scene + repeat * scene_count for scene in scenes)
        repeat += 1
    return result[:count]


def summarize_diversity(metadata: Sequence[dict[str, Any]]) -> dict[str, Any]:
    starts_by_scene: dict[str, list[int]] = defaultdict(list)
    for item in metadata:
        starts_by_scene[str(item["scene_name"])].append(int(item["window_start"]))
    pairs = [
        (scene, start)
        for scene, starts in starts_by_scene.items()
        for start in starts
    ]
    return {
        "observations": len(pairs),
        "unique_scenes": len(starts_by_scene),
        "unique_scene_start_pairs": len(set(pairs)),
        "starts_by_scene": {
            scene: sorted(starts)
            for scene, starts in sorted(starts_by_scene.items())
        },
    }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def load_scene_contracts(dataset_root: Path) -> dict[str, dict[str, Any]]:
    prepared_root = dataset_root / "PointOdyssey_MVTracker"
    report_path = prepared_root / "validation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 5:
        raise ValueError("PointOdyssey validation_report.json must use schema version 5")
    if report.get("status") != "completed" or report.get("failures") != []:
        raise ValueError("PointOdyssey validation_report.json is not clean and completed")

    train_root = prepared_root / "train"
    scene_roots = sorted(
        (path for path in train_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    expected_names = [f"{index:06d}" for index in range(EXPECTED_SCENES)]
    if [path.name for path in scene_roots] != expected_names:
        raise ValueError("prepared training split must contain scenes 000000 through 000077")

    contracts = {}
    for scene_root in scene_roots:
        metadata = json.loads((scene_root / "scene.json").read_text(encoding="utf-8"))
        window_exclusion = metadata["output"]["window_exclusion"]
        contracts[scene_root.name] = {
            "frame_count": int(metadata["output"]["frame_count"]),
            "invalid_frame_indices": [
                int(frame)
                for frame in window_exclusion["invalid_frame_indices"]
            ],
            "legal_start_count": int(window_exclusion["legal_start_count"]),
            "excluded_start_count": int(window_exclusion["excluded_start_count"]),
        }
    return contracts


class ResourceTracker:
    """Take lightweight process-tree RSS and I/O snapshots."""

    def __init__(self) -> None:
        import psutil

        self.psutil = psutil
        self.root = psutil.Process(os.getpid())
        self.first: dict[int, dict[str, int]] = {}
        self.last: dict[int, dict[str, int]] = {}
        self.peak_rss_bytes = 0
        self.sample()
        self.first = {pid: dict(value) for pid, value in self.last.items()}
        self.baseline_rss_bytes = self.peak_rss_bytes

    def sample(self) -> None:
        try:
            processes = [self.root, *self.root.children(recursive=True)]
        except self.psutil.Error:
            return
        rss = 0
        for process in processes:
            try:
                with process.oneshot():
                    io = process.io_counters()
                    values = {
                        "read_bytes": int(io.read_bytes),
                        "read_chars": int(io.read_chars),
                    }
                    rss += int(process.memory_info().rss)
            except self.psutil.Error:
                continue
            self.first.setdefault(process.pid, {"read_bytes": 0, "read_chars": 0})
            self.last[process.pid] = values
        self.peak_rss_bytes = max(self.peak_rss_bytes, rss)

    def stop(self) -> dict[str, Any]:
        self.sample()
        read_bytes = sum(
            max(0, value["read_bytes"] - self.first[pid]["read_bytes"])
            for pid, value in self.last.items()
        )
        read_chars = sum(
            max(0, value["read_chars"] - self.first[pid]["read_chars"])
            for pid, value in self.last.items()
        )
        return {
            "baseline_process_tree_rss_bytes": self.baseline_rss_bytes,
            "sampled_peak_process_tree_rss_bytes": self.peak_rss_bytes,
            "sampled_peak_rss_increase_bytes": max(
                0, self.peak_rss_bytes - self.baseline_rss_bytes
            ),
            "physical_read_bytes": read_bytes,
            "read_characters": read_chars,
            "semantics": (
                "RSS is sampled at progress updates and may miss brief peaks; shared pages may "
                "be double-counted and allocator memory can persist between lanes; physical reads "
                "are page-cache dependent; read_characters includes cached and non-dataset reads"
            ),
        }


def _load_runtime(dataset_root: Path):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    import cv2
    import numpy as np
    import torch
    import torchvision
    from omegaconf import OmegaConf
    from torchdata.stateful_dataloader import StatefulDataLoader

    from mvtracker.datasets.pointodyssey_multiview_dataset import (
        PointOdysseyMultiViewDataset,
    )
    from mvtracker.datasets.utils import collate_fn

    config = OmegaConf.merge(
        OmegaConf.load(repo_root / "configs" / "train.yaml"),
        OmegaConf.load(repo_root / "configs" / "experiment" / "pointodyssey.yaml"),
    )
    config.datasets.root = str(dataset_root)
    dataset = PointOdysseyMultiViewDataset.from_name(
        config.datasets.train.name,
        str(dataset_root),
        training_args=config,
        fabric=SimpleNamespace(world_size=1),
    )
    if dataset.real_len != EXPECTED_SCENES:
        raise ValueError(f"loader found {dataset.real_len} scenes, expected {EXPECTED_SCENES}")
    if bool(config.reproducibility.deterministic):
        raise ValueError("benchmark expects current training behavior in_order=False")
    return SimpleNamespace(
        config=config,
        dataset=dataset,
        torch=torch,
        StatefulDataLoader=StatefulDataLoader,
        collate_fn=collate_fn,
        versions={
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
        },
    )


def _seed(seed: int, torch: Any) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _metadata(sample: Any, *, batched: bool) -> dict[str, Any]:
    value = sample.sample_metadata
    if batched:
        if not isinstance(value, list) or len(value) != 1:
            raise ValueError("collated sample metadata must contain one item")
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError("sample metadata is missing")
    return value


def validate_provenance(
    metadata: dict[str, Any],
    *,
    gotit: bool,
    allowed_indices: set[int],
    contracts: dict[str, dict[str, Any]],
) -> list[str]:
    failures = []
    required = ("virtual_index", "scene_index", "scene_name", "gotit")
    if any(key not in metadata for key in required):
        return ["sample metadata is incomplete"]
    virtual_index = int(metadata["virtual_index"])
    scene_index = int(metadata["scene_index"])
    scene_name = str(metadata["scene_name"])
    if virtual_index not in allowed_indices:
        failures.append("virtual index is outside the benchmark schedule")
    if scene_index != virtual_index % EXPECTED_SCENES:
        failures.append("scene index does not match virtual index")
    if scene_name != f"{scene_index:06d}" or scene_name not in contracts:
        failures.append("scene name does not match scene index")
    if bool(metadata["gotit"]) != gotit:
        failures.append("metadata gotit disagrees with loader gotit")
    if not gotit or scene_name not in contracts:
        return failures

    start = int(metadata.get("window_start", -1))
    end = int(metadata.get("window_end_exclusive", -1))
    contract = contracts[scene_name]
    if end - start != 24 or start < 0 or end > int(contract["frame_count"]):
        failures.append("sample window is not a legal 24-frame interval")
    invalid = set(int(frame) for frame in contract["invalid_frame_indices"])
    if any(start <= frame < end for frame in invalid):
        failures.append("sample window intersects an invalid frame")
    return failures


def validate_sample(
    sample: Any,
    torch: Any,
    *,
    batched: bool,
    full: bool,
) -> dict[str, Any]:
    prefix = (1,) if batched else ()
    queries = sample.query_points_3d
    track_count = int(queries.shape[1 if batched else 0])
    tensors = {
        "video": sample.video,
        "videodepth": sample.videodepth,
        "segmentation": sample.segmentation,
        "trajectory": sample.trajectory,
        "trajectory_3d": sample.trajectory_3d,
        "visibility": sample.visibility,
        "valid": sample.valid,
        "intrs": sample.intrs,
        "extrs": sample.extrs,
        "query_points_3d": queries,
    }
    expected_shapes = {
        "video": prefix + (4, 24, 3, 384, 512),
        "videodepth": prefix + (4, 24, 1, 384, 512),
        "segmentation": prefix + (4, 24, 1, 384, 512),
        "trajectory": prefix + (4, 24, track_count, 3),
        "trajectory_3d": prefix + (24, track_count, 3),
        "visibility": prefix + (4, 24, track_count),
        "valid": prefix + (24, track_count),
        "intrs": prefix + (4, 24, 3, 3),
        "extrs": prefix + (4, 24, 3, 4),
        "query_points_3d": prefix + (track_count, 4),
    }
    expected_dtypes = {
        name: torch.float32
        for name in tensors
        if name != "visibility"
    }
    expected_dtypes["visibility"] = torch.bool
    failures = [
        f"{name} shape {tuple(tensors[name].shape)} != {shape}"
        for name, shape in expected_shapes.items()
        if tuple(tensors[name].shape) != shape
    ]
    failures.extend(
        f"{name} dtype {tensor.dtype} != {expected_dtypes[name]}"
        for name, tensor in tensors.items()
        if tensor.dtype != expected_dtypes[name]
    )

    statistics = {
        "track_count": track_count,
        "shapes": {
            name: list(tensor.shape)
            for name, tensor in tensors.items()
        },
        "visibility_true": int(sample.visibility.sum()),
        "visibility_total": int(sample.visibility.numel()),
    }
    if full and not failures:
        floating = [
            tensor
            for name, tensor in tensors.items()
            if name != "visibility"
        ]
        if not all(bool(torch.isfinite(tensor).all()) for tensor in floating):
            failures.append("one or more returned tensors contain nonfinite values")
        query_times = queries[..., 0]
        if (
            not bool((query_times == query_times.round()).all())
            or float(query_times.min()) < 0
            or float(query_times.max()) > 23
        ):
            failures.append("query times are not integers in [0, 23]")
        else:
            if batched:
                times = query_times[0].long()
                expected_xyz = sample.trajectory_3d[
                    0, times, torch.arange(track_count), :
                ]
                query_xyz = queries[0, :, 1:]
            else:
                times = query_times.long()
                expected_xyz = sample.trajectory_3d[
                    times, torch.arange(track_count), :
                ]
                query_xyz = queries[:, 1:]
            if not bool(
                torch.allclose(query_xyz, expected_xyz, atol=1e-4, rtol=1e-5)
            ):
                failures.append("query XYZ does not match trajectory_3d at query time")
        statistics["finite"] = not any("nonfinite" in failure for failure in failures)
        statistics["valid_depth_ratio"] = float(
            (sample.videodepth > 0).float().mean()
        )
    return {"failures": failures, "statistics": statistics}


def run_coverage(
    runtime: Any,
    contracts: dict[str, dict[str, Any]],
    *,
    seed: int,
    progress_every: int,
) -> dict[str, Any]:
    _seed(seed, runtime.torch)
    records = []
    allowed = set(range(EXPECTED_SCENES))
    successes = gotit_false = exceptions = invariant_failures = 0
    visibility_true = visibility_total = 0
    track_counts = []
    for scene_index in range(EXPECTED_SCENES):
        record: dict[str, Any] = {"scene_index": scene_index}
        try:
            sample, gotit = runtime.dataset[scene_index]
            record["gotit"] = bool(gotit)
            metadata = _metadata(sample, batched=False)
            record["metadata"] = metadata
            failures = validate_provenance(
                metadata,
                gotit=bool(gotit),
                allowed_indices=allowed,
                contracts=contracts,
            )
            if int(metadata["virtual_index"]) != scene_index:
                failures.append("coverage returned the wrong virtual index")
            if gotit:
                validation = validate_sample(
                    sample,
                    runtime.torch,
                    batched=False,
                    full=True,
                )
                failures.extend(validation["failures"])
                record["statistics"] = validation["statistics"]
                stats = validation["statistics"]
                track_counts.append(int(stats["track_count"]))
                visibility_true += int(stats["visibility_true"])
                visibility_total += int(stats["visibility_total"])
                successes += 1
            else:
                gotit_false += 1
            record["failures"] = failures
            invariant_failures += int(bool(failures))
        except Exception as exc:
            exceptions += 1
            record["error"] = repr(exc)
            record["traceback"] = traceback.format_exc()
        records.append(record)
        if (scene_index + 1) % progress_every == 0 or scene_index == EXPECTED_SCENES - 1:
            print(
                "POINTODYSSEY_COVERAGE "
                f"{scene_index + 1}/{EXPECTED_SCENES} success={successes} "
                f"gotit_false={gotit_false} exceptions={exceptions} "
                f"invariants={invariant_failures}",
                flush=True,
            )
        if "sample" in locals():
            del sample
    return {
        "attempted": EXPECTED_SCENES,
        "successes": successes,
        "gotit_false": gotit_false,
        "exceptions": exceptions,
        "invariant_failures": invariant_failures,
        "all_78_scenes_load": (
            successes == EXPECTED_SCENES
            and gotit_false == 0
            and exceptions == 0
            and invariant_failures == 0
        ),
        "track_count_min": min(track_counts) if track_counts else None,
        "track_count_max": max(track_counts) if track_counts else None,
        "visibility_true": visibility_true,
        "visibility_total": visibility_total,
        "visibility_ratio": (
            visibility_true / visibility_total if visibility_total else None
        ),
        "records": records,
    }


def make_loader(
    runtime: Any,
    indices: Sequence[int],
    *,
    workers: int,
    seed: int,
):
    kwargs = {
        "dataset": runtime.dataset,
        "batch_size": 1,
        "sampler": FixedIndexSampler(indices),
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": True,
        "collate_fn": runtime.collate_fn,
        "drop_last": True,
        "in_order": False,
        "generator": runtime.torch.Generator().manual_seed(seed),
    }
    if workers > 0:
        kwargs["prefetch_factor"] = PREFETCH_FACTOR
    return runtime.StatefulDataLoader(**kwargs)


def run_lane(
    runtime: Any,
    contracts: dict[str, dict[str, Any]],
    *,
    workers: int,
    warmup_indices: Sequence[int],
    measured_indices: Sequence[int],
    tail_indices: Sequence[int],
    seed: int,
    progress_every: int,
) -> dict[str, Any]:
    _seed(seed, runtime.torch)
    all_indices = list(warmup_indices) + list(measured_indices) + list(tail_indices)
    allowed = set(all_indices)
    loader = make_loader(runtime, all_indices, workers=workers, seed=seed)
    tracker = ResourceTracker()
    monitor_started = time.perf_counter()
    iterator = iter(loader)
    returned_indices: set[int] = set()
    warmup_failures = 0
    lane_error = None
    returned = successes = gotit_false = invariant_failures = 0
    track_counts: list[int] = []
    visibility_true = visibility_total = 0
    successful_metadata: list[dict[str, Any]] = []
    measured_started = None
    try:
        for _ in warmup_indices:
            batch, gotit = next(iterator)
            metadata = _metadata(batch, batched=True)
            failures = validate_provenance(
                metadata,
                gotit=bool(gotit[0]),
                allowed_indices=allowed,
                contracts=contracts,
            )
            virtual_index = int(metadata["virtual_index"])
            if virtual_index in returned_indices:
                failures.append("duplicate virtual index")
            returned_indices.add(virtual_index)
            warmup_failures += int(not bool(gotit[0]) or bool(failures))

        measured_started = time.perf_counter()
        for ordinal in range(len(measured_indices)):
            batch, gotit = next(iterator)
            returned += 1
            metadata = _metadata(batch, batched=True)
            failures = validate_provenance(
                metadata,
                gotit=bool(gotit[0]),
                allowed_indices=allowed,
                contracts=contracts,
            )
            virtual_index = int(metadata["virtual_index"])
            if virtual_index in returned_indices:
                failures.append("duplicate virtual index")
            returned_indices.add(virtual_index)
            if gotit[0]:
                successes += 1
                validation = validate_sample(
                    batch, runtime.torch, batched=True, full=False
                )
                failures.extend(validation["failures"])
                stats = validation["statistics"]
                track_counts.append(int(stats["track_count"]))
                visibility_true += int(stats["visibility_true"])
                visibility_total += int(stats["visibility_total"])
                if "window_start" in metadata:
                    successful_metadata.append(metadata)
            else:
                gotit_false += 1
            invariant_failures += int(bool(failures))
            completed = ordinal + 1
            if completed % progress_every == 0 or completed == len(measured_indices):
                tracker.sample()
                elapsed = time.perf_counter() - measured_started
                print(
                    "POINTODYSSEY_BENCHMARK "
                    f"workers={workers} {completed}/{len(measured_indices)} "
                    f"success={successes} gotit_false={gotit_false} "
                    f"invariants={invariant_failures} "
                    f"samples_per_second={completed / elapsed:.3f}",
                    flush=True,
                )
    except Exception as exc:
        lane_error = {"error": repr(exc), "traceback": traceback.format_exc()}

    measured_seconds = (
        time.perf_counter() - measured_started
        if measured_started is not None
        else 0.0
    )
    resources = tracker.stop()
    monitor_seconds = time.perf_counter() - monitor_started
    del iterator, loader
    gc.collect()
    requested = len(measured_indices)
    resources.update(
        {
            "monitor_seconds": monitor_seconds,
            "physical_read_mib_per_second": (
                resources["physical_read_bytes"] / MIB / monitor_seconds
                if monitor_seconds
                else None
            ),
            "read_characters_mib_per_second": (
                resources["read_characters"] / MIB / monitor_seconds
                if monitor_seconds
                else None
            ),
            "scope": "iterator startup, warmup, measured samples, and tail prefetch",
        }
    )
    return {
        "workers": workers,
        "requested_samples": requested,
        "returned_samples": returned,
        "successes": successes,
        "gotit_false": gotit_false,
        "exceptions": int(lane_error is not None),
        "invariant_failures": invariant_failures,
        "warmup_failures": warmup_failures,
        "measured_seconds": measured_seconds,
        "samples_per_second": (
            successes / measured_seconds if measured_seconds else None
        ),
        "sample_success_rate": successes / requested,
        "sample_failure_rate": (requested - successes) / requested,
        "scene_temporal_diversity": summarize_diversity(successful_metadata),
        "track_count_min": min(track_counts) if track_counts else None,
        "track_count_max": max(track_counts) if track_counts else None,
        "visibility_true": visibility_true,
        "visibility_total": visibility_total,
        "visibility_ratio": (
            visibility_true / visibility_total if visibility_total else None
        ),
        "resources": resources,
        "error": lane_error,
    }


def _lane_is_clean(lane: dict[str, Any]) -> bool:
    return bool(
        lane["returned_samples"] == lane["requested_samples"]
        and lane["successes"] == lane["requested_samples"]
        and lane["gotit_false"] == 0
        and lane["exceptions"] == 0
        and lane["invariant_failures"] == 0
        and lane["warmup_failures"] == 0
    )


def _wandb_log_lane(run: Any, lane: dict[str, Any]) -> None:
    if run is None:
        return
    prefix = f"workers_{lane['workers']}"
    run.log(
        {
            f"{prefix}/samples_per_second": lane["samples_per_second"],
            f"{prefix}/sample_success_rate": lane["sample_success_rate"],
            f"{prefix}/sample_failure_rate": lane["sample_failure_rate"],
            f"{prefix}/invariant_failures": lane["invariant_failures"],
            f"{prefix}/unique_scenes": lane["scene_temporal_diversity"]["unique_scenes"],
            f"{prefix}/unique_scene_start_pairs": lane[
                "scene_temporal_diversity"
            ]["unique_scene_start_pairs"],
            f"{prefix}/sampled_peak_rss_gib": lane["resources"][
                "sampled_peak_process_tree_rss_bytes"
            ]
            / (1024 ** 3),
            f"{prefix}/sampled_peak_rss_increase_gib": lane["resources"][
                "sampled_peak_rss_increase_bytes"
            ]
            / (1024 ** 3),
            f"{prefix}/physical_read_mib_per_second": lane["resources"][
                "physical_read_mib_per_second"
            ],
            f"{prefix}/read_characters_mib_per_second": lane["resources"][
                "read_characters_mib_per_second"
            ],
        }
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    contracts = load_scene_contracts(dataset_root)
    prepared_window_exclusion = {
        "scene_count": len(contracts),
        "scenes_with_excluded_starts": sum(
            int(contract["excluded_start_count"] > 0)
            for contract in contracts.values()
        ),
        "legal_start_count": sum(
            contract["legal_start_count"] for contract in contracts.values()
        ),
        "excluded_start_count": sum(
            contract["excluded_start_count"] for contract in contracts.values()
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=False)

    tail_count = max(args.worker_counts) * PREFETCH_FACTOR
    warmup_indices = build_balanced_schedule(
        args.warmup_samples,
        repeat_offset=100,
        seed=args.seed,
    )
    measured_indices = build_balanced_schedule(
        args.samples_per_worker,
        repeat_offset=200,
        seed=args.seed + 1,
    )
    tail_indices = build_balanced_schedule(
        tail_count,
        repeat_offset=300,
        seed=args.seed + 2,
    )
    run_config = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "settings": {
            "worker_counts": args.worker_counts,
            "batch_size": 1,
            "sequence_length": 24,
            "views": 4,
            "in_order": False,
            "pin_memory": True,
            "prefetch_factor": PREFETCH_FACTOR,
            "warmup_samples": args.warmup_samples,
            "measured_samples_per_worker": args.samples_per_worker,
            "tail_samples": tail_count,
            "skip_coverage": args.skip_coverage,
            "seed": args.seed,
            "cache_state": "system page cache is not dropped",
        },
        "prepared_window_exclusion": prepared_window_exclusion,
    }
    _atomic_json(output_dir / "run_config.json", run_config)
    summary: dict[str, Any] = {
        "status": "running",
        "run_config": run_config,
        "coverage": None,
        "lanes": [],
    }
    _atomic_json(output_dir / "summary.json", summary)

    wandb_run = None
    try:
        runtime = _load_runtime(dataset_root)
        summary["software"] = runtime.versions
        if args.wandb_mode != "disabled":
            import wandb

            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name or output_dir.name,
                mode=args.wandb_mode,
                config=run_config,
                job_type="loader-benchmark",
            )

        if args.skip_coverage:
            summary["coverage"] = {"skipped": True}
            print("POINTODYSSEY_COVERAGE skipped", flush=True)
        else:
            summary["coverage"] = run_coverage(
                runtime,
                contracts,
                seed=args.seed,
                progress_every=args.progress_every,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "coverage/all_78_scenes_load": int(
                            summary["coverage"]["all_78_scenes_load"]
                        ),
                        "coverage/successes": summary["coverage"]["successes"],
                        "coverage/gotit_false": summary["coverage"]["gotit_false"],
                        "coverage/exceptions": summary["coverage"]["exceptions"],
                        "coverage/invariant_failures": summary["coverage"][
                            "invariant_failures"
                        ],
                    }
                )
        _atomic_json(output_dir / "summary.json", summary)

        for workers in args.worker_counts:
            print(
                "POINTODYSSEY_BENCHMARK_START "
                f"workers={workers} warmup={len(warmup_indices)} "
                f"measured={len(measured_indices)} tail={len(tail_indices)}",
                flush=True,
            )
            lane = run_lane(
                runtime,
                contracts,
                workers=workers,
                warmup_indices=warmup_indices,
                measured_indices=measured_indices,
                tail_indices=tail_indices,
                seed=args.seed,
                progress_every=args.progress_every,
            )
            summary["lanes"].append(lane)
            _atomic_json(output_dir / "summary.json", summary)
            _wandb_log_lane(wandb_run, lane)

        clean_lanes = [lane for lane in summary["lanes"] if _lane_is_clean(lane)]
        summary["best_worker_count"] = (
            max(clean_lanes, key=lambda lane: lane["samples_per_second"])["workers"]
            if clean_lanes
            else None
        )
        summary["status"] = (
            "completed"
            if (args.skip_coverage or summary["coverage"]["all_78_scenes_load"])
            and len(clean_lanes) == len(args.worker_counts)
            else "failed"
        )
        summary["finished_at_unix"] = time.time()
        _atomic_json(output_dir / "summary.json", summary)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "benchmark/completed": int(summary["status"] == "completed"),
                    "benchmark/best_worker_count": summary["best_worker_count"],
                }
            )
        print(
            "POINTODYSSEY_BENCHMARK_DONE "
            f"status={summary['status']} best_workers={summary['best_worker_count']} "
            f"output={output_dir}",
            flush=True,
        )
        return 0 if summary["status"] == "completed" else 1
    except Exception as exc:
        summary["status"] = "error"
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        summary["finished_at_unix"] = time.time()
        _atomic_json(output_dir / "summary.json", summary)
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
