#!/usr/bin/env python3
"""Benchmark the real PointOdyssey MV-Tracker training loader.

The benchmark is intentionally separate from training.  It runs one serial
health pass over every prepared training scene, then compares a fixed virtual-
index schedule across worker counts. Full resource metrics use psutil over
Linux procfs, which is available on the target UCL machines.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import multiprocessing
import os
import platform
import random
import queue as queue_module
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Sequence


FORMAT_NAME = "pointodyssey_loader_benchmark"
SCHEMA_VERSION = 1
DEFAULT_WORKERS = (0, 2, 4, 8)
DEFAULT_EXPECTED_SCENES = 78
DEFAULT_SCHEDULE_SEED = 20260714
MIB = 1024 * 1024


class FixedIndexSampler:
    """Yield a precomputed virtual-index schedule unchanged."""

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = tuple(int(index) for index in indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.handle = path.open("x", encoding="utf-8")

    def write(self, event: dict[str, Any]) -> None:
        self.handle.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _worker_counts(values: Sequence[str]) -> tuple[int, ...]:
    counts = tuple(int(value) for value in values)
    if any(count < 0 for count in counts):
        raise argparse.ArgumentTypeError("worker counts must be non-negative")
    if len(set(counts)) != len(counts):
        raise argparse.ArgumentTypeError("worker counts must be unique")
    if not counts:
        raise argparse.ArgumentTypeError("at least one worker count is required")
    return counts


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Parent containing PointOdyssey_MVTracker (for mallard-l: /tmp/thakwani).",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--worker-counts",
        nargs="+",
        default=[str(value) for value in DEFAULT_WORKERS],
        metavar="N",
    )
    parser.add_argument("--expected-scenes", type=_positive_int, default=DEFAULT_EXPECTED_SCENES)
    parser.add_argument("--warmup-samples", type=_nonnegative_int, default=32)
    parser.add_argument("--measured-scene-repeats", type=_positive_int, default=2)
    parser.add_argument("--confirmation-warmup", type=_nonnegative_int, default=32)
    parser.add_argument("--confirmation-samples", type=_nonnegative_int, default=256)
    parser.add_argument("--schedule-seed", type=int, default=DEFAULT_SCHEDULE_SEED)
    parser.add_argument("--monitor-interval", type=_positive_float, default=0.1)
    parser.add_argument("--child-timeout-seconds", type=_positive_float, default=7200.0)
    parser.add_argument("--progress-every", type=_positive_int, default=8)
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--wandb-project", default="mvtracker-loader-benchmark")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    args = parser.parse_args(argv)
    try:
        args.worker_counts = _worker_counts(args.worker_counts)
    except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    return args


def build_balanced_schedule(
    real_len: int,
    count: int,
    *,
    repeat_offset: int,
    seed: int,
) -> list[int]:
    """Build a deterministic, near-balanced virtual-index schedule."""
    if real_len <= 0 or count < 0 or repeat_offset < 0:
        raise ValueError("real_len must be positive; count and repeat_offset must be non-negative")
    rng = random.Random(seed)
    schedule: list[int] = []
    repeat = repeat_offset
    while len(schedule) < count:
        scenes = list(range(real_len))
        rng.shuffle(scenes)
        schedule.extend(scene + repeat * real_len for scene in scenes)
        repeat += 1
    return schedule[:count]


def percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_temporal_diversity(metadata: Iterable[dict[str, Any]]) -> dict[str, Any]:
    starts_by_scene: dict[str, list[int]] = defaultdict(list)
    pairs: list[tuple[str, int]] = []
    for item in metadata:
        scene_name = str(item["scene_name"])
        start = int(item["window_start"])
        starts_by_scene[scene_name].append(start)
        pairs.append((scene_name, start))

    distinct_counts = sorted(len(set(starts)) for starts in starts_by_scene.values())
    total = len(pairs)
    unique_pairs = len(set(pairs))
    return {
        "observations": total,
        "unique_scenes": len(starts_by_scene),
        "unique_scene_start_pairs": unique_pairs,
        "repeated_scene_start_observations": total - unique_pairs,
        "scene_start_repeat_fraction": (total - unique_pairs) / total if total else None,
        "distinct_starts_per_scene": {
            "minimum": min(distinct_counts) if distinct_counts else None,
            "median": percentile(distinct_counts, 0.5),
            "maximum": max(distinct_counts) if distinct_counts else None,
        },
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


def _git_metadata(repo_root: Path) -> dict[str, Any]:
    def run(*command: str) -> str:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "sha": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
        "branch": run("git", "branch", "--show-current"),
    }


def _required_scene_files(scene_root: Path) -> list[Path]:
    metadata = json.loads((scene_root / "scene.json").read_text(encoding="utf-8"))
    frame_count = int(metadata["output"]["frame_count"])
    files = [scene_root / "scene.json", scene_root / "tracks_3d.npy"]
    for view in range(4):
        view_root = scene_root / f"view_{view}"
        files.extend(
            view_root / name
            for name in (
                "depth.npy",
                "intrinsics.npy",
                "extrinsics_w2c.npy",
                "visibility.npy",
            )
        )
        files.extend(view_root / f"rgba_{frame:05d}.jpg" for frame in range(frame_count))
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required prepared files are missing: {missing[:10]}")
    return files


def dataset_tree_fingerprint(root: Path) -> dict[str, Any]:
    """Fingerprint names and stat metadata without reading dataset contents."""
    digest = hashlib.sha256()
    entry_count = 0
    regular_file_count = 0
    regular_file_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        stat = path.lstat()
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            kind = "symlink"
            size = stat.st_size
        elif path.is_dir():
            kind = "directory"
            size = 0
        elif path.is_file():
            kind = "file"
            size = stat.st_size
            regular_file_count += 1
            regular_file_bytes += size
        else:
            kind = "other"
            size = stat.st_size
        digest.update(
            f"{kind}\0{relative}\0{size}\0{stat.st_mtime_ns}\n".encode("utf-8")
        )
        entry_count += 1
    return {
        "sha256": digest.hexdigest(),
        "entry_count": entry_count,
        "regular_file_count": regular_file_count,
        "regular_file_bytes": regular_file_bytes,
        "semantics": "relative path, entry type, byte size, and mtime_ns; file contents are not read",
    }


def preflight_dataset(dataset_root: Path, expected_scenes: int) -> dict[str, Any]:
    prepared_root = dataset_root / "PointOdyssey_MVTracker"
    train_root = prepared_root / "train"
    report_path = prepared_root / "validation_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "completed" or report.get("failures") != []:
        raise ValueError("prepared validation report is not a clean completed report")
    scene_roots = sorted(
        (path for path in train_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    expected_names = [f"{index:06d}" for index in range(expected_scenes)]
    actual_names = [path.name for path in scene_roots]
    if actual_names != expected_names:
        raise ValueError(f"training scenes do not match {expected_names[0]}..{expected_names[-1]}")
    scene_bytes: dict[str, int] = {}
    scene_contracts: dict[str, dict[str, Any]] = {}
    for scene_root in scene_roots:
        files = _required_scene_files(scene_root)
        metadata = json.loads((scene_root / "scene.json").read_text(encoding="utf-8"))
        frame_count = int(metadata["output"]["frame_count"])
        invalid_frames = [int(frame) for frame in metadata["output"]["rgb"]["invalid_frame_indices"]]
        scene_bytes[scene_root.name] = sum(path.stat().st_size for path in files)
        scene_contracts[scene_root.name] = {
            "frame_count": frame_count,
            "invalid_rgb_frame_indices": invalid_frames,
            "decoded_camera_frames_per_sample": frame_count * 4,
        }
    return {
        "prepared_root": str(prepared_root),
        "train_root": str(train_root),
        "report_path": str(report_path),
        "scene_names": actual_names,
        "scene_bytes": scene_bytes,
        "scene_contracts": scene_contracts,
        "tree_fingerprint_before": dataset_tree_fingerprint(train_root),
    }


class ProcfsMonitor:
    """Poll aggregate RSS and I/O for the benchmark process and loader workers."""

    def __init__(self, root_pid: int, interval: float) -> None:
        if not Path("/proc/self/stat").is_file():
            raise RuntimeError("full benchmark metrics require Linux procfs")
        import psutil

        self.psutil = psutil
        self.root_pid = root_pid
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.baseline: dict[tuple[int, int], dict[str, int]] = {}
        self.latest: dict[tuple[int, int], dict[str, int]] = {}
        self.baseline_rss_bytes = 0
        self.peak_rss_bytes = 0
        self.peak_process_count = 0
        self.observed_pids: set[int] = set()

    def _snapshot(self) -> dict[tuple[int, int], dict[str, int]]:
        psutil = self.psutil
        try:
            root = psutil.Process(self.root_pid)
            processes = [root, *root.children(recursive=True)]
        except psutil.Error:
            return {}
        snapshot: dict[tuple[int, int], dict[str, int]] = {}
        for process in processes:
            try:
                with process.oneshot():
                    identity = (process.pid, int(process.create_time() * 1_000_000))
                    rss_bytes = int(process.memory_info().rss)
                    io = process.io_counters()
                snapshot[identity] = {
                    "rss_bytes": rss_bytes,
                    "read_bytes": int(io.read_bytes),
                    "rchar": int(io.read_chars),
                }
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
        return snapshot

    def _sample(self, *, establish_baseline: bool = False) -> None:
        snapshot = self._snapshot()
        if establish_baseline:
            self.baseline = {identity: dict(values) for identity, values in snapshot.items()}
            self.baseline_rss_bytes = sum(
                values["rss_bytes"] for values in snapshot.values()
            )
        for identity, values in snapshot.items():
            if identity not in self.baseline:
                self.baseline[identity] = {
                    **values,
                    "read_bytes": 0,
                    "rchar": 0,
                }
            self.latest[identity] = dict(values)
            self.observed_pids.add(identity[0])
        self.peak_rss_bytes = max(
            self.peak_rss_bytes,
            sum(values["rss_bytes"] for values in snapshot.values()),
        )
        self.peak_process_count = max(self.peak_process_count, len(snapshot))

    def start(self) -> None:
        self._sample(establish_baseline=True)

        def poll() -> None:
            while not self.stop_event.wait(self.interval):
                self._sample()

        self.thread = threading.Thread(target=poll, name="procfs-monitor", daemon=True)
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
        self._sample()
        read_bytes = 0
        rchar = 0
        for identity, latest in self.latest.items():
            baseline = self.baseline[identity]
            read_bytes += max(0, latest["read_bytes"] - baseline["read_bytes"])
            rchar += max(0, latest["rchar"] - baseline["rchar"])
        return {
            "baseline_process_tree_rss_bytes": self.baseline_rss_bytes,
            "peak_process_tree_rss_bytes": self.peak_rss_bytes,
            "peak_rss_increase_bytes": max(
                0,
                self.peak_rss_bytes - self.baseline_rss_bytes,
            ),
            "physical_read_bytes": read_bytes,
            "read_characters": rchar,
            "observed_pids": sorted(self.observed_pids),
            "observed_process_count": len(self.observed_pids),
            "peak_process_count": self.peak_process_count,
            "peak_descendant_count": max(0, self.peak_process_count - 1),
            "rss_semantics": "sum of per-process VmRSS; shared pages may be counted more than once",
            "physical_read_semantics": "psutil read_bytes over the process tree; page-cache hits may report zero",
            "read_characters_semantics": "psutil read_chars includes reads satisfied from page cache and non-dataset files",
        }


def _load_runtime(dataset_root: Path):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from omegaconf import OmegaConf
    from torchdata.stateful_dataloader import StatefulDataLoader

    import cv2
    import numpy as np
    import torch
    import torchvision

    from mvtracker.datasets.pointodyssey_multiview_dataset import PointOdysseyMultiViewDataset
    from mvtracker.datasets.utils import collate_fn

    config = OmegaConf.merge(
        OmegaConf.load(repo_root / "configs" / "train.yaml"),
        OmegaConf.load(repo_root / "configs" / "experiment" / "pointodyssey.yaml"),
    )
    config.datasets.root = str(dataset_root)
    fabric_stub = SimpleNamespace(world_size=1)
    dataset = PointOdysseyMultiViewDataset.from_name(
        config.datasets.train.name,
        str(dataset_root),
        training_args=config,
        fabric=fabric_stub,
    )
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
    }
    return SimpleNamespace(
        config=config,
        dataset=dataset,
        torch=torch,
        StatefulDataLoader=StatefulDataLoader,
        collate_fn=collate_fn,
        versions=versions,
    )


def _tensor_bytes(sample: Any) -> int:
    fields = (
        "video",
        "videodepth",
        "segmentation",
        "trajectory",
        "trajectory_3d",
        "visibility",
        "valid",
        "intrs",
        "extrs",
        "query_points_3d",
    )
    total = 0
    for field in fields:
        tensor = getattr(sample, field, None)
        if tensor is not None:
            total += int(tensor.numel() * tensor.element_size())
    return total


def _validate_sample(sample: Any, torch: Any, *, batched: bool, full_scan: bool) -> dict[str, Any]:
    offset = 1 if batched else 0
    video = sample.video
    depth = sample.videodepth
    segmentation = sample.segmentation
    trajectory = sample.trajectory
    trajectory_3d = sample.trajectory_3d
    visibility = sample.visibility
    valid = sample.valid
    intrs = sample.intrs
    extrs = sample.extrs
    queries = sample.query_points_3d
    expected_prefix = (1,) if batched else ()
    n_tracks = int(queries.shape[offset])
    expected = {
        "video": expected_prefix + (4, 24, 3, 384, 512),
        "videodepth": expected_prefix + (4, 24, 1, 384, 512),
        "segmentation": expected_prefix + (4, 24, 1, 384, 512),
        "trajectory": expected_prefix + (4, 24, n_tracks, 3),
        "trajectory_3d": expected_prefix + (24, n_tracks, 3),
        "visibility": expected_prefix + (4, 24, n_tracks),
        "valid": expected_prefix + (24, n_tracks),
        "intrs": expected_prefix + (4, 24, 3, 3),
        "extrs": expected_prefix + (4, 24, 3, 4),
        "query_points_3d": expected_prefix + (n_tracks, 4),
    }
    actual = {
        "video": tuple(video.shape),
        "videodepth": tuple(depth.shape),
        "segmentation": tuple(segmentation.shape),
        "trajectory": tuple(trajectory.shape),
        "trajectory_3d": tuple(trajectory_3d.shape),
        "visibility": tuple(visibility.shape),
        "valid": tuple(valid.shape),
        "intrs": tuple(intrs.shape),
        "extrs": tuple(extrs.shape),
        "query_points_3d": tuple(queries.shape),
    }
    failures = [f"{name} shape {actual[name]} != {shape}" for name, shape in expected.items() if actual[name] != shape]
    expected_dtypes = {
        "video": torch.float32,
        "videodepth": torch.float32,
        "segmentation": torch.float32,
        "trajectory": torch.float32,
        "trajectory_3d": torch.float32,
        "visibility": torch.bool,
        "valid": torch.float32,
        "intrs": torch.float32,
        "extrs": torch.float32,
        "query_points_3d": torch.float32,
    }
    for name, expected_dtype in expected_dtypes.items():
        actual_dtype = getattr(sample, name).dtype
        if actual_dtype != expected_dtype:
            failures.append(f"{name} dtype {actual_dtype} != {expected_dtype}")

    stats: dict[str, Any] = {
        "shapes": {name: list(shape) for name, shape in actual.items()},
        "dtypes": {
            name: str(getattr(sample, name).dtype)
            for name in actual
        },
        "pinned": {
            name: bool(getattr(sample, name).is_pinned())
            for name in actual
        },
        "track_count": n_tracks,
        "tensor_bytes": _tensor_bytes(sample),
    }
    if full_scan and not failures:
        finite_fields = (
            video,
            depth,
            segmentation,
            trajectory,
            trajectory_3d,
            valid,
            intrs,
            extrs,
            queries,
        )
        if not all(bool(torch.isfinite(tensor).all()) for tensor in finite_fields):
            failures.append("one or more floating tensors contain nonfinite values")
        if float(video.min()) < 0.0 or float(video.max()) > 255.0:
            failures.append("video values are outside [0, 255]")
        if float(depth.min()) < 0.0:
            failures.append("depth contains negative values")
        query_times = queries[..., 0]
        if not bool((query_times == query_times.round()).all()):
            failures.append("query times are not integer-valued")
        if float(query_times.min()) < 0.0 or float(query_times.max()) > 23.0:
            failures.append("query times are outside [0, 23]")
        if batched:
            times = query_times[0].long()
            expected_xyz = trajectory_3d[0, times, torch.arange(n_tracks), :]
            query_xyz = queries[0, :, 1:]
        else:
            times = query_times.long()
            expected_xyz = trajectory_3d[times, torch.arange(n_tracks), :]
            query_xyz = queries[:, 1:]
        if not bool(torch.allclose(query_xyz, expected_xyz, atol=1e-4, rtol=1e-5)):
            failures.append("query XYZ does not match trajectory_3d at query time")
        valid_depth_ratio = float((depth > 0.0).float().mean())
        if valid_depth_ratio < 0.1:
            failures.append("valid depth ratio is below 0.1")
        stats.update(
            {
                "video_min": float(video.min()),
                "video_max": float(video.max()),
                "depth_min": float(depth.min()),
                "depth_max": float(depth.max()),
                "valid_depth_ratio": valid_depth_ratio,
                "visibility_true": int(visibility.sum()),
                "valid_true": int(valid.sum()),
            }
        )
    return {"failures": failures, "statistics": stats}


def _metadata_from_sample(sample: Any, *, batched: bool) -> dict[str, Any]:
    metadata = sample.sample_metadata
    if batched:
        if not isinstance(metadata, list) or len(metadata) != 1:
            raise ValueError("collated sample metadata must be a one-item list")
        metadata = metadata[0]
    if not isinstance(metadata, dict):
        raise ValueError("sample metadata is missing")
    return dict(metadata)


def _validate_provenance(
    metadata: dict[str, Any],
    *,
    expected_gotit: bool,
    expected_virtual_index: int | None,
    allowed_virtual_indices: set[int],
    scene_contracts: dict[str, dict[str, Any]],
    real_len: int,
    seed_base: int,
) -> list[str]:
    failures: list[str] = []
    required = ("virtual_index", "scene_index", "scene_name", "gotit")
    missing = [key for key in required if key not in metadata]
    if missing:
        return [f"sample metadata is missing {missing}"]
    virtual_index = int(metadata["virtual_index"])
    scene_index = int(metadata["scene_index"])
    scene_name = str(metadata["scene_name"])
    if virtual_index not in allowed_virtual_indices:
        failures.append(f"virtual index {virtual_index} is outside the requested schedule")
    if expected_virtual_index is not None and virtual_index != expected_virtual_index:
        failures.append(
            f"returned virtual index {virtual_index} != requested {expected_virtual_index}"
        )
    expected_scene_index = virtual_index % real_len
    expected_scene_name = f"{expected_scene_index:06d}"
    if scene_index != expected_scene_index:
        failures.append(f"scene index {scene_index} != virtual_index % {real_len}")
    if scene_name != expected_scene_name:
        failures.append(f"scene name {scene_name!r} != {expected_scene_name!r}")
    if scene_name not in scene_contracts:
        failures.append(f"scene name {scene_name!r} has no preflight contract")
        return failures
    metadata_gotit = bool(metadata["gotit"])
    if metadata_gotit != expected_gotit:
        failures.append(
            f"metadata gotit={metadata_gotit} != loader gotit={expected_gotit}"
        )
    if not expected_gotit:
        return failures

    success_required = (
        "seed",
        "window_start",
        "window_end_exclusive",
        "selected_views",
    )
    missing = [key for key in success_required if key not in metadata]
    if missing:
        return failures + [f"successful sample metadata is missing {missing}"]
    if int(metadata["seed"]) != seed_base + virtual_index:
        failures.append(
            f"seed {metadata['seed']} != {seed_base} + virtual index {virtual_index}"
        )
    start = int(metadata["window_start"])
    end = int(metadata["window_end_exclusive"])
    contract = scene_contracts[scene_name]
    frame_count = int(contract["frame_count"])
    if end - start != 24:
        failures.append(f"window [{start}, {end}) does not contain 24 frames")
    if start < 0 or end > frame_count:
        failures.append(f"window [{start}, {end}) is outside [0, {frame_count})")
    invalid_frames = set(int(frame) for frame in contract["invalid_rgb_frame_indices"])
    intersecting = sorted(frame for frame in invalid_frames if start <= frame < end)
    if intersecting:
        failures.append(f"window includes invalid RGB frames {intersecting}")
    selected_views = sorted(int(view) for view in metadata["selected_views"])
    if selected_views != [0, 1, 2, 3]:
        failures.append(f"selected views {selected_views} != [0, 1, 2, 3]")
    return failures


def _coverage_child(payload: dict[str, Any], queue: Any) -> None:
    os.setsid()
    writer = JsonlWriter(Path(payload["events_path"]))
    try:
        runtime = _load_runtime(Path(payload["dataset_root"]))
        _seed_process(int(payload["schedule_seed"]), runtime.torch)
        dataset = runtime.dataset
        expected_scenes = int(payload["expected_scenes"])
        if dataset.real_len != expected_scenes:
            raise ValueError(f"loader discovered {dataset.real_len} scenes, expected {expected_scenes}")
        successes = gotit_false = exceptions = invariant_failures = 0
        scenes_seen: set[str] = set()
        allowed_virtual_indices = set(range(expected_scenes))
        started = time.perf_counter()
        for scene_index in range(expected_scenes):
            event: dict[str, Any] = {"kind": "coverage", "scene_index": scene_index}
            item_start = time.perf_counter()
            try:
                sample, gotit = dataset[scene_index]
                event["latency_seconds"] = time.perf_counter() - item_start
                event["gotit"] = bool(gotit)
                metadata = _metadata_from_sample(sample, batched=False)
                event["sample_metadata"] = metadata
                scenes_seen.add(str(metadata["scene_name"]))
                provenance_failures = _validate_provenance(
                    metadata,
                    expected_gotit=bool(gotit),
                    expected_virtual_index=scene_index,
                    allowed_virtual_indices=allowed_virtual_indices,
                    scene_contracts=payload["scene_contracts"],
                    real_len=expected_scenes,
                    seed_base=int(dataset.seed),
                )
                if gotit:
                    validation = _validate_sample(sample, runtime.torch, batched=False, full_scan=True)
                    validation["failures"].extend(provenance_failures)
                    event["validation"] = validation
                    successes += 1
                    invariant_failures += int(bool(validation["failures"]))
                else:
                    gotit_false += 1
                    event["provenance_failures"] = provenance_failures
                    invariant_failures += int(bool(provenance_failures))
            except BaseException as exc:
                exceptions += 1
                event.update(
                    {
                        "gotit": None,
                        "latency_seconds": time.perf_counter() - item_start,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            writer.write(event)
            if (scene_index + 1) % int(payload["progress_every"]) == 0 or scene_index + 1 == expected_scenes:
                print(
                    "POINTODYSSEY_LOADER_COVERAGE "
                    f"completed={scene_index + 1}/{expected_scenes} successes={successes} "
                    f"gotit_false={gotit_false} exceptions={exceptions} "
                    f"invariant_failures={invariant_failures}",
                    flush=True,
                )
        result = {
            "status": "completed",
            "attempted": expected_scenes,
            "successes": successes,
            "gotit_false": gotit_false,
            "exceptions": exceptions,
            "invariant_failures": invariant_failures,
            "unique_scenes": len(scenes_seen),
            "all_scenes_seen": len(scenes_seen) == expected_scenes,
            "wall_seconds": time.perf_counter() - started,
            "schedule_seed": int(payload["schedule_seed"]),
            "software": runtime.versions,
        }
        queue.put(result)
    except BaseException as exc:
        queue.put({"status": "error", "error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        writer.close()


def _seed_process(seed: int, torch: Any) -> None:
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_loader(
    runtime: Any,
    indices: Sequence[int],
    workers: int,
    *,
    in_order: bool,
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
        "in_order": in_order,
        "persistent_workers": False,
        "generator": runtime.torch.Generator().manual_seed(seed),
    }
    if workers > 0:
        kwargs["prefetch_factor"] = 4
    return runtime.StatefulDataLoader(**kwargs)


def _trial_child(payload: dict[str, Any], queue: Any) -> None:
    os.setsid()
    writer = JsonlWriter(Path(payload["events_path"]))
    loader = iterator = None
    try:
        runtime = _load_runtime(Path(payload["dataset_root"]))
        _seed_process(int(payload["schedule_seed"]), runtime.torch)
        if runtime.dataset.real_len != int(payload["expected_scenes"]):
            raise ValueError("loader scene count changed after preflight")
        workers = int(payload["workers"])
        in_order = bool(payload["in_order"])
        configured_in_order = bool(runtime.config.reproducibility.deterministic)
        if in_order != configured_in_order:
            raise ValueError(
                f"benchmark in_order={in_order} does not match training in_order={configured_in_order}"
            )
        warmup_indices = list(payload["warmup_indices"])
        measured_indices = list(payload["measured_indices"])
        tail_indices = list(payload["tail_indices"])
        indices = warmup_indices + measured_indices + tail_indices
        allowed_virtual_indices = set(indices)
        scene_contracts = payload["scene_contracts"]
        loader = _make_loader(
            runtime,
            indices,
            workers,
            in_order=in_order,
            seed=int(payload["schedule_seed"]),
        )

        monitor = ProcfsMonitor(os.getpid(), float(payload["monitor_interval"]))
        resource_started = time.perf_counter()
        monitor.start()
        iterator_started = time.perf_counter()
        iterator = iter(loader)
        iterator_creation_seconds = time.perf_counter() - iterator_started

        first_next_wait_seconds = None
        time_to_first_batch_seconds = None
        warmup_gotit_false = warmup_invariant_failures = 0
        returned_virtual_indices: set[int] = set()
        for warmup_ordinal, virtual_index in enumerate(warmup_indices):
            started = time.perf_counter()
            batch, gotit = next(iterator)
            latency = time.perf_counter() - started
            if first_next_wait_seconds is None:
                first_next_wait_seconds = latency
                time_to_first_batch_seconds = time.perf_counter() - iterator_started
            warmup_gotit_false += int(not bool(gotit[0]))
            metadata = _metadata_from_sample(batch, batched=True)
            provenance_failures = _validate_provenance(
                metadata,
                expected_gotit=bool(gotit[0]),
                expected_virtual_index=virtual_index if in_order else None,
                allowed_virtual_indices=allowed_virtual_indices,
                scene_contracts=scene_contracts,
                real_len=int(payload["expected_scenes"]),
                seed_base=int(runtime.dataset.seed),
            )
            actual_virtual_index = int(metadata["virtual_index"])
            if actual_virtual_index in returned_virtual_indices:
                provenance_failures.append(
                    f"virtual index {actual_virtual_index} was returned more than once"
                )
            returned_virtual_indices.add(actual_virtual_index)
            warmup_invariant_failures += int(bool(provenance_failures))

        latencies: list[float] = []
        metadata_items: list[dict[str, Any]] = []
        events: list[dict[str, Any]] = []
        successes = gotit_false = invariant_failures = 0
        tensor_bytes = logical_scene_bytes = decoded_camera_frames = 0
        pinned_video_samples = pinned_depth_samples = 0
        trial_error: dict[str, Any] | None = None
        measured_started = time.perf_counter()
        for measured_index, virtual_index in enumerate(measured_indices):
            event: dict[str, Any] = {
                "kind": "trial_sample",
                "workers": workers,
                "in_order": in_order,
                "ordinal": measured_index,
                "virtual_index_expected": int(virtual_index) if in_order else None,
            }
            item_started = time.perf_counter()
            try:
                batch, gotit = next(iterator)
                latency = time.perf_counter() - item_started
                if first_next_wait_seconds is None:
                    first_next_wait_seconds = latency
                    time_to_first_batch_seconds = time.perf_counter() - iterator_started
                latencies.append(latency)
                event["loader_wait_seconds"] = latency
                event["gotit"] = bool(gotit[0])
                metadata = _metadata_from_sample(batch, batched=True)
                event["sample_metadata"] = metadata
                provenance_failures = _validate_provenance(
                    metadata,
                    expected_gotit=bool(gotit[0]),
                    expected_virtual_index=virtual_index if in_order else None,
                    allowed_virtual_indices=allowed_virtual_indices,
                    scene_contracts=scene_contracts,
                    real_len=int(payload["expected_scenes"]),
                    seed_base=int(runtime.dataset.seed),
                )
                actual_virtual_index = int(metadata["virtual_index"])
                if actual_virtual_index in returned_virtual_indices:
                    provenance_failures.append(
                        f"virtual index {actual_virtual_index} was returned more than once"
                    )
                returned_virtual_indices.add(actual_virtual_index)
                if "window_start" in metadata:
                    metadata_items.append(metadata)
                scene_name = str(metadata["scene_name"])
                logical_scene_bytes += int(payload["scene_bytes"][scene_name])
                decoded_camera_frames += int(
                    scene_contracts[scene_name]["decoded_camera_frames_per_sample"]
                )
                if gotit[0]:
                    validation = _validate_sample(batch, runtime.torch, batched=True, full_scan=False)
                    validation["failures"].extend(provenance_failures)
                    event["validation"] = validation
                    successes += 1
                    invariant_failures += int(bool(validation["failures"]))
                    tensor_bytes += int(validation["statistics"]["tensor_bytes"])
                    pinned_video_samples += int(validation["statistics"]["pinned"]["video"])
                    pinned_depth_samples += int(validation["statistics"]["pinned"]["videodepth"])
                else:
                    gotit_false += 1
                    event["provenance_failures"] = provenance_failures
                    invariant_failures += int(bool(provenance_failures))
                events.append(event)
            except BaseException as exc:
                trial_error = {"error": repr(exc), "traceback": traceback.format_exc()}
                event.update(trial_error)
                events.append(event)
                break
            if (measured_index + 1) % int(payload["progress_every"]) == 0:
                elapsed = time.perf_counter() - measured_started
                print(
                    "POINTODYSSEY_LOADER_TRIAL "
                    f"workers={payload['workers']} completed={measured_index + 1}/"
                    f"{len(payload['measured_indices'])} successes={successes} "
                    f"gotit_false={gotit_false} samples_per_second={(measured_index + 1) / elapsed:.3f}",
                    flush=True,
                )
        wall_seconds = time.perf_counter() - measured_started
        resources = monitor.stop()
        resource_wall_seconds = time.perf_counter() - resource_started
        for event in events:
            writer.write(event)
        returned = len(latencies)
        attempted = len(events)
        exception_failures = int(trial_error is not None)
        result = {
            "status": "error" if trial_error else "completed",
            "workers": workers,
            "in_order": in_order,
            "schedule_seed": int(payload["schedule_seed"]),
            "warmup_samples": len(warmup_indices),
            "warmup_gotit_false": warmup_gotit_false,
            "warmup_invariant_failures": warmup_invariant_failures,
            "tail_samples_queued": len(tail_indices),
            "requested_measured_samples": len(measured_indices),
            "attempted_measured_samples": attempted,
            "returned_measured_samples": returned,
            "successes": successes,
            "gotit_false": gotit_false,
            "exception_failures": exception_failures,
            "sample_success_rate": successes / attempted if attempted else None,
            "sample_failure_rate": (
                (gotit_false + exception_failures) / attempted if attempted else None
            ),
            "sample_rate_semantics": (
                "success means gotit=True; failure means gotit=False or a loader exception; "
                "tensor/provenance invariant failures are reported separately"
            ),
            "invariant_failures": invariant_failures,
            "wall_seconds": wall_seconds,
            "attempted_samples_per_second": attempted / wall_seconds if wall_seconds else None,
            "returned_samples_per_second": returned / wall_seconds if wall_seconds else None,
            "successful_samples_per_second": successes / wall_seconds if wall_seconds else None,
            "returned_camera_frames_per_second": successes * 4 * 24 / wall_seconds if wall_seconds else None,
            "returned_sample_source_camera_frames": decoded_camera_frames,
            "returned_sample_source_camera_frames_per_measured_second": (
                decoded_camera_frames / wall_seconds if wall_seconds else None
            ),
            "iterator_creation_seconds": iterator_creation_seconds,
            "first_next_wait_seconds": first_next_wait_seconds,
            "time_to_first_batch_seconds": time_to_first_batch_seconds,
            "loader_wait_seconds": {
                "total": sum(latencies),
                "p50": percentile(latencies, 0.50),
                "p90": percentile(latencies, 0.90),
                "p95": percentile(latencies, 0.95),
                "p99": percentile(latencies, 0.99),
                "maximum": max(latencies) if latencies else None,
            },
            "tensor_bytes": tensor_bytes,
            "tensor_mib_per_second": tensor_bytes / MIB / wall_seconds if wall_seconds else None,
            "logical_prepared_bytes": logical_scene_bytes,
            "logical_prepared_mib_per_second": logical_scene_bytes / MIB / wall_seconds if wall_seconds else None,
            "logical_prepared_rate_semantics": (
                "sum of whole prepared-scene file sizes associated with returned measured samples, "
                "divided by measured delivery time; this is not physical disk traffic"
            ),
            "pin_memory": {
                "configured": True,
                "video_samples_pinned": pinned_video_samples,
                "depth_samples_pinned": pinned_depth_samples,
                "successful_samples": successes,
            },
            "timed_validation_level": (
                "per-sample tensor shapes/dtypes, query count, visibility dtype, provenance, and pin state; "
                "the serial coverage pass performs full tensor finiteness/value/query checks"
            ),
            "temporal_diversity": summarize_temporal_diversity(metadata_items),
            "resources": resources,
            "resource_monitor_wall_seconds": resource_wall_seconds,
            "resource_measurement_scope": (
                "iterator startup, warmup, measured delivery, and speculative tail prefetch; "
                "worker shutdown is excluded"
            ),
            "software": runtime.versions,
            "error": trial_error,
        }
        result["resources"]["physical_read_mib_per_second_full_run"] = (
            resources["physical_read_bytes"] / MIB / resource_wall_seconds
            if resource_wall_seconds
            else None
        )
        result["resources"]["read_characters_mib_per_second_full_run"] = (
            resources["read_characters"] / MIB / resource_wall_seconds
            if resource_wall_seconds
            else None
        )
        queue.put(result)
    except BaseException as exc:
        queue.put({"status": "error", "workers": payload.get("workers"), "error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        writer.close()
        del iterator, loader
        gc.collect()


def _run_child(
    target: Any,
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=target, args=(payload, result_queue))
    process.start()
    result = None
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if result is None:
            try:
                result = result_queue.get(timeout=min(1.0, max(0.01, deadline - time.monotonic())))
            except queue_module.Empty:
                pass
        if result is not None and not process.is_alive():
            break
        if result is None and not process.is_alive():
            break
    timed_out = process.is_alive() and time.monotonic() >= deadline
    if timed_out:
        group_signalled = False
        try:
            os.killpg(process.pid, signal.SIGTERM)
            group_signalled = True
        except ProcessLookupError:
            if process.is_alive():
                process.terminate()
        process.join(timeout=30.0)
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            if process.is_alive():
                process.kill()
        process.join(timeout=30.0)
        result = {
            "status": "error",
            "error": f"child exceeded timeout of {timeout_seconds:.1f} seconds",
            "timed_out": True,
            "process_group_signalled": group_signalled,
        }
    else:
        process.join()
    if result is None:
        try:
            result = result_queue.get(timeout=1.0)
        except queue_module.Empty:
            result = {
                "status": "error",
                "error": f"child exited with code {process.exitcode} without returning a result",
            }
    result["child_exit_code"] = process.exitcode
    if process.exitcode != 0 and result.get("status") != "error":
        result["status"] = "error"
        result["error"] = f"child exited with code {process.exitcode}"
    return result


def _wandb_init(args: argparse.Namespace, config: dict[str, Any]):
    if args.wandb_mode == "disabled":
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or args.output_dir.name,
        mode=args.wandb_mode,
        config=config,
        job_type="loader-benchmark",
    )


def _wandb_log(run: Any, prefix: str, result: dict[str, Any]) -> None:
    if run is None:
        return
    metrics = {
        f"{prefix}/successful_samples_per_second": result.get("successful_samples_per_second"),
        f"{prefix}/returned_samples_per_second": result.get("returned_samples_per_second"),
        f"{prefix}/attempted_samples_per_second": result.get("attempted_samples_per_second"),
        f"{prefix}/sample_success_rate": result.get("sample_success_rate"),
        f"{prefix}/sample_failure_rate": result.get("sample_failure_rate"),
        f"{prefix}/gotit_false": result.get("gotit_false"),
        f"{prefix}/exception_failures": result.get("exception_failures"),
        f"{prefix}/invariant_failures": result.get("invariant_failures"),
        f"{prefix}/loader_wait_p50_seconds": (result.get("loader_wait_seconds") or {}).get("p50"),
        f"{prefix}/loader_wait_p95_seconds": (result.get("loader_wait_seconds") or {}).get("p95"),
        f"{prefix}/logical_prepared_mib_per_second": result.get("logical_prepared_mib_per_second"),
        f"{prefix}/source_camera_frames_per_measured_second": result.get(
            "returned_sample_source_camera_frames_per_measured_second"
        ),
        f"{prefix}/time_to_first_batch_seconds": result.get("time_to_first_batch_seconds"),
        f"{prefix}/video_samples_pinned": (result.get("pin_memory") or {}).get(
            "video_samples_pinned"
        ),
        f"{prefix}/depth_samples_pinned": (result.get("pin_memory") or {}).get(
            "depth_samples_pinned"
        ),
        f"{prefix}/unique_scene_start_pairs": (
            result.get("temporal_diversity") or {}
        ).get("unique_scene_start_pairs"),
    }
    resources = result.get("resources") or {}
    metrics[f"{prefix}/peak_process_tree_rss_gib"] = resources.get("peak_process_tree_rss_bytes", 0) / (1024 ** 3)
    metrics[f"{prefix}/physical_read_mib_per_second_full_run"] = resources.get(
        "physical_read_mib_per_second_full_run"
    )
    metrics[f"{prefix}/peak_descendant_count"] = resources.get("peak_descendant_count")
    run.log({key: value for key, value in metrics.items() if value is not None})


def _trial_is_clean(trial: dict[str, Any]) -> bool:
    return bool(
        trial.get("status") == "completed"
        and trial.get("gotit_false") == 0
        and trial.get("warmup_gotit_false") == 0
        and trial.get("invariant_failures") == 0
        and trial.get("warmup_invariant_failures") == 0
        and trial.get("exception_failures") == 0
        and trial.get("successes") == trial.get("requested_measured_samples")
        and trial.get("attempted_measured_samples") == trial.get("requested_measured_samples")
        and trial.get("returned_measured_samples") == trial.get("requested_measured_samples")
    )


def _all_trials_clean(
    trials: Sequence[dict[str, Any]],
    worker_counts: Sequence[int],
    *,
    expected_in_order: bool,
) -> bool:
    if len(trials) != len(worker_counts):
        return False
    return all(
        int(trial.get("workers", -1)) == int(workers)
        and trial.get("in_order") is expected_in_order
        and _trial_is_clean(trial)
        for trial, workers in zip(trials, worker_counts)
    )


def _best_trial(trials: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        trial
        for trial in trials
        if _trial_is_clean(trial)
    ]
    if not eligible:
        return None
    return max(eligible, key=lambda trial: float(trial["successful_samples_per_second"]))


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if platform.system() != "Linux" or not Path("/proc/self/io").is_file():
        raise RuntimeError("PointOdyssey loader resource benchmarking requires Linux procfs")
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[1]

    preflight = preflight_dataset(dataset_root, args.expected_scenes)
    train_root = Path(preflight["train_root"])
    if output_dir == train_root or train_root in output_dir.parents:
        raise ValueError("--output-dir must be outside the prepared training tree")
    output_dir.mkdir(parents=True, exist_ok=False)
    measured_count = args.expected_scenes * args.measured_scene_repeats
    warmup_indices = build_balanced_schedule(
        args.expected_scenes,
        args.warmup_samples,
        repeat_offset=100,
        seed=args.schedule_seed,
    )
    measured_indices = build_balanced_schedule(
        args.expected_scenes,
        measured_count,
        repeat_offset=200,
        seed=args.schedule_seed + 1,
    )
    sweep_tail_count = max(args.worker_counts) * 4
    sweep_tail_indices = build_balanced_schedule(
        args.expected_scenes,
        sweep_tail_count,
        repeat_offset=300,
        seed=args.schedule_seed + 100,
    )
    run_config = {
        "format": FORMAT_NAME,
        "schema_version": SCHEMA_VERSION,
        "created_at_unix": time.time(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "git": _git_metadata(repo_root),
        "environment": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES",
            )
        },
        "settings": {
            "coverage_scenes": args.expected_scenes,
            "worker_counts": list(args.worker_counts),
            "batch_size": 1,
            "sequence_length": 24,
            "views": 4,
            "pin_memory": True,
            "prefetch_factor_when_workers_positive": 4,
            "persistent_workers": False,
            "worker_sweep_in_order": False,
            "worker_sweep_semantics": (
                "matches current training; actual returned provenance is authoritative "
                "because worker completion order may differ"
            ),
            "confirmation_in_order": False,
            "warmup_samples": args.warmup_samples,
            "measured_scene_repeats": args.measured_scene_repeats,
            "measured_samples_per_worker": measured_count,
            "worker_sweep_tail_samples": sweep_tail_count,
            "worker_sweep_tail_semantics": (
                "same fixed tail for every lane; keeps the largest configured worker x 4-prefetch queue "
                "saturated through the timed boundary"
            ),
            "confirmation_warmup": args.confirmation_warmup,
            "confirmation_samples": args.confirmation_samples,
            "schedule_seed": args.schedule_seed,
            "monitor_interval_seconds": args.monitor_interval,
            "child_timeout_seconds": args.child_timeout_seconds,
            "cache_state": "uncontrolled warm cache; system caches are not dropped",
            "augmentation_profile": "exact configs/train.yaml + configs/experiment/pointodyssey.yaml",
            "coverage_validation": "full tensor shapes/dtypes/finiteness/value/query/provenance checks",
            "timed_validation": "lightweight shapes/dtypes/query count/visibility dtype/provenance/pin checks",
        },
        "preflight": preflight,
    }
    _atomic_json(output_dir / "run_config.json", run_config)

    summary: dict[str, Any] = {
        "format": FORMAT_NAME,
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "run_config": run_config,
        "coverage": None,
        "trials": [],
        "confirmation": None,
    }
    _atomic_json(output_dir / "summary.json", summary)
    wandb_run = None
    try:
        wandb_run = _wandb_init(args, run_config)
        coverage = _run_child(
            _coverage_child,
            {
                "dataset_root": str(dataset_root),
                "events_path": str(output_dir / "coverage_samples.jsonl"),
                "expected_scenes": args.expected_scenes,
                "progress_every": args.progress_every,
                "scene_contracts": preflight["scene_contracts"],
                "schedule_seed": args.schedule_seed,
            },
            timeout_seconds=args.child_timeout_seconds,
        )
        summary["coverage"] = coverage
        _atomic_json(output_dir / "summary.json", summary)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "coverage/successes": coverage.get("successes", 0),
                    "coverage/gotit_false": coverage.get("gotit_false", 0),
                    "coverage/exceptions": coverage.get("exceptions", 0),
                    "coverage/invariant_failures": coverage.get("invariant_failures", 0),
                }
            )

        for workers in args.worker_counts:
            print(
                f"POINTODYSSEY_LOADER_TRIAL_START workers={workers} "
                f"warmup={len(warmup_indices)} measured={len(measured_indices)} "
                f"tail={len(sweep_tail_indices)} in_order=false",
                flush=True,
            )
            result = _run_child(
                _trial_child,
                {
                    "dataset_root": str(dataset_root),
                    "events_path": str(output_dir / f"workers_{workers}_samples.jsonl"),
                    "expected_scenes": args.expected_scenes,
                    "workers": workers,
                    "in_order": False,
                    "warmup_indices": warmup_indices,
                    "measured_indices": measured_indices,
                    "tail_indices": sweep_tail_indices,
                    "scene_bytes": preflight["scene_bytes"],
                    "scene_contracts": preflight["scene_contracts"],
                    "schedule_seed": args.schedule_seed,
                    "monitor_interval": args.monitor_interval,
                    "progress_every": args.progress_every,
                },
                timeout_seconds=args.child_timeout_seconds,
            )
            summary["trials"].append(result)
            _atomic_json(output_dir / "summary.json", summary)
            _wandb_log(wandb_run, f"workers_{workers}", result)

        best = _best_trial(summary["trials"])
        if best is not None and args.confirmation_samples > 0:
            confirmation_workers = int(best["workers"])
            confirmation_warmup = build_balanced_schedule(
                args.expected_scenes,
                args.confirmation_warmup,
                repeat_offset=1000,
                seed=args.schedule_seed + 2,
            )
            confirmation_measured = build_balanced_schedule(
                args.expected_scenes,
                args.confirmation_samples,
                repeat_offset=2000,
                seed=args.schedule_seed + 3,
            )
            confirmation_tail = build_balanced_schedule(
                args.expected_scenes,
                sweep_tail_count,
                repeat_offset=3000,
                seed=args.schedule_seed + 4,
            )
            print(
                "POINTODYSSEY_LOADER_CONFIRMATION_START "
                f"workers={confirmation_workers} warmup={len(confirmation_warmup)} "
                f"measured={len(confirmation_measured)} tail={len(confirmation_tail)} "
                "in_order=false",
                flush=True,
            )
            summary["confirmation"] = _run_child(
                _trial_child,
                {
                    "dataset_root": str(dataset_root),
                    "events_path": str(output_dir / "confirmation_samples.jsonl"),
                    "expected_scenes": args.expected_scenes,
                    "workers": confirmation_workers,
                    "in_order": False,
                    "warmup_indices": confirmation_warmup,
                    "measured_indices": confirmation_measured,
                    "tail_indices": confirmation_tail,
                    "scene_bytes": preflight["scene_bytes"],
                    "scene_contracts": preflight["scene_contracts"],
                    "schedule_seed": args.schedule_seed + 2,
                    "monitor_interval": args.monitor_interval,
                    "progress_every": args.progress_every,
                },
                timeout_seconds=args.child_timeout_seconds,
            )
            _atomic_json(output_dir / "summary.json", summary)
            _wandb_log(wandb_run, "confirmation", summary["confirmation"])

        tree_fingerprint_after = dataset_tree_fingerprint(train_root)
        summary["tree_fingerprint_after"] = tree_fingerprint_after
        summary["dataset_tree_unchanged"] = (
            tree_fingerprint_after == preflight["tree_fingerprint_before"]
        )
        coverage_clean = (
            coverage.get("status") == "completed"
            and coverage.get("successes") == args.expected_scenes
            and coverage.get("gotit_false") == 0
            and coverage.get("exceptions") == 0
            and coverage.get("invariant_failures") == 0
            and coverage.get("all_scenes_seen") is True
        )
        trials_clean = _all_trials_clean(
            summary["trials"],
            args.worker_counts,
            expected_in_order=False,
        )
        summary["best_worker_count"] = int(best["workers"]) if best is not None else None
        confirmation_clean = (
            summary["confirmation"] is None
            or (
                summary["confirmation"].get("status") == "completed"
                and summary["confirmation"].get("in_order") is False
                and summary["confirmation"].get("workers") == summary["best_worker_count"]
                and _trial_is_clean(summary["confirmation"])
            )
        )
        summary["status"] = (
            "completed"
            if coverage_clean and trials_clean and confirmation_clean and summary["dataset_tree_unchanged"]
            else "failed"
        )
        summary["finished_at_unix"] = time.time()
        _atomic_json(output_dir / "summary.json", summary)
        print(
            "POINTODYSSEY_LOADER_BENCHMARK_DONE "
            f"status={summary['status']} best_workers={summary['best_worker_count']} "
            f"output={output_dir}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "benchmark/completed": int(summary["status"] == "completed"),
                    "benchmark/best_worker_count": summary["best_worker_count"],
                    "benchmark/dataset_tree_unchanged": int(summary["dataset_tree_unchanged"]),
                }
            )
        return 0 if summary["status"] == "completed" else 1
    except BaseException as exc:
        summary["status"] = "error"
        summary["finished_at_unix"] = time.time()
        summary["error"] = repr(exc)
        summary["traceback"] = traceback.format_exc()
        _atomic_json(output_dir / "summary.json", summary)
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    raise SystemExit(main())
