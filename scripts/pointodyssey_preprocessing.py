#!/usr/bin/env python3
"""Convert the fixed PointOdyssey Track A collection for MV-Tracker.

This script deliberately contains the approved split rather than discovering or
inventing one at runtime.  It writes a Kubric-like split/scene/view directory
tree while preserving PointOdyssey's world tracks and optical-Z depth.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import platform
import shutil
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar

import cv2
import numpy as np

if __package__:
    from .pointodyssey_contract import (
        ASSIGNMENTS,
        POINT_COUNT,
        SOURCE_FPS,
        SOURCE_FRAME_COUNTS,
        SOURCE_HEIGHT,
        SOURCE_SUBROOTS,
        SOURCE_WIDTH,
        VIEW_IDS,
        SourceAssignment,
    )
else:
    from pointodyssey_contract import (
        ASSIGNMENTS,
        POINT_COUNT,
        SOURCE_FPS,
        SOURCE_FRAME_COUNTS,
        SOURCE_HEIGHT,
        SOURCE_SUBROOTS,
        SOURCE_WIDTH,
        VIEW_IDS,
        SourceAssignment,
    )


SCHEMA_VERSION = 4
OUTPUT_HEIGHT = 384
OUTPUT_WIDTH = 512
JPEG_QUALITY = 95
LONG_CHUNK_LENGTH = 120
PROJECTION_TOLERANCE_PX = 0.01
DEPTH_TRACK_TOLERANCE_METRES = 0.05
RIGID_DETERMINANT_TOLERANCE = 1e-3
SCALE_X = OUTPUT_WIDTH / SOURCE_WIDTH
SCALE_Y = OUTPUT_HEIGHT / SOURCE_HEIGHT
DEFAULT_WORKERS = 4
RGB_MAX_BATCH_FRAMES = 8
RGB_MAX_BATCH_BYTES = 8 * 1024 * 1024
JPEG_VALIDATION_BATCH_FILES = 32


@dataclass(frozen=True)
class SceneSpec:
    split: str
    scene_id: str
    layout: str
    source_sequence: str
    environment_family: str
    source_frame_start: int
    source_frame_end: int
    source_frame_count: int
    source_fps: int

    @property
    def frame_count(self) -> int:
        return self.source_frame_end - self.source_frame_start

    @property
    def source_key(self) -> tuple[str, str]:
        return self.layout, self.source_sequence


@dataclass(frozen=True)
class RGBFrameJob:
    layout: str
    sequence: str
    view: int
    source_frame: int
    split: str
    scene_id: str
    local_frame: int
    output_path: str
    encoded_jpeg: bytes
    source_height: int
    source_width: int
    output_height: int
    output_width: int
    jpeg_quality: int


@dataclass(frozen=True)
class RGBFrameResult:
    output_path: str
    encoded_jpeg: bytes
    source_frame: int
    local_frame: int
    source_decode_error: str | None


@dataclass(frozen=True)
class JPEGValidationJob:
    split: str
    scene_id: str
    view: int
    frame: int
    path: str
    expected_height: int
    expected_width: int


BatchInput = TypeVar("BatchInput")
BatchOutput = TypeVar("BatchOutput")


class SemanticValidationError(RuntimeError):
    """Raised after conversion when strict semantic validation has failed."""


class ValidationRecorder:
    """Collect semantic failures without changing or dropping source data."""

    def __init__(self, ignore_failures: bool):
        self.ignore_failures = bool(ignore_failures)
        self.failures: list[dict[str, Any]] = []
        self._by_scene: dict[tuple[str, str], list[dict[str, Any]]] = {}

    def record(
        self,
        spec: SceneSpec,
        check: str,
        observed: Any,
        expected: str,
        *,
        view: int | None = None,
        frame: int | None = None,
    ) -> None:
        failure = {
            "split": spec.split,
            "scene_id": spec.scene_id,
            "source_sequence": spec.source_sequence,
            "check": check,
            "observed": _json_value(observed),
            "expected": expected,
            "ignored": self.ignore_failures,
        }
        if view is not None:
            failure["view"] = int(view)
        if frame is not None:
            failure["frame"] = int(frame)
        self.failures.append(failure)
        self._by_scene.setdefault((spec.split, spec.scene_id), []).append(failure)

    def for_scene(self, spec: SceneSpec) -> list[dict[str, Any]]:
        return list(self._by_scene.get((spec.split, spec.scene_id), ()))

    def raise_if_strict(self) -> None:
        if self.failures and not self.ignore_failures:
            raise SemanticValidationError(
                f"{len(self.failures)} semantic validation failure(s); "
                "rerun with --ignore-validation-failures only if these are accepted"
            )


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "nan"
        return "infinity" if value > 0 else "-infinity"
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _initialize_process_worker() -> None:
    """Prevent each spawned OpenCV worker from creating its own thread pool."""
    cv2.setNumThreads(1)


def _black_placeholder_jpeg_bytes(height: int, width: int, quality: int) -> bytes:
    """Encode the deterministic black JPEG used only for invalid RGB frames."""
    placeholder = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(
        ".jpg",
        placeholder,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not ok:
        raise RuntimeError("failed to encode black JPEG placeholder")
    return encoded.tobytes()


def _transcode_rgb_batch(batch: tuple[RGBFrameJob, ...]) -> tuple[RGBFrameResult, ...]:
    """Decode, resize, and encode one ordered batch without writing files."""
    results: list[RGBFrameResult] = []
    for job in batch:
        try:
            encoded = np.frombuffer(job.encoded_jpeg, dtype=np.uint8)
            source_decode_error: str | None = None
            try:
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            except cv2.error as exc:
                decoded = None
                source_decode_error = f"cv2.imdecode failed: {exc}"
            if decoded is None:
                if source_decode_error is None:
                    source_decode_error = "cv2.imdecode returned no image"
                encoded_output = _black_placeholder_jpeg_bytes(
                    job.output_height,
                    job.output_width,
                    job.jpeg_quality,
                )
            else:
                expected_source_shape = (job.source_height, job.source_width, 3)
                if decoded.shape != expected_source_shape:
                    raise ValueError(
                        f"source JPEG decodes to {decoded.shape}, "
                        f"expected {expected_source_shape}"
                    )
                resized = cv2.resize(
                    decoded,
                    (job.output_width, job.output_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                ok, jpeg = cv2.imencode(
                    ".jpg",
                    resized,
                    [cv2.IMWRITE_JPEG_QUALITY, job.jpeg_quality],
                )
                if not ok:
                    raise RuntimeError("failed to encode resized JPEG")
                encoded_output = jpeg.tobytes()
            results.append(
                RGBFrameResult(
                    output_path=job.output_path,
                    encoded_jpeg=encoded_output,
                    source_frame=job.source_frame,
                    local_frame=job.local_frame,
                    source_decode_error=source_decode_error,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "RGB transcode failed "
                f"layout={job.layout} sequence={job.sequence} view={job.view} "
                f"source_frame={job.source_frame} split={job.split} "
                f"scene={job.scene_id} local_frame={job.local_frame}"
            ) from exc
    return tuple(results)


def _validate_jpeg_batch(batch: tuple[JPEGValidationJob, ...]) -> int:
    """Decode persisted JPEGs in deterministic batch order."""
    for job in batch:
        try:
            decoded = cv2.imread(job.path, cv2.IMREAD_COLOR)
            expected_shape = (job.expected_height, job.expected_width, 3)
            if decoded is None or decoded.shape != expected_shape:
                observed = None if decoded is None else decoded.shape
                raise ValueError(
                    f"prepared JPEG decodes to {observed}, expected {expected_shape}"
                )
        except Exception as exc:
            raise RuntimeError(
                "prepared JPEG validation failed "
                f"split={job.split} scene={job.scene_id} view={job.view} "
                f"frame={job.frame} path={job.path}"
            ) from exc
    return len(batch)


def _batch_context(batch: Sequence[Any]) -> str:
    first = batch[0]
    last = batch[-1]
    if isinstance(first, RGBFrameJob) and isinstance(last, RGBFrameJob):
        return (
            f"{first.layout}/{first.sequence}/view_{first.view} "
            f"frames={first.source_frame}-{last.source_frame}"
        )
    if isinstance(first, JPEGValidationJob) and isinstance(last, JPEGValidationJob):
        return (
            f"split={first.split} scene={first.scene_id} view={first.view} "
            f"frames={first.frame}-{last.frame}"
        )
    return f"items={len(batch)}"


def _ordered_bounded_batches(
    executor: ProcessPoolExecutor,
    worker: Callable[[BatchInput], BatchOutput],
    batches: Iterable[BatchInput],
    *,
    max_in_flight: int,
    stage: str,
) -> Iterator[BatchOutput]:
    """Submit a bounded number of batches and consume them in input order."""
    if max_in_flight <= 0:
        raise ValueError("max_in_flight must be positive")
    iterator = iter(batches)
    pending: deque[tuple[str, Future[BatchOutput]]] = deque()
    exhausted = False
    deferred_error: BaseException | None = None

    def cancel_pending() -> None:
        for _other_context, other in pending:
            other.cancel()

    def submit_one() -> None:
        nonlocal deferred_error, exhausted
        if exhausted or deferred_error is not None:
            return
        try:
            batch = next(iterator)
        except StopIteration:
            exhausted = True
            return
        except BaseException as exc:
            exhausted = True
            deferred_error = exc
            return
        try:
            context = _batch_context(batch)
            future = executor.submit(worker, batch)
        except BaseException as exc:
            exhausted = True
            deferred_error = exc
            return
        pending.append((context, future))

    for _ in range(max_in_flight):
        submit_one()
    while pending:
        context, future = pending.popleft()
        try:
            yield future.result()
        except BaseException as exc:
            cancel_pending()
            if isinstance(exc, Exception):
                raise RuntimeError(f"{stage} batch failed: {context}") from exc
            raise
        submit_one()
    if deferred_error is not None:
        raise deferred_error


def _iter_rgb_batches(jobs: Iterable[RGBFrameJob]) -> Iterator[tuple[RGBFrameJob, ...]]:
    batch: list[RGBFrameJob] = []
    batch_bytes = 0
    iterator = iter(jobs)
    while True:
        try:
            job = next(iterator)
        except StopIteration:
            break
        except BaseException:
            if batch:
                yield tuple(batch)
            raise
        encoded_bytes = len(job.encoded_jpeg)
        if batch and (
            len(batch) >= RGB_MAX_BATCH_FRAMES
            or batch_bytes + encoded_bytes > RGB_MAX_BATCH_BYTES
        ):
            yield tuple(batch)
            batch = []
            batch_bytes = 0
        batch.append(job)
        batch_bytes += encoded_bytes
    if batch:
        yield tuple(batch)


def _iter_scene_rgb_jobs(
    images: np.ndarray,
    spec: SceneSpec,
    view: int,
    output_view: Path,
) -> Iterator[RGBFrameJob]:
    for local_frame, source_frame in enumerate(
        range(spec.source_frame_start, spec.source_frame_end)
    ):
        encoded = images[source_frame]
        if not isinstance(encoded, np.ndarray) or encoded.ndim != 1 or encoded.dtype != np.uint8:
            raise ValueError(
                "invalid source JPEG object "
                f"layout={spec.layout} sequence={spec.source_sequence} view={view} "
                f"source_frame={source_frame}: expected one-dimensional uint8 array"
            )
        encoded_bytes = encoded.tobytes()
        images[source_frame] = None
        yield RGBFrameJob(
            layout=spec.layout,
            sequence=spec.source_sequence,
            view=view,
            source_frame=source_frame,
            split=spec.split,
            scene_id=spec.scene_id,
            local_frame=local_frame,
            output_path=str(output_view / f"rgba_{local_frame:05d}.jpg"),
            encoded_jpeg=encoded_bytes,
            source_height=SOURCE_HEIGHT,
            source_width=SOURCE_WIDTH,
            output_height=OUTPUT_HEIGHT,
            output_width=OUTPUT_WIDTH,
            jpeg_quality=JPEG_QUALITY,
        )


def _iter_validation_batches(
    jobs: Iterable[JPEGValidationJob],
) -> Iterator[tuple[JPEGValidationJob, ...]]:
    batch: list[JPEGValidationJob] = []
    for job in jobs:
        batch.append(job)
        if len(batch) >= JPEG_VALIDATION_BATCH_FILES:
            yield tuple(batch)
            batch = []
    if batch:
        yield tuple(batch)


def _run_batches(
    batches: Iterable[BatchInput],
    worker: Callable[[BatchInput], BatchOutput],
    *,
    process_pool: ProcessPoolExecutor | None,
    process_workers: int,
    stage: str,
) -> Iterator[BatchOutput]:
    if process_pool is None:
        for batch in batches:
            yield worker(batch)
        return
    yield from _ordered_bounded_batches(
        process_pool,
        worker,
        batches,
        max_in_flight=2 * process_workers,
        stage=stage,
    )


def chunk_ranges(frame_count: int, chunk_length: int = LONG_CHUNK_LENGTH) -> list[tuple[int, int]]:
    if frame_count <= 0 or chunk_length <= 0:
        raise ValueError("frame_count and chunk_length must be positive")
    return [(start, min(start + chunk_length, frame_count)) for start in range(0, frame_count, chunk_length)]


def build_scene_specs() -> list[SceneSpec]:
    specs: list[SceneSpec] = []
    for assignment in ASSIGNMENTS:
        source_frames = SOURCE_FRAME_COUNTS[assignment.layout]
        ranges = chunk_ranges(source_frames) if assignment.layout == "long" else [(0, source_frames)]
        for offset, (start, end) in enumerate(ranges):
            specs.append(
                SceneSpec(
                    split=assignment.split,
                    scene_id=f"{assignment.first_scene_id + offset:06d}",
                    layout=assignment.layout,
                    source_sequence=assignment.sequence,
                    environment_family=assignment.environment_family,
                    source_frame_start=start,
                    source_frame_end=end,
                    source_frame_count=source_frames,
                    source_fps=SOURCE_FPS[assignment.layout],
                )
            )
    validate_split_plan(specs)
    return specs


def validate_split_plan(specs: Sequence[SceneSpec]) -> None:
    expected = {
        "train": {"scenes": 78, "frames": 8846},
        "validation": {"scenes": 3, "frames": 301},
        "test": {"scenes": 3, "frames": 301},
    }
    for split, counts in expected.items():
        split_specs = [spec for spec in specs if spec.split == split]
        actual_ids = [spec.scene_id for spec in split_specs]
        expected_ids = [f"{index:06d}" for index in range(counts["scenes"])]
        if actual_ids != expected_ids:
            raise ValueError(f"{split} scene IDs are not exactly contiguous: {actual_ids}")
        if sum(spec.frame_count for spec in split_specs) != counts["frames"]:
            raise ValueError(f"{split} frame total does not match the approved split")

    families_by_split = {
        split: {spec.environment_family for spec in specs if spec.split == split}
        for split in expected
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = families_by_split[left] & families_by_split[right]
        if overlap:
            raise ValueError(f"environment leakage between {left} and {right}: {sorted(overlap)}")

    for assignment in (item for item in ASSIGNMENTS if item.layout == "long"):
        source_specs = [
            spec
            for spec in specs
            if spec.layout == assignment.layout and spec.source_sequence == assignment.sequence
        ]
        ranges = [(spec.source_frame_start, spec.source_frame_end) for spec in source_specs]
        if ranges != chunk_ranges(SOURCE_FRAME_COUNTS["long"]):
            raise ValueError(f"invalid long-form partition for {assignment.sequence}: {ranges}")


def scale_intrinsics(intrinsics_4: np.ndarray) -> np.ndarray:
    intrinsics_4 = np.asarray(intrinsics_4)
    if intrinsics_4.shape != (4,):
        raise ValueError(f"intrinsics must have shape (4,), got {intrinsics_4.shape}")
    fx, fy, cx, cy = intrinsics_4.astype(np.float32, copy=False)
    return np.asarray(
        [[fx * SCALE_X, 0.0, cx * SCALE_X], [0.0, fy * SCALE_Y, cy * SCALE_Y], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _source_scene_dir(source_root: Path, spec: SceneSpec) -> Path:
    return source_root / SOURCE_SUBROOTS[spec.layout] / spec.source_sequence


def _output_scene_dir(output_root: Path, spec: SceneSpec) -> Path:
    return output_root / spec.split / spec.scene_id


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _require_dir(path: Path) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(path)
    return path


def _load_npy(
    path: Path,
    *,
    mmap_mode: str | None = "r",
    allow_pickle: bool = False,
) -> np.ndarray:
    _require_file(path)
    return np.load(path, mmap_mode=mmap_mode, allow_pickle=allow_pickle)


def _npy_header(path: Path) -> tuple[tuple[int, ...], bool, np.dtype[Any]]:
    _require_file(path)
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(handle)
        if version in ((2, 0), (3, 0)):
            return np.lib.format.read_array_header_2_0(handle)
    raise ValueError(f"unsupported NPY version in {path}: {version}")


def _require_array(
    path: Path,
    expected_shape: tuple[int, ...],
    expected_dtype: np.dtype[Any] | type[Any],
    *,
    mmap_mode: str | None = "r",
    allow_pickle: bool = False,
) -> np.ndarray:
    array = _load_npy(path, mmap_mode=mmap_mode, allow_pickle=allow_pickle)
    expected = np.dtype(expected_dtype)
    if array.shape != expected_shape or array.dtype != expected:
        raise ValueError(
            f"{path} must have shape {expected_shape} and dtype {expected}, "
            f"got shape {array.shape} and dtype {array.dtype}"
        )
    return array


def _open_depth(source_view: Path, layout: str, frame_count: int) -> Any:
    expected_shape = (frame_count, SOURCE_HEIGHT, SOURCE_WIDTH)
    if layout != "long":
        return _require_array(source_view / "depth.npy", expected_shape, np.float32)
    depth_path = source_view / "depth.zarr"
    _require_dir(depth_path)
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("zarr is required to read fixed long-form depth.zarr stores") from exc
    depth = zarr.open(str(depth_path), mode="r")
    if tuple(depth.shape) != expected_shape or np.dtype(depth.dtype) != np.dtype(np.float32):
        raise ValueError(
            f"{depth_path} must be a root array with shape {expected_shape} and dtype float32; "
            f"got shape {getattr(depth, 'shape', None)} and dtype {getattr(depth, 'dtype', None)}"
        )
    return depth


def _preflight_source(source_root: Path, source_specs: Sequence[SceneSpec]) -> int:
    first = source_specs[0]
    source_dir = _require_dir(_source_scene_dir(source_root, first))
    frames = first.source_frame_count
    _require_array(source_dir / "tracks_xyz.npy", (frames, POINT_COUNT, 3), np.float32)
    _require_array(source_dir / "queries_xytv.npy", (POINT_COUNT, 4), np.float32)
    rgb_bytes = 0
    for view in VIEW_IDS:
        source_view = _require_dir(source_dir / str(view))
        image_path = source_view / "images_jpeg_bytes.npy"
        shape, _fortran_order, dtype = _npy_header(image_path)
        if shape != (frames,) or dtype != np.dtype(object):
            raise ValueError(f"{image_path} must have shape {(frames,)} and dtype object, got {shape}, {dtype}")
        rgb_bytes += image_path.stat().st_size
        _require_array(source_view / "intrinsics.npy", (4,), np.float32)
        _require_array(source_view / "extrinsics_w2c.npy", (frames, 4, 4), np.float32)
        _require_array(source_view / "visibility.npy", (frames, POINT_COUNT), np.bool_)
        _open_depth(source_view, first.layout, frames)
    return rgb_bytes


def preflight(source_root: Path, output_root: Path, specs: Sequence[SceneSpec]) -> None:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output root: {output_root}")
    source_groups = _group_by_source(specs)
    source_rgb_bytes = sum(_preflight_source(source_root, group) for group in source_groups)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_depth_bytes = sum(
        spec.frame_count * len(VIEW_IDS) * OUTPUT_HEIGHT * OUTPUT_WIDTH * np.dtype(np.float32).itemsize
        for spec in specs
    )
    # Original 1080p JPEG arrays are a conservative upper bound for resized JPEG output.
    required_bytes = output_depth_bytes + source_rgb_bytes + 2 * 1024**3
    free_bytes = shutil.disk_usage(output_root.parent).free
    if free_bytes < required_bytes:
        raise OSError(
            f"insufficient free space under {output_root.parent}: need at least "
            f"{required_bytes / 1024**3:.1f} GiB, have {free_bytes / 1024**3:.1f} GiB"
        )


def _group_by_source(specs: Sequence[SceneSpec]) -> list[list[SceneSpec]]:
    groups: dict[tuple[str, str], list[SceneSpec]] = {}
    order: list[tuple[str, str]] = []
    for spec in specs:
        if spec.source_key not in groups:
            groups[spec.source_key] = []
            order.append(spec.source_key)
        groups[spec.source_key].append(spec)
    return [groups[key] for key in order]


def _project_points(
    tracks: np.ndarray,
    extrinsics_3x4: np.ndarray,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ones = np.ones((*tracks.shape[:-1], 1), dtype=np.float32)
    homogeneous = np.concatenate((tracks.astype(np.float32, copy=False), ones), axis=-1)
    camera = np.einsum("fij,fpj->fpi", extrinsics_3x4, homogeneous, optimize=True)
    z = camera[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        x = intrinsics[0, 0] * camera[..., 0] / z + intrinsics[0, 2]
        y = intrinsics[1, 1] * camera[..., 1] / z + intrinsics[1, 2]
    return np.stack((x, y), axis=-1).astype(np.float32, copy=False), z


def _depth_track_consistency_masks(
    depth_frame: np.ndarray,
    projected_xy: np.ndarray,
    camera_z: np.ndarray,
    candidate_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return within-tolerance and any-valid-depth masks for candidate tracks."""
    depth_frame = np.asarray(depth_frame)
    projected_xy = np.asarray(projected_xy)
    camera_z = np.asarray(camera_z)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if depth_frame.ndim != 2 or depth_frame.shape != (OUTPUT_HEIGHT, OUTPUT_WIDTH):
        raise ValueError(
            f"depth frame must be 2-D with shape {(OUTPUT_HEIGHT, OUTPUT_WIDTH)}, "
            f"got ndim={depth_frame.ndim} and shape={depth_frame.shape}"
        )
    if depth_frame.dtype != np.dtype(np.float32):
        raise ValueError(f"depth frame must have dtype float32, got {depth_frame.dtype}")
    if projected_xy.shape != (*candidate_mask.shape, 2):
        raise ValueError(
            "projected coordinates must have candidate-mask shape plus a final xy axis; "
            f"got {projected_xy.shape} and {candidate_mask.shape}"
        )
    if camera_z.shape != candidate_mask.shape:
        raise ValueError(
            f"camera-Z shape must match candidate mask; got {camera_z.shape} and "
            f"{candidate_mask.shape}"
        )

    consistent_mask = np.zeros(candidate_mask.shape, dtype=bool)
    any_valid_depth_mask = np.zeros(candidate_mask.shape, dtype=bool)
    internally_valid_candidates = (
        candidate_mask
        & np.isfinite(projected_xy).all(axis=-1)
        & np.isfinite(camera_z)
        & (camera_z > 0.0)
        & (projected_xy[..., 0] >= -0.5)
        & (projected_xy[..., 0] < OUTPUT_WIDTH - 0.5)
        & (projected_xy[..., 1] >= -0.5)
        & (projected_xy[..., 1] < OUTPUT_HEIGHT - 0.5)
    )
    candidate_ids = np.flatnonzero(internally_valid_candidates.reshape(-1))
    if not candidate_ids.size:
        return consistent_mask, any_valid_depth_mask

    flat_xy = projected_xy.reshape(-1, 2)
    flat_camera_z = camera_z.reshape(-1)
    selected_xy = flat_xy[candidate_ids]
    selected_camera_z = flat_camera_z[candidate_ids]
    center_x = np.floor(selected_xy[:, 0] + 0.5).astype(np.int64)
    center_y = np.floor(selected_xy[:, 1] + 0.5).astype(np.int64)
    offsets_y = np.asarray((-1, -1, -1, 0, 0, 0, 1, 1, 1), dtype=np.int64)
    offsets_x = np.asarray((-1, 0, 1, -1, 0, 1, -1, 0, 1), dtype=np.int64)
    sample_x = center_x[:, None] + offsets_x[None, :]
    sample_y = center_y[:, None] + offsets_y[None, :]
    inside = (
        (sample_x >= 0)
        & (sample_x < OUTPUT_WIDTH)
        & (sample_y >= 0)
        & (sample_y < OUTPUT_HEIGHT)
    )
    safe_x = np.clip(sample_x, 0, OUTPUT_WIDTH - 1)
    safe_y = np.clip(sample_y, 0, OUTPUT_HEIGHT - 1)
    sampled_depth = depth_frame[safe_y, safe_x]
    valid_depth = inside & np.isfinite(sampled_depth) & (sampled_depth > 0.0)
    any_valid = valid_depth.any(axis=1)
    accepted = (
        valid_depth
        & (
            np.abs(sampled_depth - selected_camera_z[:, None])
            <= DEPTH_TRACK_TOLERANCE_METRES
        )
    ).any(axis=1)
    consistent_mask.reshape(-1)[candidate_ids] = accepted
    any_valid_depth_mask.reshape(-1)[candidate_ids] = any_valid
    return consistent_mask, any_valid_depth_mask


def _validate_source_queries(
    source_specs: Sequence[SceneSpec],
    tracks: np.ndarray,
    queries: np.ndarray,
    intrinsics_4: Sequence[np.ndarray],
    extrinsics: Sequence[np.ndarray],
    visibility: Sequence[np.ndarray],
    recorder: ValidationRecorder,
) -> None:
    """Validate all source queries in one source-wide pass, then map failures to chunks."""
    frame_values = np.asarray(queries[:, 2])
    view_values = np.asarray(queries[:, 3])
    valid_frame_values = (
        np.isfinite(frame_values)
        & (frame_values >= 0)
        & (frame_values < source_specs[0].source_frame_count)
        & (frame_values == np.floor(frame_values))
    )
    valid_view_values = (
        np.isfinite(view_values)
        & (view_values >= 0)
        & (view_values < len(VIEW_IDS))
        & (view_values == np.floor(view_values))
    )
    frames = np.zeros(frame_values.shape, dtype=np.int64)
    views = np.zeros(view_values.shape, dtype=np.int64)
    frames[valid_frame_values] = frame_values[valid_frame_values].astype(np.int64)
    views[valid_view_values] = view_values[valid_view_values].astype(np.int64)
    valid_indices = (
        np.isfinite(queries).all(axis=1)
        & valid_frame_values
        & valid_view_values
    )
    if not valid_indices.all():
        first_bad = int(np.flatnonzero(~valid_indices)[0])
        recorder.record(
            source_specs[0],
            "query_indices",
            {"invalid_count": int((~valid_indices).sum()), "first_query": queries[first_bad].tolist()},
            "finite query with integer frame/view inside the source range",
        )

    point_ids = np.arange(POINT_COUNT)
    for view in VIEW_IDS:
        selected = valid_indices & (views == view)
        if not selected.any():
            continue
        ids = point_ids[selected]
        selected_frames = frames[selected]
        world = np.asarray(tracks[selected_frames, ids], dtype=np.float64)
        world_h = np.concatenate((world, np.ones((len(world), 1), dtype=np.float64)), axis=1)
        camera = np.einsum(
            "nij,nj->ni",
            np.asarray(extrinsics[view][selected_frames, :3, :4], dtype=np.float64),
            world_h,
            optimize=True,
        )
        fx, fy, cx, cy = np.asarray(intrinsics_4[view], dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            projected = np.stack(
                (fx * camera[:, 0] / camera[:, 2] + cx, fy * camera[:, 1] / camera[:, 2] + cy),
                axis=1,
            )
        expected_xy = np.asarray(queries[selected, :2], dtype=np.float64)
        errors = np.linalg.norm(projected - expected_xy, axis=1)
        source_visible = np.asarray(visibility[view][selected_frames, ids], dtype=bool)
        source_bounds = (
            (expected_xy[:, 0] >= -0.5)
            & (expected_xy[:, 0] < SOURCE_WIDTH - 0.5)
            & (expected_xy[:, 1] >= -0.5)
            & (expected_xy[:, 1] < SOURCE_HEIGHT - 0.5)
        )

        scaled_expected = expected_xy * np.asarray([SCALE_X, SCALE_Y], dtype=np.float64)
        resized_k = np.asarray(scale_intrinsics(intrinsics_4[view]), dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            resized_projected = np.stack(
                (
                    resized_k[0, 0] * camera[:, 0] / camera[:, 2] + resized_k[0, 2],
                    resized_k[1, 1] * camera[:, 1] / camera[:, 2] + resized_k[1, 2],
                ),
                axis=1,
            )
        resized_errors = np.linalg.norm(resized_projected - scaled_expected, axis=1)

        selected_global_ids = ids
        for spec in source_specs:
            in_scene = (selected_frames >= spec.source_frame_start) & (selected_frames < spec.source_frame_end)
            if not in_scene.any():
                continue
            _record_query_failure(
                spec,
                view,
                selected_frames,
                selected_global_ids,
                errors,
                ~np.isfinite(errors) | (errors > PROJECTION_TOLERANCE_PX),
                "query_reprojection_source",
                f"Euclidean projection error <= {PROJECTION_TOLERANCE_PX} pixels",
                recorder,
                in_scene,
            )
            _record_query_failure(
                spec,
                view,
                selected_frames,
                selected_global_ids,
                resized_errors,
                ~np.isfinite(resized_errors) | (resized_errors > PROJECTION_TOLERANCE_PX),
                "query_reprojection_resized",
                f"scaled-query Euclidean projection error <= {PROJECTION_TOLERANCE_PX} pixels",
                recorder,
                in_scene,
            )
            invalid_visible = ~source_visible | ~source_bounds
            _record_query_failure(
                spec,
                view,
                selected_frames,
                selected_global_ids,
                invalid_visible.astype(np.int64),
                invalid_visible,
                "query_visibility",
                "query is source-visible and inside source pixel bounds",
                recorder,
                in_scene,
            )


def _record_query_failure(
    spec: SceneSpec,
    view: int,
    frames: np.ndarray,
    point_ids: np.ndarray,
    values: np.ndarray,
    failed: np.ndarray,
    check: str,
    expected: str,
    recorder: ValidationRecorder,
    scene_mask: np.ndarray,
) -> None:
    mask = failed & scene_mask
    if not mask.any():
        return
    positions = np.flatnonzero(mask)
    if np.issubdtype(values.dtype, np.floating):
        sortable = np.where(np.isfinite(values[positions]), values[positions], np.inf)
        position = positions[int(np.argmax(sortable))]
        value: Any = float(values[position])
    else:
        position = positions[0]
        value = int(values[position])
    recorder.record(
        spec,
        check,
        {"failure_count": int(mask.sum()), "point": int(point_ids[position]), "value": value},
        expected,
        view=view,
        frame=int(frames[position]),
    )


def _record_finite_failure(
    recorder: ValidationRecorder,
    spec: SceneSpec,
    check: str,
    array: np.ndarray,
    expected: str,
    *,
    view: int | None = None,
    frame_axis: bool = True,
    finite_mask: np.ndarray | None = None,
) -> None:
    finite = np.isfinite(array) if finite_mask is None else np.asarray(finite_mask, dtype=bool)
    if finite.shape != array.shape:
        raise ValueError(
            f"finite mask shape must match array shape; got {finite.shape} and {array.shape}"
        )
    if finite.all():
        return
    first = np.argwhere(~finite)[0]
    frame = spec.source_frame_start + int(first[0]) if frame_axis and array.ndim > 0 else None
    recorder.record(
        spec,
        check,
        {"nonfinite_count": int((~finite).sum())},
        expected,
        view=view,
        frame=frame,
    )


def _prepare_scene_roots(build_root: Path, specs: Sequence[SceneSpec]) -> None:
    for split in ("train", "validation", "test"):
        (build_root / split).mkdir(parents=True, exist_ok=False)
    for spec in specs:
        scene_dir = _output_scene_dir(build_root, spec)
        scene_dir.mkdir(parents=False, exist_ok=False)
        for view in VIEW_IDS:
            (scene_dir / f"view_{view}").mkdir(parents=False, exist_ok=False)


def _convert_source_group(
    source_root: Path,
    build_root: Path,
    source_specs: Sequence[SceneSpec],
    recorder: ValidationRecorder,
    scene_stats: dict[tuple[str, str], dict[str, Any]],
    process_pool: ProcessPoolExecutor | None = None,
    process_workers: int = 1,
) -> None:
    if process_workers <= 0:
        raise ValueError("process_workers must be positive")
    first = source_specs[0]
    source_dir = _source_scene_dir(source_root, first)
    tracks = _require_array(
        source_dir / "tracks_xyz.npy",
        (first.source_frame_count, POINT_COUNT, 3),
        np.float32,
    )
    queries = _require_array(source_dir / "queries_xytv.npy", (POINT_COUNT, 4), np.float32)
    intrinsics_4: list[np.ndarray] = []
    extrinsics: list[np.ndarray] = []
    visibilities: list[np.ndarray] = []
    for view in VIEW_IDS:
        source_view = source_dir / str(view)
        intrinsics_4.append(np.asarray(_require_array(source_view / "intrinsics.npy", (4,), np.float32)))
        extrinsics.append(
            _require_array(
                source_view / "extrinsics_w2c.npy",
                (first.source_frame_count, 4, 4),
                np.float32,
            )
        )
        visibilities.append(
            _require_array(
                source_view / "visibility.npy",
                (first.source_frame_count, POINT_COUNT),
                np.bool_,
            )
        )

    _validate_source_queries(source_specs, tracks, queries, intrinsics_4, extrinsics, visibilities, recorder)

    track_finite_masks: dict[tuple[str, str], np.ndarray] = {}
    for spec in source_specs:
        track_start = time.perf_counter()
        output_scene = _output_scene_dir(build_root, spec)
        tracks_chunk = np.asarray(tracks[spec.source_frame_start : spec.source_frame_end], dtype=np.float32)
        np.save(output_scene / "tracks_3d.npy", np.ascontiguousarray(tracks_chunk))
        finite_coordinates = np.isfinite(tracks_chunk)
        _record_finite_failure(
            recorder,
            spec,
            "tracks_finite",
            tracks_chunk,
            "all world-track coordinates are finite",
            finite_mask=finite_coordinates,
        )
        finite_track_samples = finite_coordinates.all(axis=-1)
        track_finite_masks[(spec.split, spec.scene_id)] = finite_track_samples
        coordinate_count = int(tracks_chunk.size)
        finite_coordinate_count = int(finite_coordinates.sum())
        track_seconds = time.perf_counter() - track_start
        scene_stats[(spec.split, spec.scene_id)] = {
            "tracks": {
                "track_id_count": POINT_COUNT,
                "frame_count": spec.frame_count,
                "track_sample_count": spec.frame_count * POINT_COUNT,
                "coordinate_value_count": coordinate_count,
                "finite_coordinate_value_count": finite_coordinate_count,
                "nonfinite_coordinate_value_count": coordinate_count - finite_coordinate_count,
            },
            "exclusive_stage_seconds": {
                "tracks_validate_and_write": track_seconds,
                "projection_and_geometric_visibility": 0.0,
                "depth_resize_write_and_consistency": 0.0,
                "rgb_decode_resize_encode_write": 0.0,
            },
            "views": {},
        }

    for view in VIEW_IDS:
        source_view = source_dir / str(view)
        images = np.load(source_view / "images_jpeg_bytes.npy", allow_pickle=True)
        if images.shape != (first.source_frame_count,) or images.dtype != np.dtype(object):
            raise ValueError(f"invalid object JPEG array: {source_view / 'images_jpeg_bytes.npy'}")

        source_depth = _open_depth(source_view, first.layout, first.source_frame_count)
        resized_k = scale_intrinsics(intrinsics_4[view])

        for spec in source_specs:
            start, end = spec.source_frame_start, spec.source_frame_end
            output_view = _output_scene_dir(build_root, spec) / f"view_{view}"
            extrinsics_chunk_4x4 = np.asarray(extrinsics[view][start:end], dtype=np.float32)
            extrinsics_chunk = np.ascontiguousarray(extrinsics_chunk_4x4[:, :3, :4])
            source_visibility = np.asarray(visibilities[view][start:end], dtype=bool)
            tracks_chunk = np.asarray(tracks[start:end], dtype=np.float32)

            _record_finite_failure(
                recorder,
                spec,
                "intrinsics_finite",
                resized_k,
                "all resized intrinsic values are finite",
                view=view,
                frame_axis=False,
            )
            _record_finite_failure(
                recorder,
                spec,
                "extrinsics_finite",
                extrinsics_chunk,
                "all W2C extrinsic values are finite",
                view=view,
            )
            bottom_rows = extrinsics_chunk_4x4[:, 3, :]
            expected_bottom = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            valid_bottom = np.all(bottom_rows == expected_bottom, axis=1)
            if not valid_bottom.all():
                bad = int(np.flatnonzero(~valid_bottom)[0])
                recorder.record(
                    spec,
                    "extrinsics_homogeneous_row",
                    bottom_rows[bad].tolist(),
                    "homogeneous row is exactly [0,0,0,1] in FP32",
                    view=view,
                    frame=start + bad,
                )

            determinants = np.linalg.det(extrinsics_chunk_4x4[:, :3, :3])
            determinant_errors = np.abs(determinants - 1.0)
            invalid_determinants = ~np.isfinite(determinant_errors) | (
                determinant_errors > RIGID_DETERMINANT_TOLERANCE
            )
            if invalid_determinants.any():
                bad = int(np.flatnonzero(invalid_determinants)[0])
                recorder.record(
                    spec,
                    "extrinsics_rotation_determinant",
                    float(determinants[bad]),
                    (
                        "finite rigid rotation determinant with "
                        f"abs(det(R)-1) <= {RIGID_DETERMINANT_TOLERANCE}"
                    ),
                    view=view,
                    frame=start + bad,
                )

            projection_start = time.perf_counter()
            projected_xy, camera_z = _project_points(tracks_chunk, extrinsics_chunk, resized_k)
            finite_tracks = track_finite_masks[(spec.split, spec.scene_id)]
            finite_projection = np.isfinite(projected_xy).all(axis=-1) & np.isfinite(camera_z)
            inside = (
                (projected_xy[..., 0] >= -0.5)
                & (projected_xy[..., 0] < OUTPUT_WIDTH - 0.5)
                & (projected_xy[..., 1] >= -0.5)
                & (projected_xy[..., 1] < OUTPUT_HEIGHT - 0.5)
            )
            geometric_visibility = (
                source_visibility
                & finite_tracks
                & finite_projection
                & (camera_z > 0.0)
                & inside
            )
            output_visibility = geometric_visibility.copy()
            projection_seconds = time.perf_counter() - projection_start

            np.save(output_view / "intrinsics.npy", np.ascontiguousarray(resized_k))
            np.save(output_view / "extrinsics_w2c.npy", extrinsics_chunk)

            depth_start = time.perf_counter()
            depth_output = np.lib.format.open_memmap(
                output_view / "depth.npy",
                mode="w+",
                dtype=np.float32,
                shape=(spec.frame_count, OUTPUT_HEIGHT, OUTPUT_WIDTH),
            )
            first_nonfinite_frame: int | None = None
            first_negative_frame: int | None = None
            source_nonfinite_count = 0
            source_negative_count = 0
            prepared_depth_positive_count = 0
            prepared_depth_zero_count = 0
            prepared_depth_negative_count = 0
            prepared_depth_nonfinite_count = 0
            prepared_depth_finite_count = 0
            prepared_depth_finite_min: float | None = None
            prepared_depth_finite_max: float | None = None
            rejected_no_valid_depth = 0
            rejected_residual_over_tolerance = 0
            for local_frame, source_frame in enumerate(range(start, end)):
                source_depth_frame = np.asarray(source_depth[source_frame], dtype=np.float32)
                nonfinite = ~np.isfinite(source_depth_frame)
                negative = source_depth_frame < 0.0
                if nonfinite.any() and first_nonfinite_frame is None:
                    first_nonfinite_frame = source_frame
                if negative.any() and first_negative_frame is None:
                    first_negative_frame = source_frame
                source_nonfinite_count += int(nonfinite.sum())
                source_negative_count += int(negative.sum())
                resized_depth_frame = cv2.resize(
                    source_depth_frame,
                    (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                    interpolation=cv2.INTER_NEAREST,
                )
                depth_output[local_frame] = resized_depth_frame

                prepared_finite = np.isfinite(resized_depth_frame)
                prepared_positive = prepared_finite & (resized_depth_frame > 0.0)
                prepared_negative = prepared_finite & (resized_depth_frame < 0.0)
                prepared_zero = prepared_finite & (resized_depth_frame == 0.0)
                frame_finite_count = int(prepared_finite.sum())
                prepared_depth_finite_count += frame_finite_count
                prepared_depth_positive_count += int(prepared_positive.sum())
                prepared_depth_zero_count += int(prepared_zero.sum())
                prepared_depth_negative_count += int(prepared_negative.sum())
                prepared_depth_nonfinite_count += int((~prepared_finite).sum())
                if frame_finite_count:
                    frame_values = resized_depth_frame[prepared_finite]
                    frame_min = float(frame_values.min())
                    frame_max = float(frame_values.max())
                    prepared_depth_finite_min = (
                        frame_min
                        if prepared_depth_finite_min is None
                        else min(prepared_depth_finite_min, frame_min)
                    )
                    prepared_depth_finite_max = (
                        frame_max
                        if prepared_depth_finite_max is None
                        else max(prepared_depth_finite_max, frame_max)
                    )

                depth_consistent, any_valid_depth = _depth_track_consistency_masks(
                    resized_depth_frame,
                    projected_xy[local_frame],
                    camera_z[local_frame],
                    geometric_visibility[local_frame],
                )
                output_visibility[local_frame] &= depth_consistent
                frame_candidates = geometric_visibility[local_frame]
                rejected_no_valid_depth += int((frame_candidates & ~any_valid_depth).sum())
                rejected_residual_over_tolerance += int(
                    (frame_candidates & any_valid_depth & ~depth_consistent).sum()
                )
            depth_output.flush()
            del depth_output
            np.save(output_view / "visibility.npy", np.ascontiguousarray(output_visibility))
            depth_seconds = time.perf_counter() - depth_start

            before = int(source_visibility.sum())
            after_geometry = int(geometric_visibility.sum())
            after_depth_consistency = int(output_visibility.sum())
            if (
                after_depth_consistency
                + rejected_no_valid_depth
                + rejected_residual_over_tolerance
                != after_geometry
            ):
                raise AssertionError("depth-consistency visibility counts do not reconcile")
            prepared_depth_pixel_count = spec.frame_count * OUTPUT_HEIGHT * OUTPUT_WIDTH
            if (
                prepared_depth_positive_count
                + prepared_depth_zero_count
                + prepared_depth_negative_count
                + prepared_depth_nonfinite_count
                != prepared_depth_pixel_count
            ):
                raise AssertionError("prepared-depth category counts do not reconcile")

            view_stats = {
                "source_chunk_frame_count": spec.frame_count,
                "output_frame_count": spec.frame_count,
                "prepared_depth": {
                    "value_count": prepared_depth_pixel_count,
                    "finite_count": prepared_depth_finite_count,
                    "positive_count": prepared_depth_positive_count,
                    "zero_count": prepared_depth_zero_count,
                    "negative_count": prepared_depth_negative_count,
                    "nonfinite_count": prepared_depth_nonfinite_count,
                    "finite_min": prepared_depth_finite_min,
                    "finite_max": prepared_depth_finite_max,
                },
                "source_depth_diagnostics": {
                    "negative_count": source_negative_count,
                    "nonfinite_count": source_nonfinite_count,
                },
                "visibility_true_before_gating": before,
                "visibility_true_after_geometric_gating": after_geometry,
                "visibility_removed_by_geometric_gating": before - after_geometry,
                "depth_consistency_candidate_count": after_geometry,
                "depth_consistency_tolerance_metres": DEPTH_TRACK_TOLERANCE_METRES,
                "visibility_rejected_no_valid_depth": rejected_no_valid_depth,
                "visibility_rejected_residual_over_tolerance": (
                    rejected_residual_over_tolerance
                ),
                "visibility_true_after_depth_consistency_gating": after_depth_consistency,
                "visibility_removed_by_depth_consistency_gating": (
                    after_geometry - after_depth_consistency
                ),
                "visibility_true_after_gating": after_depth_consistency,
                "visibility_removed_by_gating": before - after_depth_consistency,
                "rgb": {
                    "source_frame_count": spec.frame_count,
                    "output_frame_count": 0,
                    "written_file_count": 0,
                    "output_bytes": 0,
                    "source_decode_failure_count": 0,
                    "placeholder_file_count": 0,
                    "invalid_frame_indices": [],
                    "source_decode_failures": [],
                },
                "planned_view_regular_file_count": 4 + spec.frame_count,
                "written_view_regular_file_count": 4,
                "exclusive_stage_seconds": {
                    "projection_and_geometric_visibility": projection_seconds,
                    "depth_resize_write_and_consistency": depth_seconds,
                    "rgb_decode_resize_encode_write": 0.0,
                },
            }
            scene_stats[(spec.split, spec.scene_id)]["views"][str(view)] = view_stats
            scene_stats[(spec.split, spec.scene_id)]["exclusive_stage_seconds"][
                "depth_resize_write_and_consistency"
            ] += depth_seconds
            scene_stats[(spec.split, spec.scene_id)]["exclusive_stage_seconds"][
                "projection_and_geometric_visibility"
            ] += projection_seconds

            if source_nonfinite_count:
                recorder.record(
                    spec,
                    "depth_finite",
                    {"nonfinite_count": source_nonfinite_count},
                    "all source optical-Z values are finite",
                    view=view,
                    frame=first_nonfinite_frame,
                )
            if source_negative_count:
                recorder.record(
                    spec,
                    "depth_nonnegative",
                    {"negative_count": source_negative_count},
                    "source optical-Z is non-negative; zero denotes invalid depth",
                    view=view,
                    frame=first_negative_frame,
                )

            rgb_start = time.perf_counter()
            rgb_jobs = _iter_scene_rgb_jobs(images, spec, view, output_view)
            rgb_batches = _iter_rgb_batches(rgb_jobs)
            for results in _run_batches(
                rgb_batches,
                _transcode_rgb_batch,
                process_pool=process_pool,
                process_workers=process_workers,
                stage="RGB transcode",
            ):
                for result in results:
                    written_bytes = Path(result.output_path).write_bytes(result.encoded_jpeg)
                    if written_bytes != len(result.encoded_jpeg):
                        raise OSError(
                            f"short JPEG write for {result.output_path}: "
                            f"expected {len(result.encoded_jpeg)}, wrote {written_bytes}"
                        )
                    view_stats["rgb"]["output_frame_count"] += 1
                    view_stats["rgb"]["written_file_count"] += 1
                    view_stats["rgb"]["output_bytes"] += written_bytes
                    if result.source_decode_error is not None:
                        if not 0 <= result.local_frame < spec.frame_count:
                            raise AssertionError("invalid placeholder local-frame index")
                        expected_source_frame = spec.source_frame_start + result.local_frame
                        if result.source_frame != expected_source_frame:
                            raise AssertionError(
                                "placeholder source/local frame indices do not reconcile"
                            )
                        failure = {
                            "layout": spec.layout,
                            "source_sequence": spec.source_sequence,
                            "split": spec.split,
                            "scene_id": spec.scene_id,
                            "view": view,
                            "local_frame": result.local_frame,
                            "source_frame": result.source_frame,
                            "error": result.source_decode_error,
                            "output_file": (
                                f"view_{view}/rgba_{result.local_frame:05d}.jpg"
                            ),
                            "recovery": "black_output_resolution_jpeg_quality_95",
                        }
                        view_stats["rgb"]["source_decode_failures"].append(failure)
                        view_stats["rgb"]["invalid_frame_indices"].append(
                            result.local_frame
                        )
                        view_stats["rgb"]["source_decode_failure_count"] += 1
                        view_stats["rgb"]["placeholder_file_count"] += 1
            if view_stats["rgb"]["output_frame_count"] != spec.frame_count:
                raise AssertionError("RGB output frame count does not match scene frame count")
            invalid_view_frames = view_stats["rgb"]["invalid_frame_indices"]
            if invalid_view_frames != sorted(set(invalid_view_frames)):
                raise AssertionError("invalid RGB frame indices must be sorted and unique")
            if view_stats["rgb"]["source_decode_failure_count"] != len(
                view_stats["rgb"]["source_decode_failures"]
            ):
                raise AssertionError("RGB decode-failure counts do not reconcile")
            if [
                failure["local_frame"]
                for failure in view_stats["rgb"]["source_decode_failures"]
            ] != invalid_view_frames:
                raise AssertionError("RGB decode failures do not match invalid frames")
            if view_stats["rgb"]["placeholder_file_count"] != len(
                invalid_view_frames
            ):
                raise AssertionError("RGB placeholder counts do not reconcile")
            rgb_seconds = time.perf_counter() - rgb_start
            view_stats["written_view_regular_file_count"] += view_stats["rgb"][
                "written_file_count"
            ]
            view_stats["exclusive_stage_seconds"][
                "rgb_decode_resize_encode_write"
            ] = rgb_seconds
            scene_stats[(spec.split, spec.scene_id)]["exclusive_stage_seconds"][
                "rgb_decode_resize_encode_write"
            ] += rgb_seconds

        del images

    for spec in source_specs:
        stats = scene_stats[(spec.split, spec.scene_id)]
        view_values = list(stats["views"].values())
        finite_mins = [
            float(item["prepared_depth"]["finite_min"])
            for item in view_values
            if item["prepared_depth"]["finite_min"] is not None
        ]
        finite_maxes = [
            float(item["prepared_depth"]["finite_max"])
            for item in view_values
            if item["prepared_depth"]["finite_max"] is not None
        ]
        stats["prepared_depth_all_views"] = {
            "value_count": sum(int(item["prepared_depth"]["value_count"]) for item in view_values),
            "finite_count": sum(
                int(item["prepared_depth"]["finite_count"]) for item in view_values
            ),
            "positive_count": sum(
                int(item["prepared_depth"]["positive_count"]) for item in view_values
            ),
            "zero_count": sum(
                int(item["prepared_depth"]["zero_count"]) for item in view_values
            ),
            "negative_count": sum(
                int(item["prepared_depth"]["negative_count"]) for item in view_values
            ),
            "nonfinite_count": sum(
                int(item["prepared_depth"]["nonfinite_count"]) for item in view_values
            ),
            "finite_min": min(finite_mins) if finite_mins else None,
            "finite_max": max(finite_maxes) if finite_maxes else None,
        }
        stats["rgb_all_views"] = {
            "source_frame_count": sum(
                int(item["rgb"]["source_frame_count"]) for item in view_values
            ),
            "output_frame_count": sum(
                int(item["rgb"]["output_frame_count"]) for item in view_values
            ),
            "written_file_count": sum(
                int(item["rgb"]["written_file_count"]) for item in view_values
            ),
            "output_bytes": sum(int(item["rgb"]["output_bytes"]) for item in view_values),
            "source_decode_failure_count": sum(
                int(item["rgb"]["source_decode_failure_count"])
                for item in view_values
            ),
            "placeholder_file_count": sum(
                int(item["rgb"]["placeholder_file_count"])
                for item in view_values
            ),
        }
        planned_scene_files = 2 + sum(
            int(item["planned_view_regular_file_count"]) for item in view_values
        )
        written_before_scene_metadata_files = 1 + sum(
            int(item["written_view_regular_file_count"]) for item in view_values
        )
        stats["io_counts"] = {
            "source": {
                "source_sequence_frame_count": spec.source_frame_count,
                "source_chunk_frame_count": spec.frame_count,
                "source_camera_chunk_frame_count": spec.frame_count * len(VIEW_IDS),
                "logical_asset_reference_count": 2 + 5 * len(VIEW_IDS),
                "logical_asset_semantics": (
                    "tracks and queries plus images, depth, intrinsics, extrinsics, and "
                    "visibility per view; each Zarr store counts as one logical asset"
                ),
            },
            "output": {
                "scene_frame_count": spec.frame_count,
                "camera_frame_count": spec.frame_count * len(VIEW_IDS),
                "jpeg_file_count": spec.frame_count * len(VIEW_IDS),
                "npy_file_count": 1 + 4 * len(VIEW_IDS),
                "json_file_count": 1,
                "planned_scene_regular_file_count": planned_scene_files,
                "written_before_scene_metadata_file_count": (
                    written_before_scene_metadata_files
                ),
            },
        }
        stats["accounted_scene_stage_total"] = sum(
            float(seconds) for seconds in stats["exclusive_stage_seconds"].values()
        )


def _scene_metadata(
    spec: SceneSpec,
    recorder: ValidationRecorder,
    stats: dict[str, Any],
) -> dict[str, Any]:
    invalid_rgb_frame_indices = sorted(
        {
            int(frame)
            for view_stats in stats["views"].values()
            for frame in view_stats["rgb"]["invalid_frame_indices"]
        }
    )
    if any(frame < 0 or frame >= spec.frame_count for frame in invalid_rgb_frame_indices):
        raise AssertionError("scene invalid RGB frame index is outside the scene")
    return {
        "schema_version": SCHEMA_VERSION,
        "format": "pointodyssey_mvtracker_preprocessed",
        "split": spec.split,
        "scene_id": spec.scene_id,
        "source": {
            "layout": spec.layout,
            "sequence": spec.source_sequence,
            "environment_family": spec.environment_family,
            "relative_scene_path": str(SOURCE_SUBROOTS[spec.layout] / spec.source_sequence),
            "frame_range_half_open": [spec.source_frame_start, spec.source_frame_end],
            "frame_count": spec.source_frame_count,
            "fps_provenance": spec.source_fps,
        },
        "output": {
            "frame_count": spec.frame_count,
            "views": list(VIEW_IDS),
            "resolution_hw": [OUTPUT_HEIGHT, OUTPUT_WIDTH],
            "rgb": {
                "format": "jpeg",
                "quality": JPEG_QUALITY,
                "resize_interpolation": "cv2.INTER_LINEAR",
                "invalid_frame_indices": invalid_rgb_frame_indices,
                "invalid_frame_semantics": (
                    "scene-local frame indices for which at least one view's source "
                    "JPEG failed cv2.imdecode; loaders must not sample windows containing "
                    "these frames"
                ),
                "decode_failure_placeholder": {
                    "image": "constant_black",
                    "resolution_hw": [OUTPUT_HEIGHT, OUTPUT_WIDTH],
                    "quality": JPEG_QUALITY,
                    "training_use": "forbidden_by_invalid_frame_indices",
                },
            },
            "depth": {
                "format": "npy",
                "dtype": "float32",
                "semantics": "optical_z_meters",
                "invalid_value": 0.0,
                "clipped": False,
                "resize_interpolation": "cv2.INTER_NEAREST",
            },
            "visibility": {
                "format": "npy",
                "dtype": "bool",
                "geometric_gate": (
                    "source-visible, finite track and projection, positive camera-Z, "
                    "inside resized pixel-center bounds"
                ),
                "depth_track_consistency": {
                    "depth_source": "exact resized float32 optical-Z frame written to depth.npy",
                    "nearest_pixel_rule": "floor(coordinate + 0.5)",
                    "neighborhood": (
                        "in-bounds samples from the 3x3 around nearest pixel; "
                        "out-of-bounds offsets are skipped"
                    ),
                    "acceptance": (
                        "at least one finite positive depth with "
                        "abs(depth-camera_z) <= tolerance_metres"
                    ),
                    "tolerance_metres": DEPTH_TRACK_TOLERANCE_METRES,
                },
            },
            "intrinsic_scale_xy": [SCALE_X, SCALE_Y],
        },
        "validation": {
            "failure_count": len(recorder.for_scene(spec)),
            "failures": recorder.for_scene(spec),
        },
        "statistics": stats,
    }


def _write_scene_metadata(
    build_root: Path,
    specs: Sequence[SceneSpec],
    recorder: ValidationRecorder,
    scene_stats: dict[tuple[str, str], dict[str, Any]],
) -> None:
    for spec in specs:
        metadata = _scene_metadata(
            spec,
            recorder,
            scene_stats[(spec.split, spec.scene_id)],
        )
        with (_output_scene_dir(build_root, spec) / "scene.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")


def _validate_output_tree(
    build_root: Path,
    specs: Sequence[SceneSpec],
    process_pool: ProcessPoolExecutor | None = None,
    process_workers: int = 1,
) -> dict[str, int]:
    if process_workers <= 0:
        raise ValueError("process_workers must be positive")
    expected_scene_dirs = {(spec.split, spec.scene_id) for spec in specs}
    actual_scene_dirs: set[tuple[str, str]] = set()
    forbidden_names = {"queries_xytv.npy", "trajs_2d.npy", "trajs_3d.npy", "annotations.npz"}
    for split in ("train", "validation", "test"):
        split_root = _require_dir(build_root / split)
        for scene_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            actual_scene_dirs.add((split, scene_dir.name))
    if actual_scene_dirs != expected_scene_dirs:
        raise ValueError("prepared output scene set does not match the fixed split plan")
    offenders = [str(path) for path in build_root.rglob("*") if path.is_file() and path.name in forbidden_names]
    if offenders:
        raise ValueError(f"forbidden derived/query files in prepared tree: {offenders[:10]}")

    expected_output_file_count = 0
    expected_rgb_file_count = 0
    actual_rgb_file_count = 0
    validated_rgb_file_count = 0
    for spec in specs:
        scene_dir = _output_scene_dir(build_root, spec)
        _require_file(scene_dir / "scene.json")
        _require_array(scene_dir / "tracks_3d.npy", (spec.frame_count, POINT_COUNT, 3), np.float32)
        expected_output_file_count += 2 + len(VIEW_IDS) * (4 + spec.frame_count)
        for view in VIEW_IDS:
            view_dir = _require_dir(scene_dir / f"view_{view}")
            _require_array(view_dir / "depth.npy", (spec.frame_count, OUTPUT_HEIGHT, OUTPUT_WIDTH), np.float32)
            _require_array(view_dir / "intrinsics.npy", (3, 3), np.float32)
            _require_array(view_dir / "extrinsics_w2c.npy", (spec.frame_count, 3, 4), np.float32)
            _require_array(view_dir / "visibility.npy", (spec.frame_count, POINT_COUNT), np.bool_)
            expected_jpegs = [f"rgba_{frame:05d}.jpg" for frame in range(spec.frame_count)]
            actual_jpegs = sorted(path.name for path in view_dir.glob("rgba_*.jpg"))
            expected_rgb_file_count += len(expected_jpegs)
            actual_rgb_file_count += len(actual_jpegs)
            if actual_jpegs != expected_jpegs:
                raise ValueError(f"non-contiguous JPEG sequence in {view_dir}")
            validation_jobs = (
                JPEGValidationJob(
                    split=spec.split,
                    scene_id=spec.scene_id,
                    view=view,
                    frame=frame,
                    path=str(view_dir / jpeg_name),
                    expected_height=OUTPUT_HEIGHT,
                    expected_width=OUTPUT_WIDTH,
                )
                for frame, jpeg_name in enumerate(expected_jpegs)
            )
            for count in _run_batches(
                _iter_validation_batches(validation_jobs),
                _validate_jpeg_batch,
                process_pool=process_pool,
                process_workers=process_workers,
                stage="persisted JPEG validation",
            ):
                validated_rgb_file_count += count
        actual_view_dirs = {path.name for path in scene_dir.iterdir() if path.is_dir()}
        expected_view_dirs = {f"view_{view}" for view in VIEW_IDS}
        if actual_view_dirs != expected_view_dirs:
            raise ValueError(f"{scene_dir} must contain exactly {sorted(expected_view_dirs)}")

    actual_output_file_count = sum(
        1 for path in build_root.rglob("*") if path.is_file()
    )
    if actual_output_file_count != expected_output_file_count:
        raise ValueError(
            "prepared output file count does not match the exact contract: "
            f"expected {expected_output_file_count}, got {actual_output_file_count}"
        )
    if validated_rgb_file_count != expected_rgb_file_count:
        raise AssertionError("validated JPEG count does not match expected JPEG count")
    return {
        "planned_temporary_tree_file_count_before_root_report": expected_output_file_count,
        "written_temporary_tree_file_count_before_root_report": actual_output_file_count,
        "planned_rgb_file_count": expected_rgb_file_count,
        "written_rgb_file_count": actual_rgb_file_count,
        "validated_rgb_file_count": validated_rgb_file_count,
    }


def _aggregate_scene_statistics(
    specs: Sequence[SceneSpec],
    scene_stats: dict[tuple[str, str], dict[str, Any]],
    output_validation: dict[str, int],
    timings_seconds: dict[str, float],
) -> dict[str, Any]:
    track_fields = (
        "track_id_count",
        "frame_count",
        "track_sample_count",
        "coordinate_value_count",
        "finite_coordinate_value_count",
        "nonfinite_coordinate_value_count",
    )
    depth_count_fields = (
        "value_count",
        "finite_count",
        "positive_count",
        "zero_count",
        "negative_count",
        "nonfinite_count",
    )
    rgb_fields = (
        "source_frame_count",
        "output_frame_count",
        "written_file_count",
        "output_bytes",
        "source_decode_failure_count",
        "placeholder_file_count",
    )
    visibility_fields = (
        "visibility_true_before_gating",
        "visibility_true_after_geometric_gating",
        "visibility_removed_by_geometric_gating",
        "depth_consistency_candidate_count",
        "visibility_rejected_no_valid_depth",
        "visibility_rejected_residual_over_tolerance",
        "visibility_true_after_depth_consistency_gating",
        "visibility_removed_by_depth_consistency_gating",
        "visibility_true_after_gating",
        "visibility_removed_by_gating",
    )
    track_totals = {field: 0 for field in track_fields}
    depth_totals = {field: 0 for field in depth_count_fields}
    rgb_totals = {field: 0 for field in rgb_fields}
    visibility_totals = {field: 0 for field in visibility_fields}
    source_chunk_frame_count = 0
    source_camera_chunk_frame_count = 0
    source_logical_asset_reference_count = 0
    output_scene_frame_count = 0
    output_camera_frame_count = 0
    output_npy_file_count = 0
    output_scene_json_file_count = 0
    planned_scene_regular_file_count = 0
    written_before_scene_metadata_file_count = 0
    scene_accounted_stage_total = 0.0
    decode_failures: list[dict[str, Any]] = []
    invalid_scene_frames: list[dict[str, Any]] = []
    finite_min: float | None = None
    finite_max: float | None = None
    for spec in specs:
        stats = scene_stats[(spec.split, spec.scene_id)]
        for field in track_fields:
            track_totals[field] += int(stats["tracks"][field])
        for field in depth_count_fields:
            depth_totals[field] += int(stats["prepared_depth_all_views"][field])
        for field in rgb_fields:
            rgb_totals[field] += int(stats["rgb_all_views"][field])
        scene_depth = stats["prepared_depth_all_views"]
        if scene_depth["finite_min"] is not None:
            value = float(scene_depth["finite_min"])
            finite_min = value if finite_min is None else min(finite_min, value)
        if scene_depth["finite_max"] is not None:
            value = float(scene_depth["finite_max"])
            finite_max = value if finite_max is None else max(finite_max, value)
        scene_invalid_rgb_frames: set[int] = set()
        for view_stats in stats["views"].values():
            for field in visibility_fields:
                visibility_totals[field] += int(view_stats[field])
            view_rgb = view_stats["rgb"]
            view_failures = view_rgb["source_decode_failures"]
            view_invalid_frames = view_rgb["invalid_frame_indices"]
            if int(view_rgb["source_decode_failure_count"]) != len(view_failures):
                raise AssertionError("scene RGB decode-failure counts do not reconcile")
            if int(view_rgb["placeholder_file_count"]) != len(view_invalid_frames):
                raise AssertionError("scene RGB placeholder counts do not reconcile")
            if len(view_failures) != len(view_invalid_frames):
                raise AssertionError("scene RGB failure and placeholder counts differ")
            decode_failures.extend(view_failures)
            scene_invalid_rgb_frames.update(int(frame) for frame in view_invalid_frames)
        if scene_invalid_rgb_frames:
            invalid_scene_frames.append(
                {
                    "split": spec.split,
                    "scene_id": spec.scene_id,
                    "layout": spec.layout,
                    "source_sequence": spec.source_sequence,
                    "invalid_frame_indices": sorted(scene_invalid_rgb_frames),
                }
            )
        source_io = stats["io_counts"]["source"]
        output_io = stats["io_counts"]["output"]
        source_chunk_frame_count += int(source_io["source_chunk_frame_count"])
        source_camera_chunk_frame_count += int(
            source_io["source_camera_chunk_frame_count"]
        )
        source_logical_asset_reference_count += int(
            source_io["logical_asset_reference_count"]
        )
        output_scene_frame_count += int(output_io["scene_frame_count"])
        output_camera_frame_count += int(output_io["camera_frame_count"])
        output_npy_file_count += int(output_io["npy_file_count"])
        output_scene_json_file_count += int(output_io["json_file_count"])
        planned_scene_regular_file_count += int(
            output_io["planned_scene_regular_file_count"]
        )
        written_before_scene_metadata_file_count += int(
            output_io["written_before_scene_metadata_file_count"]
        )
        scene_accounted_stage_total += float(stats["accounted_scene_stage_total"])

    if track_totals["finite_coordinate_value_count"] + track_totals[
        "nonfinite_coordinate_value_count"
    ] != track_totals["coordinate_value_count"]:
        raise AssertionError("root track finite-value counts do not reconcile")
    if depth_totals["finite_count"] + depth_totals["nonfinite_count"] != depth_totals[
        "value_count"
    ]:
        raise AssertionError("root prepared-depth finite counts do not reconcile")
    if depth_totals["positive_count"] + depth_totals["zero_count"] + depth_totals[
        "negative_count"
    ] != depth_totals["finite_count"]:
        raise AssertionError("root prepared-depth sign counts do not reconcile")
    if (finite_min is None) != (depth_totals["finite_count"] == 0) or (
        (finite_max is None) != (depth_totals["finite_count"] == 0)
    ):
        raise AssertionError("root prepared-depth finite extrema do not reconcile")
    if rgb_totals["source_frame_count"] != source_camera_chunk_frame_count:
        raise AssertionError("source RGB frames do not reconcile with source camera frames")
    if rgb_totals["output_frame_count"] != output_camera_frame_count:
        raise AssertionError("output RGB frames do not reconcile with output camera frames")
    if rgb_totals["written_file_count"] != rgb_totals["output_frame_count"]:
        raise AssertionError("output RGB files do not reconcile with output RGB frames")
    if rgb_totals["source_decode_failure_count"] != len(decode_failures):
        raise AssertionError("root RGB decode-failure counts do not reconcile")
    if rgb_totals["placeholder_file_count"] != len(decode_failures):
        raise AssertionError("root RGB placeholder counts do not reconcile")

    before = visibility_totals["visibility_true_before_gating"]
    after_geometry = visibility_totals["visibility_true_after_geometric_gating"]
    after_depth = visibility_totals[
        "visibility_true_after_depth_consistency_gating"
    ]
    no_depth = visibility_totals["visibility_rejected_no_valid_depth"]
    residual = visibility_totals["visibility_rejected_residual_over_tolerance"]
    if before != after_geometry + visibility_totals["visibility_removed_by_geometric_gating"]:
        raise AssertionError("root geometric-visibility counts do not reconcile")
    if visibility_totals["depth_consistency_candidate_count"] != after_geometry:
        raise AssertionError("root depth-consistency candidate counts do not reconcile")
    if after_geometry != after_depth + no_depth + residual:
        raise AssertionError("root depth-consistency rejection counts do not reconcile")
    if visibility_totals["visibility_removed_by_depth_consistency_gating"] != (
        no_depth + residual
    ):
        raise AssertionError("root depth-consistency removed counts do not reconcile")
    if visibility_totals["visibility_true_after_gating"] != after_depth:
        raise AssertionError("root final visibility aliases do not reconcile")
    if visibility_totals["visibility_removed_by_gating"] != before - after_depth:
        raise AssertionError("root total visibility-removed counts do not reconcile")

    planned_tree_files = output_validation[
        "planned_temporary_tree_file_count_before_root_report"
    ]
    written_tree_files = output_validation[
        "written_temporary_tree_file_count_before_root_report"
    ]
    if planned_scene_regular_file_count != planned_tree_files:
        raise AssertionError("planned scene files do not reconcile with output validation")
    if written_tree_files != planned_tree_files:
        raise AssertionError("written temporary-tree files do not reconcile with plan")
    if output_scene_json_file_count != len(specs):
        raise AssertionError("scene metadata file counts do not reconcile with scene count")
    if rgb_totals["written_file_count"] != output_validation["written_rgb_file_count"]:
        raise AssertionError("written RGB files do not reconcile with output validation")
    if rgb_totals["written_file_count"] != output_validation["planned_rgb_file_count"]:
        raise AssertionError("planned RGB files do not reconcile with conversion")
    if rgb_totals["written_file_count"] != output_validation["validated_rgb_file_count"]:
        raise AssertionError("validated RGB files do not reconcile with conversion")

    source_sequence_count = len({spec.source_key for spec in specs})
    source_unique_frame_count = sum(
        grouped_specs[0].source_frame_count for grouped_specs in _group_by_source(specs)
    )
    source_distinct_logical_asset_count = source_sequence_count * (
        2 + 5 * len(VIEW_IDS)
    )

    canonical_timing_fields = (
        "path_and_scene_spec_setup",
        "preflight",
        "temporary_tree_setup",
        "source_conversion",
        "scene_metadata_write",
        "output_validation",
        "process_pool_shutdown",
    )
    for field in (*canonical_timing_fields, "measured_total", "accounted_stage_total", "unattributed_orchestration"):
        value = float(timings_seconds[field])
        if not np.isfinite(value) or value < 0.0:
            raise AssertionError(f"invalid root timing value for {field}: {value}")
    calculated_accounted_total = sum(
        float(timings_seconds[field]) for field in canonical_timing_fields
    )
    if not np.isclose(
        calculated_accounted_total,
        float(timings_seconds["accounted_stage_total"]),
        rtol=1e-9,
        atol=1e-9,
    ):
        raise AssertionError("root accounted timing does not reconcile")
    if not np.isclose(
        calculated_accounted_total + float(timings_seconds["unattributed_orchestration"]),
        float(timings_seconds["measured_total"]),
        rtol=1e-9,
        atol=1e-9,
    ):
        raise AssertionError("root measured timing does not reconcile")

    return {
        "scene_count": len(specs),
        "source_sequence_count": source_sequence_count,
        "tracks": {
            "prepared_scene_track_id_slot_count": track_totals["track_id_count"],
            "prepared_scene_frame_count": track_totals["frame_count"],
            "track_sample_count": track_totals["track_sample_count"],
            "coordinate_value_count": track_totals["coordinate_value_count"],
            "finite_coordinate_value_count": track_totals[
                "finite_coordinate_value_count"
            ],
            "nonfinite_coordinate_value_count": track_totals[
                "nonfinite_coordinate_value_count"
            ],
        },
        "prepared_depth": {
            **depth_totals,
            "finite_min": finite_min,
            "finite_max": finite_max,
        },
        "rgb": {
            **rgb_totals,
            "invalid_scene_frame_count": sum(
                len(item["invalid_frame_indices"]) for item in invalid_scene_frames
            ),
            "scenes_with_invalid_rgb_count": len(invalid_scene_frames),
            "invalid_scene_frames": invalid_scene_frames,
            "source_decode_failures": decode_failures,
        },
        "visibility": {
            **visibility_totals,
            "depth_consistency_tolerance_metres": DEPTH_TRACK_TOLERANCE_METRES,
        },
        "io_counts": {
            "source_sequence_count": source_sequence_count,
            "source_unique_frame_count": source_unique_frame_count,
            "source_chunk_frame_count": source_chunk_frame_count,
            "source_camera_chunk_frame_count": source_camera_chunk_frame_count,
            "source_distinct_logical_asset_count": source_distinct_logical_asset_count,
            "source_logical_asset_reference_count": (
                source_logical_asset_reference_count
            ),
            "source_logical_asset_semantics": (
                "tracks and queries plus images, depth, intrinsics, extrinsics, and "
                "visibility per view; each Zarr store counts as one logical asset"
            ),
            "prepared_scene_count": len(specs),
            "output_scene_frame_count": output_scene_frame_count,
            "output_camera_frame_count": output_camera_frame_count,
            "output_jpeg_file_count": rgb_totals["written_file_count"],
            "output_npy_file_count": output_npy_file_count,
            "output_scene_json_file_count": output_scene_json_file_count,
            "output_root_json_file_count": 1,
            "converted_file_count_before_scene_metadata": (
                written_before_scene_metadata_file_count
            ),
            "temporary_tree_file_count_before_root_report": written_tree_files,
            "published_regular_file_count": written_tree_files + 1,
        },
        "output_validation": output_validation,
        "timing": {
            "clock": "time.perf_counter",
            "scope": (
                "path setup through process-pool shutdown; report write and atomic "
                "publication are excluded"
            ),
            "scene_exclusive_stage_seconds": scene_accounted_stage_total,
            "wall_seconds": timings_seconds,
        },
    }


def _report(
    specs: Sequence[SceneSpec],
    recorder: ValidationRecorder,
    scene_stats: dict[tuple[str, str], dict[str, Any]] | None = None,
    output_validation: dict[str, int] | None = None,
    timings_seconds: dict[str, float] | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    split_counts = {}
    for split in ("train", "validation", "test"):
        split_specs = [spec for spec in specs if spec.split == split]
        split_counts[split] = {
            "prepared_scenes": len(split_specs),
            "frames": sum(spec.frame_count for spec in split_specs),
            "source_sequences": len({spec.source_key for spec in split_specs}),
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "format": "pointodyssey_mvtracker_preprocessed_validation",
        "status": "completed_with_ignored_validation_failures" if recorder.failures else "completed",
        "ignore_validation_failures": recorder.ignore_failures,
        "counts": split_counts,
        "totals": {
            "prepared_scenes": len(specs),
            "frames": sum(spec.frame_count for spec in specs),
            "source_sequences": len({spec.source_key for spec in specs}),
            "semantic_validation_failures": len(recorder.failures),
        },
        "failures": recorder.failures,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "opencv_parent_thread_count": int(cv2.getNumThreads()),
        },
        "workers": workers,
        "processing_config": {
            "worker_count": workers,
            "process_pool_enabled": workers > 1,
            "process_start_method": "spawn" if workers > 1 else None,
            "opencv_threads_per_process_worker": 1,
        },
        "split_plan": [asdict(spec) for spec in specs],
    }
    if scene_stats is not None and output_validation is not None:
        report["statistics"] = _aggregate_scene_statistics(
            specs,
            scene_stats,
            output_validation,
            timings_seconds or {},
        )
    return report


def preprocess(
    source_root: Path,
    output_root: Path,
    *,
    ignore_validation_failures: bool = False,
    workers: int = DEFAULT_WORKERS,
) -> None:
    if workers <= 0:
        raise ValueError("workers must be a positive integer")
    total_start = time.perf_counter()
    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    specs = build_scene_specs()
    path_setup_seconds = time.perf_counter() - total_start
    preflight_start = time.perf_counter()
    preflight(source_root, output_root, specs)
    timings_seconds: dict[str, float] = {
        "path_and_scene_spec_setup": path_setup_seconds,
        "preflight": time.perf_counter() - preflight_start,
    }
    recorder = ValidationRecorder(ignore_validation_failures)
    scene_stats: dict[tuple[str, str], dict[str, Any]] = {}

    build_root: Path | None = None
    process_pool: ProcessPoolExecutor | None = None
    try:
        build_root = Path(
            tempfile.mkdtemp(
                prefix=f".{output_root.name}.tmp-",
                dir=str(output_root.parent),
            )
        )
        if workers > 1:
            process_pool = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_process_worker,
            )
        stage_start = time.perf_counter()
        _prepare_scene_roots(build_root, specs)
        temporary_tree_setup_seconds = time.perf_counter() - stage_start
        timings_seconds["temporary_tree_setup"] = temporary_tree_setup_seconds
        stage_start = time.perf_counter()
        for source_specs in _group_by_source(specs):
            print(
                f"POINTODYSSEY_PREPROCESS_SOURCE {source_specs[0].layout} "
                f"{source_specs[0].source_sequence}",
                flush=True,
            )
            _convert_source_group(
                source_root,
                build_root,
                source_specs,
                recorder,
                scene_stats,
                process_pool=process_pool,
                process_workers=workers,
            )
        timings_seconds["source_conversion"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        _write_scene_metadata(build_root, specs, recorder, scene_stats)
        timings_seconds["scene_metadata_write"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        output_validation = _validate_output_tree(
            build_root,
            specs,
            process_pool=process_pool,
            process_workers=workers,
        )
        timings_seconds["output_validation"] = time.perf_counter() - stage_start
        stage_start = time.perf_counter()
        if process_pool is not None:
            process_pool.shutdown(wait=True, cancel_futures=True)
            process_pool = None
        timings_seconds["process_pool_shutdown"] = time.perf_counter() - stage_start
        measured_total = time.perf_counter() - total_start
        accounted_fields = (
            "path_and_scene_spec_setup",
            "preflight",
            "temporary_tree_setup",
            "source_conversion",
            "scene_metadata_write",
            "output_validation",
            "process_pool_shutdown",
        )
        accounted_total = sum(timings_seconds[field] for field in accounted_fields)
        timings_seconds["measured_total"] = measured_total
        timings_seconds["accounted_stage_total"] = accounted_total
        timings_seconds["unattributed_orchestration"] = measured_total - accounted_total
        report = _report(
            specs,
            recorder,
            scene_stats,
            output_validation,
            timings_seconds,
            workers,
        )
        report_start = time.perf_counter()
        with (build_root / "validation_report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        report_write_seconds = time.perf_counter() - report_start
        recorder.raise_if_strict()
        if output_root.exists():
            raise FileExistsError(f"output root appeared during preprocessing: {output_root}")
        os.replace(build_root, output_root)
    except BaseException:
        if process_pool is not None:
            try:
                process_pool.shutdown(wait=True, cancel_futures=True)
            except BaseException as shutdown_exc:
                print(
                    f"POINTODYSSEY_PREPROCESS_SHUTDOWN_ERROR {shutdown_exc!r}",
                    file=sys.stderr,
                    flush=True,
                )
        if build_root is not None:
            shutil.rmtree(build_root, ignore_errors=True)
        raise

    total_seconds = time.perf_counter() - total_start
    print(
        f"POINTODYSSEY_PREPROCESS_DONE output={output_root} scenes={len(specs)} "
        f"semantic_failures={len(recorder.failures)} workers={workers} "
        f"report_write_seconds={report_write_seconds:.6f} total_seconds={total_seconds:.6f}",
        flush=True,
    )


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=DEFAULT_WORKERS,
        help=(
            "Number of RGB/JPEG worker processes. Use 1 for the deterministic serial path "
            f"(default: {DEFAULT_WORKERS})."
        ),
    )
    parser.add_argument(
        "--ignore-validation-failures",
        action="store_true",
        help=(
            "Publish data with recorded semantic validation failures. Structural, decoding, "
            "shape, and I/O failures remain fatal."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preprocess(
        args.source_root,
        args.output_root,
        ignore_validation_failures=args.ignore_validation_failures,
        workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
