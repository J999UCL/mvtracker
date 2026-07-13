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
import os
import platform
import shutil
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


SCHEMA_VERSION = 1
VIEW_IDS = (0, 1, 2, 3)
POINT_COUNT = 2600
SOURCE_HEIGHT = 1080
SOURCE_WIDTH = 1920
OUTPUT_HEIGHT = 384
OUTPUT_WIDTH = 512
JPEG_QUALITY = 95
LONG_CHUNK_LENGTH = 120
PROJECTION_TOLERANCE_PX = 0.01
RIGID_DETERMINANT_TOLERANCE = 1e-3
SCALE_X = OUTPUT_WIDTH / SOURCE_WIDTH
SCALE_Y = OUTPUT_HEIGHT / SOURCE_HEIGHT

SOURCE_SUBROOTS = {
    "raw": Path("raw_fp32_tarzst/tapvid3d_raw_fp32/track_a_train"),
    "short": Path(
        "shortform_zip/"
        "mixamo_camera_super_aggro_v1_fixed_intrinsics_1080p_s16_p2600_viewunion/"
        "track_a_train"
    ),
    "long": Path("longform_tar/tapvid3d_v5_zarr_combined_20260627/track_a_train"),
}

SOURCE_FRAME_COUNTS = {"raw": 120, "short": 61, "long": 2000}
SOURCE_FPS = {"raw": 30, "short": 15, "long": 15}


@dataclass(frozen=True)
class SourceAssignment:
    split: str
    layout: str
    sequence: str
    environment_family: str
    first_scene_id: int


ASSIGNMENTS = (
    SourceAssignment("train", "raw", "candidate_empty_office", "empty_office", 0),
    SourceAssignment("train", "raw", "indoor_00_pawn_shop_manual", "indoor_00_pawn_shop", 1),
    SourceAssignment("train", "raw", "indoor_09_barbershop_manual", "indoor_09_barbershop", 2),
    SourceAssignment("train", "raw", "warehouse_ledge_manual", "warehouse", 3),
    SourceAssignment("train", "short", "indoor_00_pawn_shop", "indoor_00_pawn_shop", 4),
    SourceAssignment("train", "short", "indoor_05_modern_kitchen", "indoor_05_modern_kitchen", 5),
    SourceAssignment("train", "short", "indoor_06_office", "indoor_06_office", 6),
    SourceAssignment("train", "short", "outdoor_00_namaqualand", "outdoor_00_namaqualand", 7),
    SourceAssignment("train", "short", "outdoor_07_forest_road", "outdoor_07_forest_road", 8),
    SourceAssignment("train", "short", "outdoor_08_seacliff_beach", "outdoor_08_seacliff_beach", 9),
    SourceAssignment("train", "long", "candidate_empty_office", "empty_office", 10),
    SourceAssignment("train", "long", "candidate_parking", "candidate_parking", 27),
    SourceAssignment("train", "long", "candidate_warehouse", "warehouse", 44),
    SourceAssignment("train", "long", "og_parking_lot", "og_parking_lot", 61),
    SourceAssignment("validation", "raw", "indoor_01_classroom_manual", "indoor_01_classroom", 0),
    SourceAssignment("validation", "raw", "outdoor_02_hidden_alley_manual", "outdoor_02_hidden_alley", 1),
    SourceAssignment("validation", "short", "indoor_01_classroom", "indoor_01_classroom", 2),
    SourceAssignment("test", "raw", "indoor_04_modern_loft_manual", "indoor_04_modern_loft", 0),
    SourceAssignment("test", "raw", "outdoor_06_city_scene_manual", "outdoor_06_city_scene", 1),
    SourceAssignment("test", "short", "indoor_04_modern_loft", "indoor_04_modern_loft", 2),
)


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
) -> None:
    finite = np.isfinite(array)
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
) -> None:
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

    for spec in source_specs:
        output_scene = _output_scene_dir(build_root, spec)
        tracks_chunk = np.asarray(tracks[spec.source_frame_start : spec.source_frame_end], dtype=np.float32)
        np.save(output_scene / "tracks_3d.npy", np.ascontiguousarray(tracks_chunk))
        _record_finite_failure(
            recorder,
            spec,
            "tracks_finite",
            tracks_chunk,
            "all world-track coordinates are finite",
        )
        scene_stats[(spec.split, spec.scene_id)] = {"views": {}}

    for view in VIEW_IDS:
        source_view = source_dir / str(view)
        images = np.load(source_view / "images_jpeg_bytes.npy", allow_pickle=True)
        if images.shape != (first.source_frame_count,) or images.dtype != np.dtype(object):
            raise ValueError(f"invalid object JPEG array: {source_view / 'images_jpeg_bytes.npy'}")
        for frame, image_bytes in enumerate(images):
            if not isinstance(image_bytes, np.ndarray) or image_bytes.ndim != 1 or image_bytes.dtype != np.uint8:
                raise ValueError(
                    f"{source_view / 'images_jpeg_bytes.npy'} frame {frame} is not a 1-D uint8 JPEG vector"
                )

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

            projected_xy, camera_z = _project_points(tracks_chunk, extrinsics_chunk, resized_k)
            finite_tracks = np.isfinite(tracks_chunk).all(axis=-1)
            finite_projection = np.isfinite(projected_xy).all(axis=-1) & np.isfinite(camera_z)
            inside = (
                (projected_xy[..., 0] >= -0.5)
                & (projected_xy[..., 0] < OUTPUT_WIDTH - 0.5)
                & (projected_xy[..., 1] >= -0.5)
                & (projected_xy[..., 1] < OUTPUT_HEIGHT - 0.5)
            )
            output_visibility = source_visibility & finite_tracks & finite_projection & (camera_z > 0.0) & inside

            np.save(output_view / "intrinsics.npy", np.ascontiguousarray(resized_k))
            np.save(output_view / "extrinsics_w2c.npy", extrinsics_chunk)
            np.save(output_view / "visibility.npy", np.ascontiguousarray(output_visibility))

            before = int(source_visibility.sum())
            after = int(output_visibility.sum())
            scene_stats[(spec.split, spec.scene_id)]["views"][str(view)] = {
                "visibility_true_before_gating": before,
                "visibility_true_after_gating": after,
                "visibility_removed_by_gating": before - after,
            }

            depth_output = np.lib.format.open_memmap(
                output_view / "depth.npy",
                mode="w+",
                dtype=np.float32,
                shape=(spec.frame_count, OUTPUT_HEIGHT, OUTPUT_WIDTH),
            )
            first_nonfinite_frame: int | None = None
            first_negative_frame: int | None = None
            nonfinite_count = 0
            negative_count = 0
            for local_frame, source_frame in enumerate(range(start, end)):
                source_depth_frame = np.asarray(source_depth[source_frame], dtype=np.float32)
                nonfinite = ~np.isfinite(source_depth_frame)
                negative = source_depth_frame < 0.0
                if nonfinite.any() and first_nonfinite_frame is None:
                    first_nonfinite_frame = source_frame
                if negative.any() and first_negative_frame is None:
                    first_negative_frame = source_frame
                nonfinite_count += int(nonfinite.sum())
                negative_count += int(negative.sum())
                depth_output[local_frame] = cv2.resize(
                    source_depth_frame,
                    (OUTPUT_WIDTH, OUTPUT_HEIGHT),
                    interpolation=cv2.INTER_NEAREST,
                )
            depth_output.flush()
            del depth_output
            if nonfinite_count:
                recorder.record(
                    spec,
                    "depth_finite",
                    {"nonfinite_count": nonfinite_count},
                    "all source optical-Z values are finite",
                    view=view,
                    frame=first_nonfinite_frame,
                )
            if negative_count:
                recorder.record(
                    spec,
                    "depth_nonnegative",
                    {"negative_count": negative_count},
                    "source optical-Z is non-negative; zero denotes invalid depth",
                    view=view,
                    frame=first_negative_frame,
                )

            for local_frame, source_frame in enumerate(range(start, end)):
                encoded = images[source_frame]
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if decoded is None:
                    raise ValueError(f"cannot decode JPEG {source_view} frame {source_frame}")
                if decoded.shape != (SOURCE_HEIGHT, SOURCE_WIDTH, 3):
                    raise ValueError(
                        f"{source_view} frame {source_frame} decodes to {decoded.shape}, "
                        f"expected {(SOURCE_HEIGHT, SOURCE_WIDTH, 3)}"
                    )
                resized = cv2.resize(decoded, (OUTPUT_WIDTH, OUTPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
                ok, jpeg = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                if not ok:
                    raise RuntimeError(f"failed to encode resized JPEG {source_view} frame {source_frame}")
                (output_view / f"rgba_{local_frame:05d}.jpg").write_bytes(jpeg.tobytes())

        del images


def _scene_metadata(
    spec: SceneSpec,
    recorder: ValidationRecorder,
    stats: dict[str, Any],
) -> dict[str, Any]:
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
            },
            "depth": {
                "format": "npy",
                "dtype": "float32",
                "semantics": "optical_z_meters",
                "invalid_value": 0.0,
                "clipped": False,
                "resize_interpolation": "cv2.INTER_NEAREST",
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


def _validate_output_tree(build_root: Path, specs: Sequence[SceneSpec]) -> None:
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

    for spec in specs:
        scene_dir = _output_scene_dir(build_root, spec)
        _require_file(scene_dir / "scene.json")
        _require_array(scene_dir / "tracks_3d.npy", (spec.frame_count, POINT_COUNT, 3), np.float32)
        for view in VIEW_IDS:
            view_dir = _require_dir(scene_dir / f"view_{view}")
            _require_array(view_dir / "depth.npy", (spec.frame_count, OUTPUT_HEIGHT, OUTPUT_WIDTH), np.float32)
            _require_array(view_dir / "intrinsics.npy", (3, 3), np.float32)
            _require_array(view_dir / "extrinsics_w2c.npy", (spec.frame_count, 3, 4), np.float32)
            _require_array(view_dir / "visibility.npy", (spec.frame_count, POINT_COUNT), np.bool_)
            expected_jpegs = [f"rgba_{frame:05d}.jpg" for frame in range(spec.frame_count)]
            actual_jpegs = sorted(path.name for path in view_dir.glob("rgba_*.jpg"))
            if actual_jpegs != expected_jpegs:
                raise ValueError(f"non-contiguous JPEG sequence in {view_dir}")
            for jpeg_name in expected_jpegs:
                decoded = cv2.imread(str(view_dir / jpeg_name), cv2.IMREAD_COLOR)
                if decoded is None or decoded.shape != (OUTPUT_HEIGHT, OUTPUT_WIDTH, 3):
                    raise ValueError(f"invalid prepared JPEG: {view_dir / jpeg_name}")
        actual_view_dirs = {path.name for path in scene_dir.iterdir() if path.is_dir()}
        expected_view_dirs = {f"view_{view}" for view in VIEW_IDS}
        if actual_view_dirs != expected_view_dirs:
            raise ValueError(f"{scene_dir} must contain exactly {sorted(expected_view_dirs)}")


def _report(
    specs: Sequence[SceneSpec],
    recorder: ValidationRecorder,
) -> dict[str, Any]:
    split_counts = {}
    for split in ("train", "validation", "test"):
        split_specs = [spec for spec in specs if spec.split == split]
        split_counts[split] = {
            "prepared_scenes": len(split_specs),
            "frames": sum(spec.frame_count for spec in split_specs),
            "source_sequences": len({spec.source_key for spec in split_specs}),
        }
    return {
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
        },
        "split_plan": [asdict(spec) for spec in specs],
    }


def preprocess(source_root: Path, output_root: Path, *, ignore_validation_failures: bool = False) -> None:
    source_root = Path(source_root).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    specs = build_scene_specs()
    preflight(source_root, output_root, specs)
    recorder = ValidationRecorder(ignore_validation_failures)
    scene_stats: dict[tuple[str, str], dict[str, Any]] = {}

    build_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=str(output_root.parent))
    )
    try:
        _prepare_scene_roots(build_root, specs)
        for source_specs in _group_by_source(specs):
            print(
                f"POINTODYSSEY_PREPROCESS_SOURCE {source_specs[0].layout} "
                f"{source_specs[0].source_sequence}",
                flush=True,
            )
            _convert_source_group(source_root, build_root, source_specs, recorder, scene_stats)
        _write_scene_metadata(build_root, specs, recorder, scene_stats)
        _validate_output_tree(build_root, specs)
        report = _report(specs, recorder)
        with (build_root / "validation_report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        recorder.raise_if_strict()
        if output_root.exists():
            raise FileExistsError(f"output root appeared during preprocessing: {output_root}")
        os.replace(build_root, output_root)
    except Exception:
        shutil.rmtree(build_root, ignore_errors=True)
        raise

    print(
        f"POINTODYSSEY_PREPROCESS_DONE output={output_root} scenes={len(specs)} "
        f"semantic_failures={len(recorder.failures)}",
        flush=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
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
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
