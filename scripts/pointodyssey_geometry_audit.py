#!/usr/bin/env python3
"""Audit PointOdyssey track camera-Z against native-resolution source depth.

This is deliberately read-only with respect to the extracted dataset.  It
samples deterministic anchor frames from every approved source sequence and
reports residual distributions; it does not choose a training tolerance or
classify/ignore preprocessing failures.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

if __package__:
    from .pointodyssey_contract import (
        POINT_COUNT,
        SOURCE_FRAME_COUNTS,
        SOURCE_HEIGHT,
        SOURCE_SUBROOTS,
        SOURCE_WIDTH,
        VIEW_IDS,
        unique_source_keys,
    )
else:
    from pointodyssey_contract import (
        POINT_COUNT,
        SOURCE_FRAME_COUNTS,
        SOURCE_HEIGHT,
        SOURCE_SUBROOTS,
        SOURCE_WIDTH,
        VIEW_IDS,
        unique_source_keys,
    )


SCHEMA_VERSION = 2
LAYOUT_ORDER = ("raw", "long", "short")
SAMPLE_COUNTS = {"raw": 8, "short": 8, "long": 24}
TEMPORAL_OFFSETS = (-2, -1, 0, 1, 2)
SAMPLING_MODES = ("nearest_pixel", "bilinear", "neighborhood_3x3")
RAYCAST_VISIBILITY_REFERENCE_METRES = 0.05


@dataclass(frozen=True, order=True)
class BucketKey:
    layout: str
    sequence: str
    view: int
    temporal_offset: int
    sampling_mode: str


@dataclass
class ResidualBucket:
    candidate_count: int = 0
    signed_residual_parts: List[np.ndarray] = field(default_factory=list)

    def add(self, signed_residuals: np.ndarray, candidate_count: int) -> None:
        residuals = np.asarray(signed_residuals, dtype=np.float32)
        if residuals.ndim != 1 or not np.isfinite(residuals).all():
            raise ValueError("residual batches must be finite one-dimensional arrays")
        self.candidate_count += int(candidate_count)
        if residuals.size:
            self.signed_residual_parts.append(residuals)

    @property
    def valid_count(self) -> int:
        return sum(int(part.size) for part in self.signed_residual_parts)

    def values(self) -> np.ndarray:
        if not self.signed_residual_parts:
            return np.empty((0,), dtype=np.float32)
        if len(self.signed_residual_parts) == 1:
            return self.signed_residual_parts[0]
        return np.concatenate(self.signed_residual_parts)


def sample_anchor_frames(frame_count: int, sample_count: int) -> np.ndarray:
    """Select deterministic, evenly spaced frames with room for all offsets."""
    margin = max(abs(offset) for offset in TEMPORAL_OFFSETS)
    eligible_count = frame_count - 2 * margin
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if eligible_count < sample_count:
        raise ValueError(
            f"cannot select {sample_count} unique anchors from {frame_count} frames "
            f"with a {margin}-frame margin"
        )
    positions = np.linspace(0, eligible_count - 1, num=sample_count)
    anchors = margin + np.floor(positions + 0.5).astype(np.int64)
    if np.unique(anchors).size != sample_count:
        raise AssertionError(f"anchor selection was not unique: {anchors.tolist()}")
    return anchors


def _require_array(
    path: Path,
    shape: Tuple[int, ...],
    dtype: Any,
    *,
    mmap_mode: Optional[str] = "r",
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
    expected_dtype = np.dtype(dtype)
    if tuple(array.shape) != shape or array.dtype != expected_dtype:
        raise ValueError(
            f"{path} must have shape {shape} and dtype {expected_dtype}; "
            f"got {array.shape} and {array.dtype}"
        )
    return array


def _open_depth(view_dir: Path, layout: str, frame_count: int) -> Any:
    expected_shape = (frame_count, SOURCE_HEIGHT, SOURCE_WIDTH)
    if layout != "long":
        return _require_array(view_dir / "depth.npy", expected_shape, np.float32)

    depth_path = view_dir / "depth.zarr"
    if not depth_path.is_dir():
        raise FileNotFoundError(depth_path)
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("zarr is required to audit long-form depth.zarr") from exc
    depth = zarr.open(str(depth_path), mode="r")
    if tuple(depth.shape) != expected_shape or np.dtype(depth.dtype) != np.dtype(np.float32):
        raise ValueError(
            f"{depth_path} must have shape {expected_shape} and dtype float32; "
            f"got {getattr(depth, 'shape', None)} and {getattr(depth, 'dtype', None)}"
        )
    return depth


def project_visible_tracks(
    tracks_world: np.ndarray,
    extrinsics_w2c: np.ndarray,
    intrinsics: np.ndarray,
    visibility: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project finite, positive-Z, source-visible tracks into the native raster."""
    world = np.asarray(tracks_world, dtype=np.float64)
    extrinsics = np.asarray(extrinsics_w2c, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    source_visible = np.asarray(visibility, dtype=bool)
    if world.shape != (POINT_COUNT, 3):
        raise ValueError(f"tracks must have shape {(POINT_COUNT, 3)}, got {world.shape}")
    if extrinsics.shape != (4, 4):
        raise ValueError(f"extrinsics must have shape (4, 4), got {extrinsics.shape}")
    if intrinsics.shape != (4,):
        raise ValueError(f"intrinsics must have shape (4,), got {intrinsics.shape}")
    if source_visible.shape != (POINT_COUNT,):
        raise ValueError(f"visibility must have shape {(POINT_COUNT,)}, got {source_visible.shape}")

    world_h = np.concatenate((world, np.ones((POINT_COUNT, 1), dtype=np.float64)), axis=1)
    camera = np.einsum("ij,pj->pi", extrinsics[:3, :4], world_h, optimize=True)
    camera_z = camera[:, 2]
    fx, fy, cx, cy = intrinsics
    with np.errstate(divide="ignore", invalid="ignore"):
        x = fx * camera[:, 0] / camera_z + cx
        y = fy * camera[:, 1] / camera_z + cy

    valid = (
        source_visible
        & np.isfinite(world).all(axis=1)
        & np.isfinite(camera).all(axis=1)
        & np.isfinite(x)
        & np.isfinite(y)
        & (camera_z > 0.0)
        & (x >= -0.5)
        & (x < SOURCE_WIDTH - 0.5)
        & (y >= -0.5)
        & (y < SOURCE_HEIGHT - 0.5)
    )
    return x[valid], y[valid], camera_z[valid]


def sample_depth_residuals_aligned(
    depth: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    camera_z: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return candidate-aligned signed residuals, using NaN for invalid depth."""
    depth_map = np.asarray(depth)
    if depth_map.shape != (SOURCE_HEIGHT, SOURCE_WIDTH) or depth_map.dtype != np.float32:
        raise ValueError(
            f"depth frame must have shape {(SOURCE_HEIGHT, SOURCE_WIDTH)} and dtype float32; "
            f"got {depth_map.shape} and {depth_map.dtype}"
        )
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    camera_z = np.asarray(camera_z, dtype=np.float64)
    if x.shape != y.shape or x.shape != camera_z.shape or x.ndim != 1:
        raise ValueError("x, y, and camera_z must be same-length one-dimensional arrays")

    center_x = np.floor(x + 0.5).astype(np.int64)
    center_y = np.floor(y + 0.5).astype(np.int64)
    center_inside = (
        (center_x >= 0)
        & (center_x < SOURCE_WIDTH)
        & (center_y >= 0)
        & (center_y < SOURCE_HEIGHT)
    )
    nearest_depth = np.full(x.shape, np.nan, dtype=np.float64)
    nearest_depth[center_inside] = depth_map[center_y[center_inside], center_x[center_inside]]
    nearest_valid = center_inside & np.isfinite(nearest_depth) & (nearest_depth > 0.0)
    nearest = np.full(x.shape, np.nan, dtype=np.float64)
    nearest[nearest_valid] = nearest_depth[nearest_valid] - camera_z[nearest_valid]

    bilinear_x = np.clip(x, 0.0, SOURCE_WIDTH - 1.0)
    bilinear_y = np.clip(y, 0.0, SOURCE_HEIGHT - 1.0)
    x0 = np.floor(bilinear_x).astype(np.int64)
    y0 = np.floor(bilinear_y).astype(np.int64)
    x1 = np.minimum(x0 + 1, SOURCE_WIDTH - 1)
    y1 = np.minimum(y0 + 1, SOURCE_HEIGHT - 1)
    bilinear_inside = (
        (x0 >= 0) & (y0 >= 0) & (x0 < SOURCE_WIDTH) & (y0 < SOURCE_HEIGHT)
    )
    bilinear_depth = np.full(x.shape, np.nan, dtype=np.float64)
    if bilinear_inside.any():
        ids = np.flatnonzero(bilinear_inside)
        d00 = depth_map[y0[ids], x0[ids]].astype(np.float64)
        d10 = depth_map[y0[ids], x1[ids]].astype(np.float64)
        d01 = depth_map[y1[ids], x0[ids]].astype(np.float64)
        d11 = depth_map[y1[ids], x1[ids]].astype(np.float64)
        neighbours = np.stack((d00, d10, d01, d11), axis=1)
        wx = bilinear_x[ids] - x0[ids]
        wy = bilinear_y[ids] - y0[ids]
        weights = np.stack(
            (
                (1.0 - wx) * (1.0 - wy),
                wx * (1.0 - wy),
                (1.0 - wx) * wy,
                wx * wy,
            ),
            axis=1,
        )
        required = weights > 0.0
        positive_finite = np.isfinite(neighbours) & (neighbours > 0.0)
        valid_neighbours = ((~required) | positive_finite).all(axis=1)
        valid_ids = ids[valid_neighbours]
        if valid_ids.size:
            selected_neighbours = neighbours[valid_neighbours]
            selected_weights = weights[valid_neighbours]
            bilinear_depth[valid_ids] = np.sum(
                np.where(selected_weights > 0.0, selected_neighbours, 0.0)
                * selected_weights,
                axis=1,
            )
    bilinear_valid = np.isfinite(bilinear_depth) & (bilinear_depth > 0.0)
    bilinear = np.full(x.shape, np.nan, dtype=np.float64)
    bilinear[bilinear_valid] = bilinear_depth[bilinear_valid] - camera_z[bilinear_valid]

    best_signed = np.full(x.shape, np.nan, dtype=np.float64)
    best_absolute = np.full(x.shape, np.inf, dtype=np.float64)
    for offset_y in (-1, 0, 1):
        for offset_x in (-1, 0, 1):
            sample_x = center_x + offset_x
            sample_y = center_y + offset_y
            inside = (
                (sample_x >= 0)
                & (sample_x < SOURCE_WIDTH)
                & (sample_y >= 0)
                & (sample_y < SOURCE_HEIGHT)
            )
            if not inside.any():
                continue
            ids = np.flatnonzero(inside)
            values = depth_map[sample_y[ids], sample_x[ids]].astype(np.float64)
            valid_values = np.isfinite(values) & (values > 0.0)
            ids = ids[valid_values]
            if not ids.size:
                continue
            signed = values[valid_values] - camera_z[ids]
            absolute = np.abs(signed)
            better = absolute < best_absolute[ids]
            better_ids = ids[better]
            best_absolute[better_ids] = absolute[better]
            best_signed[better_ids] = signed[better]
    return {
        "nearest_pixel": nearest.astype(np.float32, copy=False),
        "bilinear": bilinear.astype(np.float32, copy=False),
        "neighborhood_3x3": best_signed.astype(np.float32, copy=False),
    }


def sample_depth_residuals(
    depth: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    camera_z: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Return compact finite residuals for callers that do not need identities."""
    aligned = sample_depth_residuals_aligned(depth, x, y, camera_z)
    return {mode: values[np.isfinite(values)] for mode, values in aligned.items()}


def paired_temporal_residuals(
    residuals_by_offset: Mapping[int, Mapping[str, np.ndarray]],
) -> Dict[str, Dict[int, np.ndarray]]:
    """Restrict every offset to candidate identities valid at all five offsets."""
    if set(residuals_by_offset) != set(TEMPORAL_OFFSETS):
        raise ValueError(f"expected offsets {TEMPORAL_OFFSETS}, got {sorted(residuals_by_offset)}")
    paired: Dict[str, Dict[int, np.ndarray]] = {}
    for mode in SAMPLING_MODES:
        per_offset = {
            offset: np.asarray(residuals_by_offset[offset][mode], dtype=np.float32)
            for offset in TEMPORAL_OFFSETS
        }
        shapes = {values.shape for values in per_offset.values()}
        if len(shapes) != 1:
            raise ValueError(f"unaligned candidate shapes for {mode}: {sorted(shapes)}")
        valid_at_every_offset = np.logical_and.reduce(
            [np.isfinite(per_offset[offset]) for offset in TEMPORAL_OFFSETS]
        )
        paired[mode] = {
            offset: per_offset[offset][valid_at_every_offset] for offset in TEMPORAL_OFFSETS
        }
    return paired


def _json_float(value: float) -> Optional[float]:
    value = float(value)
    return value if math.isfinite(value) else None


def summarize_residuals(values: np.ndarray, candidate_count: int) -> Dict[str, Any]:
    signed = np.asarray(values, dtype=np.float64)
    valid_count = int(signed.size)
    summary: Dict[str, Any] = {
        "candidate_count": int(candidate_count),
        "valid_positive_depth_count": valid_count,
        "invalid_or_missing_depth_count": int(candidate_count) - valid_count,
        "valid_positive_depth_fraction": (
            _json_float(valid_count / candidate_count) if candidate_count else None
        ),
    }
    if not valid_count:
        summary["absolute_residual_metres"] = None
        summary["signed_depth_minus_camera_z_metres"] = None
        return summary

    absolute = np.abs(signed)
    absolute_percentiles = np.percentile(absolute, [50.0, 90.0, 95.0, 99.0, 99.9])
    signed_percentiles = np.percentile(signed, [1.0, 50.0, 99.0])
    summary["absolute_residual_metres"] = {
        "mean": _json_float(absolute.mean()),
        "rmse": _json_float(np.sqrt(np.mean(np.square(signed)))),
        "p50": _json_float(absolute_percentiles[0]),
        "p90": _json_float(absolute_percentiles[1]),
        "p95": _json_float(absolute_percentiles[2]),
        "p99": _json_float(absolute_percentiles[3]),
        "p99_9": _json_float(absolute_percentiles[4]),
        "maximum": _json_float(absolute.max()),
        "fraction_at_or_below_raycast_visibility_reference_0_05m": _json_float(
            np.mean(absolute <= RAYCAST_VISIBILITY_REFERENCE_METRES)
        ),
    }
    summary["signed_depth_minus_camera_z_metres"] = {
        "mean": _json_float(signed.mean()),
        "p01": _json_float(signed_percentiles[0]),
        "p50": _json_float(signed_percentiles[1]),
        "p99": _json_float(signed_percentiles[2]),
    }
    return summary


def _merge_buckets(buckets: Iterable[ResidualBucket]) -> Tuple[np.ndarray, int]:
    parts: List[np.ndarray] = []
    candidate_count = 0
    for bucket in buckets:
        candidate_count += bucket.candidate_count
        parts.extend(bucket.signed_residual_parts)
    if not parts:
        return np.empty((0,), dtype=np.float32), candidate_count
    return np.concatenate(parts), candidate_count


def _bucket_summary(buckets: Iterable[ResidualBucket]) -> Dict[str, Any]:
    values, candidates = _merge_buckets(buckets)
    return summarize_residuals(values, candidates)


def _matching_buckets(
    buckets: Mapping[BucketKey, ResidualBucket],
    *,
    layouts: Sequence[str],
    offset: int,
    mode: str,
    sequence: Optional[str] = None,
    view: Optional[int] = None,
) -> Iterable[ResidualBucket]:
    layout_set = set(layouts)
    for key, bucket in buckets.items():
        if key.layout not in layout_set or key.temporal_offset != offset or key.sampling_mode != mode:
            continue
        if sequence is not None and key.sequence != sequence:
            continue
        if view is not None and key.view != view:
            continue
        yield bucket


def _layout_summaries(buckets: Mapping[BucketKey, ResidualBucket]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for layout in LAYOUT_ORDER:
        result[layout] = {}
        for mode in SAMPLING_MODES:
            result[layout][mode] = {}
            for offset in TEMPORAL_OFFSETS:
                result[layout][mode][str(offset)] = _bucket_summary(
                    _matching_buckets(buckets, layouts=(layout,), offset=offset, mode=mode)
                )
    return result


def _sequence_summaries(buckets: Mapping[BucketKey, ResidualBucket]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for layout, sequence in unique_source_keys():
        for view in VIEW_IDS:
            for mode in SAMPLING_MODES:
                for offset in TEMPORAL_OFFSETS:
                    summary = _bucket_summary(
                        _matching_buckets(
                            buckets,
                            layouts=(layout,),
                            sequence=sequence,
                            view=view,
                            offset=offset,
                            mode=mode,
                        )
                    )
                    result.append(
                        {
                            "layout": layout,
                            "sequence": sequence,
                            "view": view,
                            "sampling_mode": mode,
                            "temporal_offset": offset,
                            "summary": summary,
                        }
                    )
    return result


def _temporal_comparison(layout_summaries: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for layout in LAYOUT_ORDER:
        result[layout] = {}
        for mode in SAMPLING_MODES:
            p50_by_offset: Dict[str, Optional[float]] = {}
            p95_by_offset: Dict[str, Optional[float]] = {}
            for offset in TEMPORAL_OFFSETS:
                absolute = layout_summaries[layout][mode][str(offset)]["absolute_residual_metres"]
                p50_by_offset[str(offset)] = None if absolute is None else absolute["p50"]
                p95_by_offset[str(offset)] = None if absolute is None else absolute["p95"]

            def best_offset(values: Mapping[str, Optional[float]]) -> Optional[int]:
                finite = [(int(offset), value) for offset, value in values.items() if value is not None]
                return (
                    min(finite, key=lambda item: (item[1], abs(item[0]), item[0]))[0]
                    if finite
                    else None
                )

            result[layout][mode] = {
                "p50_absolute_metres_by_offset": p50_by_offset,
                "p95_absolute_metres_by_offset": p95_by_offset,
                "best_offset_by_p50": best_offset(p50_by_offset),
                "best_offset_by_p95": best_offset(p95_by_offset),
            }
    return result


def _known_good_reference(buckets: Mapping[BucketKey, ResidualBucket]) -> Dict[str, Any]:
    result = {
        "layouts": ["raw", "long"],
        "temporal_offset": 0,
        "note": "Short-form residuals are intentionally excluded from tolerance evidence.",
        "sampling_modes": {},
    }
    for mode in SAMPLING_MODES:
        result["sampling_modes"][mode] = _bucket_summary(
            _matching_buckets(buckets, layouts=("raw", "long"), offset=0, mode=mode)
        )
    return result


def audit(source_root: Path, output_path: Path) -> None:
    source_root = Path(source_root).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    if output_path == source_root or source_root in output_path.parents:
        raise ValueError(
            f"report must be outside the read-only extracted source tree: {output_path}"
        )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    buckets: Dict[BucketKey, ResidualBucket] = {}
    paired_buckets: Dict[BucketKey, ResidualBucket] = {}
    sampled_sources: List[Dict[str, Any]] = []
    start_time = time.monotonic()

    for layout, sequence in unique_source_keys():
        frame_count = SOURCE_FRAME_COUNTS[layout]
        anchors = sample_anchor_frames(frame_count, SAMPLE_COUNTS[layout])
        source_dir = source_root / SOURCE_SUBROOTS[layout] / sequence
        if not source_dir.is_dir():
            raise FileNotFoundError(source_dir)
        tracks = _require_array(
            source_dir / "tracks_xyz.npy",
            (frame_count, POINT_COUNT, 3),
            np.float32,
        )
        sampled_sources.append(
            {"layout": layout, "sequence": sequence, "anchor_frames": anchors.tolist()}
        )
        print(
            f"POINTODYSSEY_GEOMETRY_AUDIT_SOURCE layout={layout} sequence={sequence} "
            f"anchors={','.join(str(int(frame)) for frame in anchors)}",
            flush=True,
        )

        for view in VIEW_IDS:
            view_dir = source_dir / str(view)
            if not view_dir.is_dir():
                raise FileNotFoundError(view_dir)
            intrinsics = _require_array(
                view_dir / "intrinsics.npy", (4,), np.float32, mmap_mode=None
            )
            extrinsics = _require_array(
                view_dir / "extrinsics_w2c.npy", (frame_count, 4, 4), np.float32
            )
            visibility = _require_array(
                view_dir / "visibility.npy", (frame_count, POINT_COUNT), np.bool_
            )
            depth = _open_depth(view_dir, layout, frame_count)

            for anchor in anchors:
                x, y, camera_z = project_visible_tracks(
                    tracks[int(anchor)],
                    extrinsics[int(anchor)],
                    intrinsics,
                    visibility[int(anchor)],
                )
                candidate_count = int(camera_z.size)
                residuals_by_offset: Dict[int, Dict[str, np.ndarray]] = {}
                for temporal_offset in TEMPORAL_OFFSETS:
                    depth_frame = np.asarray(depth[int(anchor) + temporal_offset])
                    aligned = sample_depth_residuals_aligned(depth_frame, x, y, camera_z)
                    residuals_by_offset[temporal_offset] = aligned
                    for mode, residuals in aligned.items():
                        key = BucketKey(layout, sequence, view, temporal_offset, mode)
                        buckets.setdefault(key, ResidualBucket()).add(
                            residuals[np.isfinite(residuals)], candidate_count
                        )
                paired = paired_temporal_residuals(residuals_by_offset)
                for mode, per_offset in paired.items():
                    for temporal_offset, residuals in per_offset.items():
                        key = BucketKey(layout, sequence, view, temporal_offset, mode)
                        paired_buckets.setdefault(key, ResidualBucket()).add(
                            residuals, candidate_count
                        )

    layout_summaries = _layout_summaries(buckets)
    paired_layout_summaries = _layout_summaries(paired_buckets)
    report = {
        "schema_version": SCHEMA_VERSION,
        "format": "pointodyssey_native_depth_track_geometry_audit",
        "status": "completed",
        "source_root": str(source_root),
        "output_path": str(output_path),
        "read_only_source_audit": True,
        "threshold_selected": False,
        "mismatch_classifier_implemented": False,
        "raycast_visibility_reference_metres": RAYCAST_VISIBILITY_REFERENCE_METRES,
        "raycast_visibility_reference_is_not_a_depth_tolerance": True,
        "configuration": {
            "native_resolution": [SOURCE_HEIGHT, SOURCE_WIDTH],
            "sample_counts_per_sequence": SAMPLE_COUNTS,
            "temporal_offsets": list(TEMPORAL_OFFSETS),
            "temporal_offset_definition": (
                "offset d compares depth[t+d] at the pixel and camera-Z projected from "
                "tracks/extrinsics/visibility[t]; +1 means depth[t+1] versus geometry[t]"
            ),
            "temporal_scope": (
                "tests depth-stream indexing relative to the combined geometry stream; "
                "it does not independently shift tracks, cameras, or visibility"
            ),
            "temporal_comparison_cohort": (
                "paired intersection of the same candidate track identities with finite "
                "positive sampled depth at every offset, separately per sampling mode"
            ),
            "sampling_modes": list(SAMPLING_MODES),
            "nearest_pixel_rule": "floor(coordinate + 0.5)",
            "bilinear_rule": (
                "projected coordinates are edge-clamped to the raster; only finite positive "
                "neighbours with nonzero interpolation weight are required"
            ),
            "neighborhood_3x3_rule": (
                "minimum absolute residual among finite positive depths around nearest pixel"
            ),
            "track_filter": (
                "source-visible, finite world/camera/projection, positive camera-Z, "
                "projected inside native pixel-center raster bounds"
            ),
        },
        "sampled_sources": sampled_sources,
        "layout_summaries": layout_summaries,
        "paired_temporal_layout_summaries": paired_layout_summaries,
        "paired_temporal_sequence_view_summaries": _sequence_summaries(paired_buckets),
        "temporal_offset_comparison": _temporal_comparison(paired_layout_summaries),
        "known_good_raw_long_reference": _known_good_reference(buckets),
        "sequence_view_summaries": _sequence_summaries(buckets),
        "elapsed_seconds": _json_float(time.monotonic() - start_time),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "notes": [
            "Residuals are sampled_depth minus projected track camera-Z in metres.",
            (
                "The 0.05 m exporter value is a world-space raycast endpoint visibility "
                "tolerance, reported only as a reference statistic; it is not a dense-depth "
                "mismatch threshold."
            ),
            "No tolerance is selected from short-form failures.",
            "No preprocessing data or failure-ignore policy is written by this audit.",
        ],
    }

    temporary_fd, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=str(output_path.parent)
    )
    try:
        with os.fdopen(temporary_fd, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    print(
        f"POINTODYSSEY_GEOMETRY_AUDIT_DONE output={output_path} "
        f"elapsed_seconds={report['elapsed_seconds']:.3f}",
        flush=True,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    audit(args.source_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
