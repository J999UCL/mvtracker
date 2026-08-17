"""High-throughput loader for the raw multi-view TAPVid-3D contract."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torchvision
from torch.nn import functional as F
from torchvision.io import ImageReadMode, decode_jpeg
from torchvision.transforms import functional as TF

from mvtracker.datasets.estimated_depth import ESTIMATED_DEPTH_TYPE_PROBABILITIES
from mvtracker.datasets.kubric_multiview_dataset import (
    KubricMultiViewDataset,
    _legal_contiguous_window_starts,
)
from mvtracker.datasets.utils import (
    Datapoint,
    SampleRequest,
    aug_depth,
)


_DATASET_PREFIX = "tapvid3d-multiview-"
_SPLITS = {"training": "train", "validation": "validation", "test": "test"}
_CACHE_FORMAT = "tapvid3d_mvtracker_jpeg_index"
_CACHE_VERSION = 1


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _numeric_view_dirs(sequence_root: Path) -> list[Path]:
    views = sorted(
        (path for path in sequence_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    if [int(path.name) for path in views] != list(range(len(views))):
        raise ValueError(f"{sequence_root}: view directories must be contiguous from 0")
    if not views:
        raise ValueError(f"{sequence_root}: no numeric view directories")
    return views


def _read_jpeg_objects(path: Path, frame_count: int) -> list[np.ndarray]:
    images = np.load(path, allow_pickle=True)
    if images.shape != (frame_count,) or images.dtype != np.dtype(object):
        raise ValueError(f"{path}: expected object array with shape {(frame_count,)}")
    result = []
    for frame, image in enumerate(images):
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 1:
            raise ValueError(f"{path}: frame {frame} is not a one-dimensional uint8 JPEG")
        result.append(image)
    return result


def _jpeg_size(encoded: np.ndarray) -> tuple[int, int]:
    import cv2

    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("JPEG decode failed")
    return int(image.shape[0]), int(image.shape[1])


def _cache_files_complete(target: Path, frame_count: int, views: Sequence[int]) -> bool:
    for view in views:
        view_root = target / f"view_{view}"
        byte_path = view_root / "jpeg_bytes.bin"
        offset_path = view_root / "jpeg_offsets.npy"
        if not byte_path.is_file() or not offset_path.is_file():
            return False
        offsets = np.load(offset_path, mmap_mode="r", allow_pickle=False)
        if (
            offsets.shape != (frame_count + 1,)
            or offsets.dtype != np.int64
            or offsets[0] != 0
            or np.any(offsets[1:] <= offsets[:-1])
            or int(offsets[-1]) != byte_path.stat().st_size
        ):
            return False
    return True


def prepare_tapvid3d_cache(
    raw_root: Path,
    cache_root: Path,
    *,
    workers: int = 1,
) -> dict[str, int]:
    """Index raw JPEG object arrays into seekable byte stores.

    Numeric arrays stay in the canonical raw dataset and are memory-mapped by the
    loader. Existing complete cache entries are retained.
    """

    raw_root = raw_root.resolve()
    cache_root = cache_root.resolve()
    if workers < 1:
        raise ValueError("workers must be at least one")
    counts = {"prepared": 0, "reused": 0}
    for split in _SPLITS.values():
        split_root = raw_root / split
        if not split_root.is_dir():
            continue
        for source in sorted(path for path in split_root.iterdir() if path.is_dir()):
            tracks_path = source / "tracks_xyz.npy"
            queries_path = source / "queries_xytv.npy"
            tracks = np.load(tracks_path, mmap_mode="r", allow_pickle=False)
            queries = np.load(queries_path, mmap_mode="r", allow_pickle=False)
            if tracks.ndim != 3 or tracks.shape[2] != 3 or tracks.dtype != np.float32:
                raise ValueError(f"{tracks_path}: expected (F, P, 3) float32")
            frame_count, point_count, _ = tracks.shape
            if queries.shape != (point_count, 4) or queries.dtype != np.float32:
                raise ValueError(f"{queries_path}: expected {(point_count, 4)} float32")

            view_roots = _numeric_view_dirs(source)
            resolution = None
            for view_root in view_roots:
                view = int(view_root.name)
                expected = {
                    "images_jpeg_bytes.npy": ((frame_count,), np.dtype(object)),
                    "intrinsics.npy": ((4,), np.dtype(np.float32)),
                    "extrinsics_w2c.npy": ((frame_count, 4, 4), np.dtype(np.float32)),
                    "visibility.npy": ((frame_count, point_count), np.dtype(np.bool_)),
                }
                for name, (shape, dtype) in expected.items():
                    path = view_root / name
                    array = np.load(path, mmap_mode=None if dtype == np.dtype(object) else "r", allow_pickle=dtype == np.dtype(object))
                    if array.shape != shape or array.dtype != dtype:
                        raise ValueError(f"{path}: expected shape {shape} and dtype {dtype}")
                for name, dtype in (("depth.npy", np.float32), ("foreground_mask.npy", np.bool_)):
                    path = view_root / name
                    array = np.load(path, mmap_mode="r", allow_pickle=False)
                    if array.shape[:1] != (frame_count,) or array.ndim != 3 or array.dtype != np.dtype(dtype):
                        raise ValueError(f"{path}: invalid shape or dtype")
                    if resolution is None:
                        resolution = tuple(int(value) for value in array.shape[1:])
                    if tuple(array.shape[1:]) != resolution:
                        raise ValueError(f"{path}: inconsistent image resolution")
            target = cache_root / split / source.name
            manifest_path = target / "manifest.json"
            if manifest_path.is_file():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if _cache_files_complete(
                    target,
                    frame_count,
                    [int(path.name) for path in view_roots],
                ):
                    counts["reused"] += 1
                    continue

            staging = target.with_name(f".{target.name}.tmp-{os.getpid()}")
            if staging.exists():
                import shutil

                shutil.rmtree(staging)
            staging.mkdir(parents=True)
            def pack_view(view_root: Path) -> None:
                view = int(view_root.name)
                encoded_frames = _read_jpeg_objects(view_root / "images_jpeg_bytes.npy", frame_count)
                current_resolution = _jpeg_size(encoded_frames[0])
                if current_resolution != resolution:
                    raise ValueError(f"{view_root}: JPEG and depth resolutions differ")
                offsets = np.zeros(frame_count + 1, dtype=np.int64)
                view_target = staging / f"view_{view}"
                view_target.mkdir()
                with (view_target / "jpeg_bytes.bin").open("wb") as handle:
                    for frame, encoded in enumerate(encoded_frames):
                        handle.write(encoded)
                        offsets[frame + 1] = offsets[frame] + encoded.size
                np.save(view_target / "jpeg_offsets.npy", offsets)

            if workers == 1:
                for view_root in view_roots:
                    pack_view(view_root)
            else:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=workers) as executor:
                    list(executor.map(pack_view, view_roots))

            manifest = {
                "format": _CACHE_FORMAT,
                "schema_version": _CACHE_VERSION,
                "source_split": split,
                "source_sequence": source.name,
                "frame_count": frame_count,
                "point_count": point_count,
                "views": [int(path.name) for path in view_roots],
                "resolution_hw": list(resolution),
            }
            _atomic_json(staging / "manifest.json", manifest)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                import shutil

                shutil.rmtree(target)
            os.replace(staging, target)
            counts["prepared"] += 1
    return counts


@dataclass
class EncodedTapVid3DSample:
    jpeg_bytes: tuple[torch.Tensor, ...]
    depth: torch.Tensor | None
    theta: torch.Tensor
    intrs: torch.Tensor
    extrs: torch.Tensor
    trajectory: torch.Tensor
    trajectory_3d: torch.Tensor
    visibility: torch.Tensor
    valid: torch.Tensor
    query_points_3d: torch.Tensor
    seq_name: str
    metadata: dict[str, Any]
    output_size: tuple[int, int]
    apply_rgb_aug: bool
    rgb_augmentation: dict[str, Any] | None
    apply_depth_aug: bool
    augmentation_seed: int
    depth_scale: float
    track_upscaling_factor: float
    max_depth: float
    depth_patch_operations: tuple[tuple[int, ...], ...]
    image_codec: str = "jpeg"
    depth_bytes: tuple[bytes, ...] = ()
    depth_sensor_widths: tuple[float, ...] = ()
    depth_focal_lengths: tuple[float, ...] = ()


@dataclass
class EncodedTapVid3DBatch:
    samples: list[EncodedTapVid3DSample]

    def pin_memory(self):
        for sample in self.samples:
            sample.jpeg_bytes = tuple(
                value.pin_memory() if isinstance(value, torch.Tensor) else value
                for value in sample.jpeg_bytes
            )
            for name in (
                "depth", "theta", "intrs", "extrs", "trajectory",
                "trajectory_3d", "visibility", "valid", "query_points_3d",
            ):
                value = getattr(sample, name)
                if value is not None:
                    setattr(sample, name, value.pin_memory())
        return self


def collate_encoded_tapvid3d(batch):
    return (
        EncodedTapVid3DBatch([sample for sample, gotit in batch if gotit]),
        [gotit for _, gotit in batch],
    )


def _read_encoded_frames(
    descriptor: int,
    offsets: np.ndarray,
    frame_indices: Sequence[int],
    *,
    label: Path,
) -> tuple[torch.Tensor, ...]:
    result = []
    for frame in frame_indices:
        start, end = int(offsets[frame]), int(offsets[frame + 1])
        encoded = os.pread(descriptor, end - start, start)
        if len(encoded) != end - start:
            raise ValueError(f"{label}: truncated JPEG byte store")
        result.append(torch.frombuffer(bytearray(encoded), dtype=torch.uint8))
    return tuple(result)


def _intrinsics_matrix(parameters: np.ndarray) -> np.ndarray:
    fx, fy, cx, cy = parameters
    return np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def _project(tracks: np.ndarray, extrinsics: np.ndarray, intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    homogeneous = np.concatenate([tracks, np.ones_like(tracks[..., :1])], axis=-1)
    camera = np.einsum("vtij,tnj->vtni", extrinsics, homogeneous)
    pixels = np.einsum("vtij,vtnj->vtni", intrinsics, camera)
    with np.errstate(divide="ignore", invalid="ignore"):
        xy = pixels[..., :2] / pixels[..., 2:]
    return np.asarray(xy, dtype=np.float32), np.asarray(camera[..., 2], dtype=np.float32)


def _spatial_transform(
    xy: np.ndarray,
    visibility: np.ndarray,
    intrinsics: np.ndarray,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
    rng: np.random.RandomState,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    views, frames, _, _ = xy.shape
    source_h, source_w = source_size
    output_h, output_w = output_size
    theta = np.zeros((views, frames, 2, 3), dtype=np.float32)
    transformed = xy.copy()
    adjusted_intrinsics = intrinsics.copy()
    adjusted_visibility = visibility.copy()
    for view in range(views):
        if enabled:
            pad_left, pad_right, pad_top, pad_bottom = (
                int(rng.randint(0, 45)) for _ in range(4)
            )
            scale_x = scale_y = float(rng.uniform(0.8, 1.2))
            scale_delta_x = scale_delta_y = 0.0
        else:
            pad_left = pad_right = pad_top = pad_bottom = 0
        frame_transforms: list[tuple[float, float, int, int]] = []
        for frame in range(frames):
            if enabled:
                padded_w = source_w + pad_left + pad_right
                padded_h = source_h + pad_top + pad_bottom
                if frame == 1:
                    scale_delta_x = float(rng.uniform(-0.15, 0.15))
                    scale_delta_y = float(rng.uniform(-0.15, 0.15))
                elif frame > 1:
                    scale_delta_x = scale_delta_x * 0.8 + float(rng.uniform(-0.15, 0.15)) * 0.2
                    scale_delta_y = scale_delta_y * 0.8 + float(rng.uniform(-0.15, 0.15)) * 0.2
                scale_x = float(np.clip(scale_x + scale_delta_x, 0.8, 1.2))
                scale_y = float(np.clip(scale_y + scale_delta_y, 0.8, 1.2))
                shared_scale = (scale_x + scale_y) * 0.5
                scale_x = scale_x * 0.5 + shared_scale * 0.5
                scale_y = scale_y * 0.5 + shared_scale * 0.5
                new_w = max(output_w + 10, int(padded_w * scale_x))
                new_h = max(output_h + 10, int(padded_h * scale_y))
                scale_x = (new_w - 1) / (padded_w - 1)
                scale_y = (new_h - 1) / (padded_h - 1)
            else:
                scale_x = (output_w - 1) / (source_w - 1)
                scale_y = (output_h - 1) / (source_h - 1)
                new_w, new_h = output_w, output_h
            transformed[view, frame, :, 0] = (xy[view, frame, :, 0] + pad_left) * scale_x
            transformed[view, frame, :, 1] = (xy[view, frame, :, 1] + pad_top) * scale_y
            adjusted_intrinsics[view, frame, 0, :] *= scale_x
            adjusted_intrinsics[view, frame, 1, :] *= scale_y
            adjusted_intrinsics[view, frame, 0, 2] += pad_left * scale_x
            adjusted_intrinsics[view, frame, 1, 2] += pad_top * scale_y
            frame_transforms.append((scale_x, scale_y, new_w, new_h))

        visible_at_start = adjusted_visibility[view, 0]
        visible_tracks = transformed[view][:, visible_at_start]
        if visible_tracks.shape[1]:
            center_x = float(visible_tracks[:, 0, 0].mean())
            center_y = float(visible_tracks[:, 0, 1].mean())
        else:
            center_x, center_y = output_w / 2, output_h / 2
        crop_x = int(center_x - output_w // 2)
        crop_y = int(center_y - output_h // 2)
        offset_x = offset_y = 0

        for frame, (scale_x, scale_y, new_w, new_h) in enumerate(frame_transforms):
            if enabled:
                if frame == 1:
                    offset_x = int(rng.randint(-36, 37))
                    offset_y = int(rng.randint(-36, 37))
                elif frame > 1:
                    offset_x = int(offset_x * 0.8 + rng.randint(-36, 37) * 0.2)
                    offset_y = int(offset_y * 0.8 + rng.randint(-36, 37) * 0.2)
                crop_x += offset_x
                crop_y += offset_y
                crop_x = int(np.clip(crop_x, 0, new_w - output_w - 1))
                crop_y = int(np.clip(crop_y, 0, new_h - output_h - 1))
            else:
                crop_x = crop_y = 0

            transformed[view, frame, :, 0] -= crop_x
            transformed[view, frame, :, 1] -= crop_y
            adjusted_intrinsics[view, frame, 0, 2] -= crop_x
            adjusted_intrinsics[view, frame, 1, 2] -= crop_y
            adjusted_visibility[view, frame] &= (
                np.isfinite(transformed[view, frame]).all(axis=-1)
                & (transformed[view, frame, :, 0] >= 0)
                & (transformed[view, frame, :, 0] < output_w)
                & (transformed[view, frame, :, 1] >= 0)
                & (transformed[view, frame, :, 1] < output_h)
            )

            ax = (output_w - 1) / (scale_x * (source_w - 1))
            ay = (output_h - 1) / (scale_y * (source_h - 1))
            bx = ax + 2 * (crop_x / scale_x - pad_left) / (source_w - 1) - 1
            by = ay + 2 * (crop_y / scale_y - pad_top) / (source_h - 1) - 1
            theta[view, frame] = ((ax, 0, bx), (0, ay, by))
    return transformed, adjusted_visibility, adjusted_intrinsics, theta


def _sample_tracks(
    tracks: np.ndarray,
    xy: np.ndarray,
    camera_z: np.ndarray,
    visibility: np.ndarray,
    count: int,
    rng: np.random.RandomState,
    *,
    augment_this_datapoint: bool = False,
    enable_variable_trajpersample_augs: bool = False,
    sample_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    any_view = visibility.any(axis=0)
    eligible = any_view.sum(axis=0) >= 2
    eligible &= any_view[0] | any_view[len(any_view) // 2]
    indices = np.flatnonzero(eligible)
    if len(indices) < max(1, count // 4):
        return np.empty(0, dtype=np.int64), np.empty((0, 4), dtype=np.float32)
    order = rng.permutation(len(indices))
    sample_count = count
    if augment_this_datapoint and enable_variable_trajpersample_augs:
        if sample_index % 20 == 0:
            sample_count //= 8
        elif sample_index % 21 != 0:
            low = max(1, sample_count // 4)
            high = min(len(indices), sample_count) + 1
            sample_count = int(rng.randint(low, high))
    else:
        sample_count = min(len(indices), sample_count)
    selected_array = indices[order[:sample_count]]
    selected_visibility = visibility[:, :, selected_array]
    visible_any = selected_visibility.any(axis=0)
    last_visible = (np.arange(len(visible_any))[:, None] * visible_any).max(axis=0)
    visible_any[last_visible, np.arange(len(selected_array))] = False
    random_query_count = len(selected_array) // 4
    query_times = np.argmax(visible_any, axis=0)
    for point in range(random_query_count):
        candidates = np.flatnonzero(visible_any[:, point])
        query_times[point] = int(candidates[rng.randint(len(candidates))])
    query_points = np.concatenate(
        [query_times[:, None].astype(np.float32), tracks[query_times, selected_array]], axis=1
    )
    return selected_array, query_points.astype(np.float32)


def _visible_path_lengths(tracks: np.ndarray, visibility: np.ndarray) -> np.ndarray:
    visible_consecutively = (visibility[:, :-1] & visibility[:, 1:]).any(axis=0)
    movement = np.linalg.norm(np.diff(tracks, axis=0), axis=-1)
    movement[~visible_consecutively] = 0
    return movement.sum(axis=0)


def _preselect_motion_tracks(
    tracks: np.ndarray,
    visibility: np.ndarray,
    rng: np.random.RandomState,
    *,
    ratio_dynamic: float,
    ratio_very_dynamic: float,
    maximum: int | None,
) -> np.ndarray:
    """Apply MVTracker's full-sequence motion-bucket preselection."""
    movement = _visible_path_lengths(tracks, visibility)
    static = movement < 0.01
    dynamic = movement > 0.1
    very_dynamic = movement > 2.0
    ratio_static = 1.0 - ratio_dynamic - ratio_very_dynamic
    available = len(tracks[0])
    available = min(
        available,
        int(dynamic.sum() / ratio_dynamic) if ratio_dynamic else available,
        int(very_dynamic.sum() // ratio_very_dynamic) if ratio_very_dynamic else available,
        int(static.sum() / ratio_static) if ratio_static else available,
    )
    if maximum is not None:
        available = min(available, maximum)
    n_dynamic = min(int(available * ratio_dynamic), int(dynamic.sum()))
    n_very_dynamic = min(int(available * ratio_very_dynamic), int(very_dynamic.sum()))
    n_static = available - n_dynamic - n_very_dynamic
    selected = np.concatenate(
        [
            rng.choice(np.flatnonzero(dynamic), n_dynamic, replace=False),
            rng.choice(np.flatnonzero(very_dynamic), n_very_dynamic, replace=False),
            rng.choice(np.flatnonzero(static), n_static, replace=False),
        ]
    )
    rng.shuffle(selected)
    return selected.astype(np.int64, copy=False)


def _scene_transform(
    tracks: np.ndarray,
    query_points: np.ndarray,
    trajectory: np.ndarray,
    extrinsics: np.ndarray,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rx, ry = np.deg2rad(rng.uniform(-15, 15, size=2))
    rotation_x = np.asarray(
        [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]],
        dtype=np.float32,
    )
    rotation_y = np.asarray(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]],
        dtype=np.float32,
    )
    rotation = rotation_y @ rotation_x
    scale = float(rng.uniform(0.8, 1.5))
    translation = rng.uniform(-2, 2, size=3).astype(np.float32)
    transformed_tracks = np.einsum("ij,tnj->tni", rotation, tracks * scale) + translation
    transformed_queries = query_points.copy()
    transformed_queries[:, 1:] = (
        np.einsum("ij,nj->ni", rotation, query_points[:, 1:] * scale) + translation
    )
    transformed_trajectory = trajectory.copy()
    transformed_trajectory[..., 2] *= scale
    rigid = np.eye(4, dtype=np.float32)
    rigid[:3, :3] = rotation
    rigid[:3, 3] = translation
    square = np.repeat(np.eye(4, dtype=np.float32)[None, None], extrinsics.shape[0], axis=0)
    square = np.repeat(square, extrinsics.shape[1], axis=1)
    square[..., :3, :3] = extrinsics[..., :3, :3]
    square[..., :3, 3] = extrinsics[..., :3, 3] * scale
    transformed_extrinsics = np.einsum("vtij,jk->vtik", square, np.linalg.inv(rigid))[..., :3, :]
    return (
        transformed_tracks.astype(np.float32),
        transformed_queries.astype(np.float32),
        transformed_trajectory.astype(np.float32),
        transformed_extrinsics.astype(np.float32),
        scale,
    )


def _sample_depth_patch_operations(
    trajectory: np.ndarray,
    visibility: np.ndarray,
    height: int,
    width: int,
    rng: np.random.RandomState,
    *,
    eraser_probability: float,
    eraser_max: int,
    eraser_bounds: Sequence[int],
    replace_probability: float,
    replace_max: int,
    replace_bounds: Sequence[int],
) -> tuple[tuple[tuple[int, ...], ...], np.ndarray]:
    views, frames, _, _ = trajectory.shape
    updated_visibility = visibility.copy()
    operations: list[tuple[int, ...]] = []
    for view in range(views):
        for frame in range(1, frames):
            for kind, probability, maximum, bounds in (
                (0, eraser_probability, eraser_max, eraser_bounds),
                (1, replace_probability, replace_max, replace_bounds),
            ):
                if rng.rand() >= probability:
                    continue
                for _ in range(int(rng.randint(1, maximum + 1))):
                    xc, yc = int(rng.randint(width)), int(rng.randint(height))
                    dx = int(rng.randint(bounds[0], bounds[1]))
                    dy = int(rng.randint(bounds[0], bounds[1]))
                    x0 = int(np.clip(round(xc - dx / 2), 0, width - 1))
                    x1 = int(np.clip(round(xc + dx / 2), x0 + 1, width))
                    y0 = int(np.clip(round(yc - dy / 2), 0, height - 1))
                    y1 = int(np.clip(round(yc + dy / 2), y0 + 1, height))
                    if kind == 0:
                        fill_mode = int(rng.choice(4, p=[0.2, 0.1, 0.35, 0.35]))
                        operations.append((kind, view, frame, x0, x1, y0, y1, fill_mode))
                    else:
                        source_view = int(rng.randint(views))
                        source_frame = int(rng.randint(frames))
                        source_x = int(rng.randint(0, width - (x1 - x0) + 1))
                        source_y = int(rng.randint(0, height - (y1 - y0) + 1))
                        operations.append((
                            kind, view, frame, x0, x1, y0, y1,
                            source_view, source_frame, source_x, source_y,
                        ))
                    point_xy = trajectory[view, frame, :, :2]
                    updated_visibility[view, frame] &= ~(
                        (point_xy[:, 0] >= x0)
                        & (point_xy[:, 0] < x1)
                        & (point_xy[:, 1] >= y0)
                        & (point_xy[:, 1] < y1)
                    )
    return tuple(operations), updated_visibility


def _sample_color_jitter(rng: np.random.RandomState) -> tuple[tuple[int, ...], tuple[float, ...]]:
    return (
        tuple(int(value) for value in rng.permutation(4)),
        (
            float(rng.uniform(0.8, 1.2)),
            float(rng.uniform(0.8, 1.2)),
            float(rng.uniform(0.8, 1.2)),
            float(rng.uniform(-0.25 / np.pi, 0.25 / np.pi)),
        ),
    )


def _sample_rgb_augmentation(
    trajectory: np.ndarray,
    visibility: np.ndarray,
    height: int,
    width: int,
    rng: np.random.RandomState,
    *,
    eraser_probability: float,
    eraser_max: int,
    eraser_bounds: Sequence[int],
    replace_probability: float,
    replace_max: int,
    replace_bounds: Sequence[int],
) -> tuple[dict[str, Any], np.ndarray]:
    views, frames, _, _ = trajectory.shape
    updated_visibility = visibility.copy()
    patch_operations: list[tuple[int, ...]] = []

    def bounds(limit: int, center: int, extent: int) -> tuple[int, int]:
        lower = int(np.clip(round(center - extent / 2), 0, limit - 1))
        upper = int(np.clip(round(center + extent / 2), lower + 1, limit))
        return lower, upper

    for kind, probability, maximum, extents in (
        (0, eraser_probability, eraser_max, eraser_bounds),
        (1, replace_probability, replace_max, replace_bounds),
    ):
        for view in range(views):
            for frame in range(1, frames):
                if rng.rand() >= probability:
                    continue
                for _ in range(int(rng.randint(1, maximum + 1))):
                    x0, x1 = bounds(width, int(rng.randint(width)), int(rng.randint(*extents)))
                    y0, y1 = bounds(height, int(rng.randint(height)), int(rng.randint(*extents)))
                    if kind == 0:
                        patch_operations.append((kind, view, frame, x0, x1, y0, y1))
                    else:
                        source_frame = int(rng.randint(frames))
                        source_x = int(rng.randint(0, width - (x1 - x0) + 1))
                        source_y = int(rng.randint(0, height - (y1 - y0) + 1))
                        patch_operations.append(
                            (kind, view, frame, x0, x1, y0, y1, source_frame, source_x, source_y)
                        )
                    point_xy = trajectory[view, frame, :, :2]
                    updated_visibility[view, frame] &= ~(
                        (point_xy[:, 0] >= x0)
                        & (point_xy[:, 0] < x1)
                        & (point_xy[:, 1] >= y0)
                        & (point_xy[:, 1] < y1)
                    )

    alternative_jitters = tuple(
        tuple((_sample_color_jitter(rng), _sample_color_jitter(rng)) for _ in range(frames))
        for _ in range(views)
    )
    color_jitters = (
        tuple(_sample_color_jitter(rng) for _ in range(frames))
        if rng.rand() < 0.25
        else ()
    )
    blur_sigmas = (
        tuple(float(rng.uniform(0.1, 2.0)) for _ in range(frames))
        if rng.rand() < 0.25
        else ()
    )
    return {
        "patch_operations": tuple(patch_operations),
        "alternative_jitters": alternative_jitters,
        "color_jitters": color_jitters,
        "blur_sigmas": blur_sigmas,
    }, updated_visibility


def _apply_color_jitter(
    image: torch.Tensor,
    parameters: tuple[tuple[int, ...], tuple[float, ...]],
) -> torch.Tensor:
    order, factors = parameters
    result = image / 255.0
    operations = (
        lambda value: TF.adjust_brightness(value, factors[0]),
        lambda value: TF.adjust_contrast(value, factors[1]),
        lambda value: TF.adjust_saturation(value, factors[2]),
        lambda value: TF.adjust_hue(value, factors[3]),
    )
    for operation in order:
        result = operations[operation](result)
    return result.clamp_(0, 1).mul_(255.0)


def _apply_rgb_augmentation(video: torch.Tensor, specification: dict[str, Any]) -> torch.Tensor:
    result = video.clone()
    for operation in specification["patch_operations"]:
        kind, view, frame, x0, x1, y0, y1, *parameters = operation
        if kind == 0:
            patch = result[view, frame, :, y0:y1, x0:x1]
            patch.copy_(patch.mean(dim=(1, 2), keepdim=True))

    alternative = result.clone()
    for view, frames in enumerate(specification["alternative_jitters"]):
        for frame, jitters in enumerate(frames):
            for jitter in jitters:
                alternative[view, frame] = _apply_color_jitter(alternative[view, frame], jitter)

    for operation in specification["patch_operations"]:
        kind, view, frame, x0, x1, y0, y1, *parameters = operation
        if kind == 1:
            source_frame, source_x, source_y = parameters
            alternative_patch = alternative[
                view,
                source_frame,
                :,
                source_y : source_y + (y1 - y0),
                source_x : source_x + (x1 - x0),
            ]
            result[view, frame, :, y0:y1, x0:x1] = alternative_patch

    for frame, jitter in enumerate(specification["color_jitters"]):
        for view in range(len(result)):
            result[view, frame] = _apply_color_jitter(result[view, frame], jitter)
    for frame, sigma in enumerate(specification["blur_sigmas"]):
        for view in range(len(result)):
            result[view, frame] = TF.gaussian_blur(
                result[view, frame], [11, 11], [sigma, sigma]
            )
    return result


class TapVid3DMultiViewDataset(KubricMultiViewDataset):
    """MVTracker sampling over cached raw TAPVid-3D sequences."""

    collate_fn = staticmethod(collate_encoded_tapvid3d)
    requires_cuda_prefetch = True

    def __init__(
        self,
        *args,
        raw_root: str,
        view_count_probabilities: Sequence[float] | None = None,
        **kwargs,
    ):
        self.raw_root = Path(raw_root)
        self.view_count_probabilities = tuple(view_count_probabilities or (0.25,) * 4)
        super().__init__(*args, **kwargs)
        self._manifests: dict[str, dict[str, Any]] = {}
        self._arrays: dict[Path, np.ndarray] = {}
        self._jpeg_descriptors: dict[Path, int] = {}
        for sequence in self.seq_names:
            self._manifests[sequence] = self._load_manifest(sequence)

    def _load_manifest(self, sequence: str) -> dict[str, Any]:
        cache_root = Path(self.data_root) / sequence
        manifest = json.loads((cache_root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != _CACHE_FORMAT or manifest.get("schema_version") != _CACHE_VERSION:
            raise ValueError(f"{cache_root}: unsupported or missing cache manifest")
        if not _cache_files_complete(
            cache_root,
            int(manifest["frame_count"]),
            manifest["views"],
        ):
            raise ValueError(f"{cache_root}: cache is incomplete; rerun TAPVid-3D preparation")
        return manifest

    def _manifest(self, sequence: str) -> dict[str, Any]:
        manifests = getattr(self, "_manifests", None)
        if manifests is None:
            self._manifests = {}
            manifests = self._manifests
        if sequence not in manifests:
            manifests[sequence] = self._load_manifest(sequence)
        return manifests[sequence]

    def _mmap(self, path: Path) -> np.ndarray:
        arrays = getattr(self, "_arrays", None)
        if arrays is None:
            self._arrays = {}
            arrays = self._arrays
        if path not in arrays:
            arrays[path] = np.load(path, mmap_mode="r", allow_pickle=False)
        return arrays[path]

    def _jpeg_descriptor(self, path: Path) -> int:
        descriptors = getattr(self, "_jpeg_descriptors", None)
        if descriptors is None:
            self._jpeg_descriptors = {}
            descriptors = self._jpeg_descriptors
        if path not in descriptors:
            descriptors[path] = os.open(path, os.O_RDONLY)
        return descriptors[path]

    @staticmethod
    def from_name(
        dataset_name,
        dataset_root,
        training_args=None,
        fabric=None,
        just_return_kwargs=False,
        include_scene_ids=None,
        exclude_scene_ids=(),
    ):
        if not dataset_name.startswith(_DATASET_PREFIX):
            raise ValueError(f"Unsupported TAPVid-3D dataset name: {dataset_name}")
        requested = dataset_name[len(_DATASET_PREFIX):]
        if requested not in _SPLITS:
            raise ValueError(f"Unsupported TAPVid-3D split: {requested}")
        if requested == "training":
            kwargs = KubricMultiViewDataset.from_name(
                "kubric-multiview-v3-training", dataset_root,
                training_args=training_args, fabric=fabric, just_return_kwargs=True,
                include_scene_ids=include_scene_ids,
                exclude_scene_ids=exclude_scene_ids,
            )
        else:
            kwargs = KubricMultiViewDataset.from_name(
                "kubric-multiview-v3", dataset_root,
                training_args=training_args, just_return_kwargs=True,
                include_scene_ids=include_scene_ids,
                exclude_scene_ids=exclude_scene_ids,
            )
        datasets_cfg = getattr(training_args, "datasets", {}) if training_args is not None else {}
        raw_dir = datasets_cfg.get("tapvid3d_raw_dir", "TAPVid3D_raw")
        cache_dir = datasets_cfg.get("tapvid3d_cache_dir", "TAPVid3D_MVTracker_cache")
        kwargs.update({
            "data_root": os.path.join(dataset_root, cache_dir, _SPLITS[requested]),
            "raw_root": os.path.join(dataset_root, raw_dir, _SPLITS[requested]),
            "num_views": int(datasets_cfg.get("tapvid3d_num_views", 4)),
            "view_count_probabilities": datasets_cfg.get(
                "tapvid3d_view_count_probabilities", (0.25,) * 4
            ),
            "views_to_return": None,
            "novel_views": None,
            "use_duster_depths": False,
            "clean_duster_depths": False,
            "duster_views": None,
            "supported_duster_views_sets": None,
            # This dataset has its own cache manifest. Never attach the native
            # MV-Kubric metadata index inherited from the shared factory.
            "metadata_index_root": None,
            "enable_variable_depth_type_augs": bool(
                requested == "training"
                and kwargs.get("enable_variable_depth_type_augs")
                and datasets_cfg.get("estimated_depth_root")
            ),
        })
        estimated_depth_root = datasets_cfg.get("estimated_depth_root")
        if estimated_depth_root is not None and not os.path.isabs(estimated_depth_root):
            estimated_depth_root = os.path.join(dataset_root, estimated_depth_root)
        kwargs["estimated_depth_root"] = estimated_depth_root
        kwargs["estimated_depth_provider"] = datasets_cfg.get("estimated_depth_provider")
        if requested == "training" and kwargs.get("enable_variable_num_views_augs"):
            kwargs["num_views"] = None
        if kwargs.get("normalize_scene_following_vggt"):
            raise ValueError("GPU TAPVid-3D loading does not support VGGT scene normalization")
        if just_return_kwargs:
            return kwargs
        return TapVid3DMultiViewDataset(**kwargs)

    def __getitem__(self, index):
        load_started = time.perf_counter()
        request = index if hasattr(index, "virtual_index") else None
        virtual_index = request.virtual_index if request is not None else int(index)
        scene_index = (
            request.scene_index
            if request is not None and request.scene_index is not None
            else virtual_index % self.real_len
        )
        if not 0 <= scene_index < self.real_len:
            raise IndexError(f"scene index {scene_index} is outside [0, {self.real_len})")
        sequence = self.seq_names[scene_index]
        if self.seed is None:
            seed = int(torch.randint(0, 2**32 - 1, ()).item())
        else:
            seed = int(self.seed + virtual_index if self.add_index_to_seed else self.seed)
        rng = np.random.RandomState(seed)
        cache_root = Path(self.data_root) / sequence
        manifest = self._manifest(sequence)
        source_root = self.raw_root / manifest["source_sequence"]

        frame_count = int(manifest["frame_count"])
        available_views = list(manifest["views"])
        if getattr(self, "enable_variable_num_views_augs", False):
            maximum_views = min(4, len(available_views))
            if request is None or request.view_count is None:
                probabilities = np.asarray(
                    getattr(self, "view_count_probabilities", (0.25,) * 4)[:maximum_views],
                    dtype=np.float64,
                )
                probabilities /= probabilities.sum()
                view_count = int(rng.choice(np.arange(1, maximum_views + 1), p=probabilities))
            else:
                view_count = int(request.view_count)
                if not 1 <= view_count <= maximum_views:
                    raise ValueError(
                        f"requested view count {view_count} is outside [1, {maximum_views}]"
                    )
        else:
            view_count = len(available_views) if self.num_views == -1 else int(self.num_views)
            if request is not None and request.view_count is not None and request.view_count != view_count:
                raise ValueError(
                    f"requested view count {request.view_count} does not match fixed count {view_count}"
                )
        if len(available_views) < view_count:
            raise ValueError(f"{source_root}: requires {view_count} views")
        views = sorted(rng.choice(available_views, view_count, replace=False).tolist())

        depth_type = "gt"
        if getattr(self, "enable_variable_depth_type_augs", False):
            depth_type = str(rng.choice(
                tuple(ESTIMATED_DEPTH_TYPE_PROBABILITIES),
                p=tuple(ESTIMATED_DEPTH_TYPE_PROBABILITIES.values()),
            ))

        tracks_all = np.asarray(self._mmap(source_root / "tracks_xyz.npy"), dtype=np.float32)
        visibility_all = np.stack(
            [
                np.asarray(self._mmap(source_root / str(view) / "visibility.npy"), dtype=np.bool_)
                for view in views
            ]
        )
        preselected = _preselect_motion_tracks(
            tracks_all,
            visibility_all,
            rng,
            ratio_dynamic=float(getattr(self, "ratio_dynamic", 0.5)),
            ratio_very_dynamic=float(getattr(self, "ratio_very_dynamic", 0.25)),
            maximum=getattr(self, "max_tracks_to_preload", 18000),
        )
        if not len(preselected):
            return None, False

        legal = _legal_contiguous_window_starts(frame_count, self.seq_len)
        if not len(legal):
            raise ValueError(f"{source_root}: fewer than {self.seq_len} frames")
        start = int(rng.choice(legal))
        frame_indices = np.arange(start, start + self.seq_len)
        tracks = tracks_all[frame_indices][:, preselected]
        extrinsics = []
        intrinsics = []
        visibility = []
        depths = []
        encoded = []
        source_size = tuple(int(value) for value in manifest["resolution_hw"])
        estimated_depths = cleaned_mask = None
        if depth_type != "gt":
            estimated_depths, cleaned_mask = self.estimated_depth_store.load(
                sequence, views, frame_indices
            )
            if estimated_depths.shape != (view_count, self.seq_len, *source_size):
                raise ValueError(
                    f"estimated depth shape {estimated_depths.shape} does not match "
                    f"sample shape {(view_count, self.seq_len, *source_size)}"
                )
            if depth_type == "estimated_cleaned":
                estimated_depths = estimated_depths * cleaned_mask
        for view_position, view in enumerate(views):
            raw_view = source_root / str(view)
            extrinsics.append(np.asarray(self._mmap(raw_view / "extrinsics_w2c.npy")[frame_indices, :3, :4], dtype=np.float32))
            k = _intrinsics_matrix(self._mmap(raw_view / "intrinsics.npy"))
            intrinsics.append(np.repeat(k[None], self.seq_len, axis=0))
            visibility.append(visibility_all[views.index(view), frame_indices][:, preselected])
            depth = (
                self._mmap(raw_view / "depth.npy")[frame_indices]
                if estimated_depths is None
                else estimated_depths[view_position]
            )
            depths.append(torch.from_numpy(np.asarray(depth, dtype=np.float32).copy()))
            cache_view = cache_root / f"view_{view}"
            encoded.extend(_read_encoded_frames(
                self._jpeg_descriptor(cache_view / "jpeg_bytes.bin"),
                self._mmap(cache_view / "jpeg_offsets.npy"),
                frame_indices,
                label=cache_view,
            ))
        extrinsics_np = np.stack(extrinsics)
        intrinsics_np = np.stack(intrinsics)
        visibility_np = np.stack(visibility)
        xy, camera_z = _project(tracks, extrinsics_np, intrinsics_np)
        visibility_np &= np.isfinite(xy).all(axis=-1) & np.isfinite(camera_z) & (camera_z > 0)
        augment_this_datapoint = bool(
            self.augmentation_probability > 0
            and rng.rand() <= self.augmentation_probability
        )
        apply_rgb_aug = bool(
            augment_this_datapoint and getattr(self, "enable_rgb_augs", False)
        )
        rgb_augmentation = None
        if apply_rgb_aug:
            pre_crop_trajectory = np.concatenate([xy, camera_z[..., None]], axis=-1)
            rgb_augmentation, visibility_np = _sample_rgb_augmentation(
                pre_crop_trajectory,
                visibility_np,
                source_size[0],
                source_size[1],
                rng,
                eraser_probability=self.eraser_aug_prob,
                eraser_max=self.eraser_max,
                eraser_bounds=self.eraser_bounds,
                replace_probability=self.replace_aug_prob,
                replace_max=self.replace_max,
                replace_bounds=self.replace_bounds,
            )
        output_size = tuple(int(value) for value in self.crop_size) if self.enable_cropping_augs else source_size
        xy, visibility_np, intrinsics_np, theta = _spatial_transform(
            xy, visibility_np, intrinsics_np, source_size, output_size, rng,
            self.enable_cropping_augs,
        )
        transformed_trajectory = np.concatenate([xy, camera_z[..., None]], axis=-1)
        apply_depth_aug = bool(augment_this_datapoint and self.enable_depth_augs)
        depth_patch_operations: tuple[tuple[int, ...], ...] = ()
        if apply_depth_aug:
            depth_patch_operations, visibility_np = _sample_depth_patch_operations(
                transformed_trajectory,
                visibility_np,
                output_size[0],
                output_size[1],
                rng,
                eraser_probability=self.eraser_aug_prob,
                eraser_max=self.eraser_max,
                eraser_bounds=self.eraser_bounds,
                replace_probability=self.replace_aug_prob,
                replace_max=self.replace_max,
                replace_bounds=self.replace_bounds,
            )
        selected, query_points = _sample_tracks(
            tracks,
            xy,
            camera_z,
            visibility_np,
            self.traj_per_sample,
            rng,
            augment_this_datapoint=augment_this_datapoint,
            enable_variable_trajpersample_augs=getattr(
                self, "enable_variable_trajpersample_augs", False
            ),
            sample_index=virtual_index,
        )
        if not len(selected):
            return None, False
        xy_z = transformed_trajectory[:, :, selected]
        selected_tracks = tracks[:, selected]
        selected_visibility = visibility_np[:, :, selected]
        selected_global = preselected[selected]
        full_movement = _visible_path_lengths(
            tracks_all[:, selected_global],
            visibility_all[:, :, selected_global],
        )
        window_movement = _visible_path_lengths(
            tracks_all[frame_indices][:, selected_global],
            visibility_all[:, frame_indices][:, :, selected_global],
        )
        motion_statistics = {
            "motion_track_count": int(len(selected_global)),
            "motion_full_mean_m": float(full_movement.mean()),
            "motion_full_median_m": float(np.median(full_movement)),
            "motion_full_p90_m": float(np.quantile(full_movement, 0.9)),
            "motion_full_static_count": int((full_movement < 0.01).sum()),
            "motion_full_dynamic_count": int((full_movement > 0.1).sum()),
            "motion_full_very_dynamic_count": int((full_movement > 2.0).sum()),
            "motion_window_mean_m": float(window_movement.mean()),
            "motion_window_median_m": float(np.median(window_movement)),
            "motion_window_p90_m": float(np.quantile(window_movement, 0.9)),
            "motion_window_static_count": int((window_movement < 0.01).sum()),
            "motion_window_dynamic_count": int((window_movement > 0.1).sum()),
            "motion_window_very_dynamic_count": int((window_movement > 2.0).sum()),
            "motion_full_dynamic_window_static_count": int(
                ((full_movement > 0.1) & (window_movement < 0.01)).sum()
            ),
        }
        depth_scale = 1.0
        if getattr(self, "enable_scene_transform_augs", False):
            selected_tracks, query_points, xy_z, extrinsics_np, depth_scale = _scene_transform(
                selected_tracks, query_points, xy_z, extrinsics_np, rng
            )
        if getattr(self, "enable_camera_params_noise_augs", False):
            intrinsics_np = intrinsics_np + rng.normal(0, 0.001, size=intrinsics_np.shape)
            extrinsics_np = extrinsics_np + rng.normal(0, 0.001, size=extrinsics_np.shape)
        sample = EncodedTapVid3DSample(
            jpeg_bytes=tuple(encoded),
            depth=torch.stack(depths)[:, :, None],
            theta=torch.from_numpy(theta),
            intrs=torch.from_numpy(intrinsics_np),
            extrs=torch.from_numpy(extrinsics_np),
            trajectory=torch.from_numpy(xy_z),
            trajectory_3d=torch.from_numpy(np.asarray(selected_tracks, dtype=np.float32)),
            visibility=torch.from_numpy(selected_visibility),
            valid=torch.ones((self.seq_len, len(selected)), dtype=torch.float32),
            query_points_3d=torch.from_numpy(query_points),
            seq_name=sequence,
            metadata={
                "virtual_index": virtual_index,
                "scene_index": scene_index,
                "scene_name": sequence,
                "seed": seed,
                "window_start": start,
                "window_end_exclusive": start + self.seq_len,
                "selected_views": views,
                "depth_source": depth_type,
                "requested_view_count": request.view_count if request is not None else None,
                "gotit": True,
                "worker_prepare_seconds": time.perf_counter() - load_started,
                **motion_statistics,
            },
            output_size=output_size,
            apply_rgb_aug=apply_rgb_aug,
            rgb_augmentation=rgb_augmentation,
            apply_depth_aug=apply_depth_aug,
            augmentation_seed=seed,
            depth_scale=depth_scale,
            track_upscaling_factor=1.0 / depth_scale,
            max_depth=float(getattr(self, "max_depth", 1000.0)),
            depth_patch_operations=depth_patch_operations,
        )
        return sample, True


def decode_tapvid3d_batch(
    batch: EncodedTapVid3DBatch,
    device: torch.device,
    *,
    timing_events: tuple[torch.cuda.Event, torch.cuda.Event, torch.cuda.Event] | None = None,
    nvimagecodec_rgb_decoder=None,
    nvimagecodec_depth_decoder=None,
    rgb_stream: torch.cuda.Stream | None = None,
    depth_stream: torch.cuda.Stream | None = None,
    prepare_stream: torch.cuda.Stream | None = None,
) -> Datapoint:
    if device.type != "cuda":
        raise RuntimeError("TAPVid-3D training requires CUDA nvJPEG decoding")
    version = tuple(int(value) for value in torchvision.__version__.split("+")[0].split(".")[:2])
    if version < (0, 20):
        raise RuntimeError("batched CUDA JPEG decoding requires torchvision 0.20 or newer")
    if timing_events is not None:
        timing_events[0].record()
    codecs = {sample.image_codec for sample in batch.samples}
    if len(codecs) != 1:
        raise ValueError(f"encoded batches cannot mix image codecs: {sorted(codecs)}")
    codec = codecs.pop()
    flat_encoded = [encoded for sample in batch.samples for encoded in sample.jpeg_bytes]
    decoded_depths = None
    if codec == "jpeg":
        if rgb_stream is None or prepare_stream is None:
            raise RuntimeError("GPU image decode requires explicit CUDA streams")
        with torch.cuda.stream(rgb_stream):
            decoded_all = decode_jpeg(flat_encoded, mode=ImageReadMode.RGB, device=device)
        prepare_stream.wait_stream(rgb_stream)
    elif codec == "nvimagecodec":
        if nvimagecodec_rgb_decoder is None or nvimagecodec_depth_decoder is None:
            raise RuntimeError("MV-Kubric GPU decode requires RGB and depth decoders")
        if rgb_stream is None or depth_stream is None or prepare_stream is None:
            raise RuntimeError("MV-Kubric GPU decode requires explicit CUDA streams")
        rgb_images = nvimagecodec_rgb_decoder.decode(
            flat_encoded,
            cuda_stream=rgb_stream.cuda_stream,
        )
        depth_images = nvimagecodec_depth_decoder.decode(
            [encoded for sample in batch.samples for encoded in sample.depth_bytes],
            cuda_stream=depth_stream.cuda_stream,
        )
        decoded_all = [torch.from_dlpack(image.to_dlpack()) for image in rgb_images]
        decoded_depths = [torch.from_dlpack(image.to_dlpack()) for image in depth_images]
        prepare_stream.wait_stream(rgb_stream)
        prepare_stream.wait_stream(depth_stream)
    else:
        raise ValueError(f"unsupported encoded image codec: {codec}")
    if timing_events is not None:
        timing_events[1].record()
    videos = []
    depths = []
    decoded_offset = 0
    depth_offset = 0
    for sample in batch.samples:
        decoded = decoded_all[decoded_offset:decoded_offset + len(sample.jpeg_bytes)]
        decoded_offset += len(sample.jpeg_bytes)
        if sample.depth is not None:
            views, frames = sample.depth.shape[:2]
            source_h, source_w = sample.depth.shape[-2:]
        else:
            views = len(sample.depth_sensor_widths)
            frames = len(sample.depth_bytes) // views
            source_h, source_w = decoded[0].shape[:2]
        output_h, output_w = sample.output_size
        if codec == "jpeg":
            rgb = torch.stack(decoded).float()
        else:
            rgb = torch.stack([
                image[..., :3].permute(2, 0, 1) for image in decoded
            ]).float()
        rgb = rgb.reshape(views, frames, 3, source_h, source_w)
        if sample.apply_rgb_aug:
            rgb = _apply_rgb_augmentation(rgb, sample.rgb_augmentation)
        rgb = rgb.reshape(views * frames, 3, source_h, source_w)
        theta = sample.theta.to(device, non_blocking=True).reshape(-1, 2, 3)
        grid = F.affine_grid(theta, (views * frames, 3, output_h, output_w), align_corners=True)
        rgb = F.grid_sample(rgb, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        if sample.depth is not None:
            depth = sample.depth.to(device, non_blocking=True).reshape(
                views * frames, 1, source_h, source_w
            )
        else:
            native_depths = decoded_depths[depth_offset:depth_offset + views * frames]
            depth_offset += views * frames
            depth = torch.stack([
                image[..., 0] if image.ndim == 3 else image for image in native_depths
            ]).float().reshape(views, frames, source_h, source_w)
            y = (torch.arange(source_h, device=device, dtype=torch.float32) - source_h / 2 + 0.5)
            x = (torch.arange(source_w, device=device, dtype=torch.float32) - source_w / 2 + 0.5)
            for view in range(views):
                sensor_width = sample.depth_sensor_widths[view]
                focal_length = sample.depth_focal_lengths[view]
                sensor_height = sensor_width * source_h / source_w
                yy, xx = torch.meshgrid(
                    y / source_h * sensor_height,
                    x / source_w * sensor_width,
                    indexing="ij",
                )
                depth[view] /= torch.sqrt(1 + (xx.square() + yy.square()) / focal_length**2)
            depth = depth.reshape(views * frames, 1, source_h, source_w)
        depth = F.grid_sample(depth, grid, mode="nearest", padding_mode="zeros", align_corners=True)
        depth[depth > sample.max_depth] = 0
        if sample.apply_depth_aug:
            generator = torch.Generator(device=device).manual_seed(sample.augmentation_seed)
            invalid = depth == 0
            depth = aug_depth(depth, grid=(16, 16), scale=(0.99, 1.01), shift=(-0.001, 0.001), gn_kernel=(5, 5), gn_sigma=(2, 2), generator=generator)
            depth_views = depth.reshape(views, frames, 1, output_h, output_w)
            for operation in sample.depth_patch_operations:
                kind, view, frame, x0, x1, y0, y1, *parameters = operation
                if kind == 0:
                    patch = depth_views[view, frame, :, y0:y1, x0:x1]
                    fill_mode = parameters[0]
                    fill = (patch.mean(), patch.min(), patch.max(), patch.new_zeros(()))[fill_mode]
                    patch.fill_(fill)
                else:
                    source_view, source_frame, source_x, source_y = parameters
                    depth_views[view, frame, :, y0:y1, x0:x1] = depth_views[
                        source_view,
                        source_frame,
                        :,
                        source_y:source_y + (y1 - y0),
                        source_x:source_x + (x1 - x0),
                    ].clone()
            depth = depth_views.reshape(views * frames, 1, output_h, output_w)
            depth[invalid] = 0
        depth *= sample.depth_scale
        videos.append(rgb.reshape(views, frames, 3, output_h, output_w))
        depths.append(depth.reshape(views, frames, 1, output_h, output_w))

    track_counts = [sample.trajectory.shape[-2] for sample in batch.samples]
    max_tracks = max(track_counts)

    def padded(value: torch.Tensor, axis: int, target: int) -> torch.Tensor:
        count = value.shape[axis]
        if count == target:
            return value
        shape = list(value.shape)
        shape[axis] = target - count
        return torch.cat((value, value.new_zeros(shape)), dim=axis)

    def stack(name: str, axis: int | None = None, *, floating=True):
        values = [getattr(sample, name) for sample in batch.samples]
        if axis is not None:
            values = [padded(value, axis, max_tracks) for value in values]
        result = torch.stack(values).to(device, non_blocking=True)
        return result.float() if floating else result

    track_padding_mask = torch.zeros(
        len(batch.samples), max_tracks, dtype=torch.bool, device=device
    )
    for scene_index, count in enumerate(track_counts):
        track_padding_mask[scene_index, count:] = True

    result = Datapoint(
        video=torch.stack(videos),
        videodepth=torch.stack(depths),
        segmentation=None,
        trajectory=stack("trajectory", -2),
        trajectory_3d=stack("trajectory_3d", -2),
        visibility=stack("visibility", -1),
        valid=stack("valid", -1),
        seq_name=[sample.seq_name for sample in batch.samples],
        intrs=stack("intrs"),
        extrs=stack("extrs"),
        query_points=None,
        query_points_3d=stack("query_points_3d", -2),
        sample_metadata=[sample.metadata for sample in batch.samples],
        track_padding_mask=track_padding_mask,
        track_upscaling_factor=torch.tensor(
            [sample.track_upscaling_factor for sample in batch.samples],
            dtype=torch.float32,
            device=device,
        ),
    )
    if timing_events is not None:
        timing_events[2].record()
    return result


def _record_stream(datapoint: Datapoint, stream: torch.cuda.Stream) -> None:
    for value in vars(datapoint).values():
        if isinstance(value, torch.Tensor) and value.is_cuda:
            value.record_stream(stream)


class _CudaPrefetchIterator:
    def __init__(
        self,
        source: Iterable,
        device: torch.device,
        timing_interval: int,
        queue_depth: int,
        decode_batch_size: int,
    ):
        self.source = iter(source)
        self.device = device
        self.rgb_stream = torch.cuda.Stream(device=device)
        self.depth_stream = torch.cuda.Stream(device=device)
        self.prepare_stream = torch.cuda.Stream(device=device)
        self.timing_interval = timing_interval
        self.queue_depth = queue_depth
        self.decode_batch_size = decode_batch_size
        self.preload_index = 0
        self.last_timing = None
        self.ready = queue.Queue(maxsize=queue_depth)
        self.finished = object()
        self.producer = threading.Thread(target=self._produce, daemon=True)
        self.producer.start()

    @staticmethod
    def _batch_key(encoded: EncodedTapVid3DBatch):
        keys = []
        for sample in encoded.samples:
            views = (
                int(sample.depth.shape[0])
                if sample.depth is not None
                else len(sample.depth_sensor_widths)
            )
            if views < 1 or len(sample.jpeg_bytes) % views:
                raise ValueError("encoded sample has an invalid view/frame layout")
            keys.append((
                sample.image_codec,
                views,
                len(sample.jpeg_bytes) // views,
                sample.output_size,
            ))
        if len(set(keys)) != 1:
            raise ValueError("each encoded source batch must have one view/frame shape")
        return keys[0]

    @staticmethod
    def _slice_datapoint(datapoint: Datapoint, start: int, end: int) -> Datapoint:
        values = {}
        batch_size = len(datapoint.seq_name)
        for field in fields(Datapoint):
            value = getattr(datapoint, field.name)
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == batch_size:
                value = value[start:end]
            elif isinstance(value, list) and len(value) == batch_size:
                value = value[start:end]
            values[field.name] = value
        return Datapoint(**values)

    def _decode_group(self, items, rgb_decoder, depth_decoder):
        samples = [sample for _, encoded, _ in items for sample in encoded.samples]
        events = None
        record_timing = (
            self.timing_interval > 0
            and self.preload_index % self.timing_interval == 0
        )
        if record_timing:
            events = tuple(torch.cuda.Event(enable_timing=True) for _ in range(3))
        with torch.cuda.stream(self.prepare_stream):
            datapoint = decode_tapvid3d_batch(
                EncodedTapVid3DBatch(samples),
                self.device,
                timing_events=events,
                nvimagecodec_rgb_decoder=rgb_decoder,
                nvimagecodec_depth_decoder=depth_decoder,
                rgb_stream=self.rgb_stream,
                depth_stream=self.depth_stream,
                prepare_stream=self.prepare_stream,
            )
            ready_event = torch.cuda.Event()
            ready_event.record(self.prepare_stream)
        offset = 0
        outputs = []
        for position, encoded, gotit in items:
            end = offset + len(encoded.samples)
            outputs.append((
                position,
                self._slice_datapoint(datapoint, offset, end),
                gotit,
                events,
                ready_event,
            ))
            offset = end
        self.preload_index += len(items)
        return outputs

    def _produce(self):
        try:
            torch.cuda.set_device(self.device)
            rgb_decoder = None
            depth_decoder = None
            while True:
                window = []
                for _ in range(self.queue_depth):
                    try:
                        window.append(next(self.source))
                    except StopIteration:
                        break
                if not window:
                    break

                outputs = [None] * len(window)
                grouped = {}
                for position, (encoded, gotit) in enumerate(window):
                    if not all(gotit):
                        outputs[position] = (None, gotit, None, None)
                        continue
                    grouped.setdefault(self._batch_key(encoded), []).append(
                        (position, encoded, gotit)
                    )

                if (
                    rgb_decoder is None
                    and any(key[0] == "nvimagecodec" for key in grouped)
                ):
                    from nvidia import nvimgcodec

                    device_id = (
                        self.device.index
                        if self.device.index is not None
                        else torch.cuda.current_device()
                    )
                    rgb_decoder = nvimgcodec.Decoder(device_id=device_id)
                    depth_decoder = nvimgcodec.Decoder(device_id=device_id)

                next_output = 0
                for items in grouped.values():
                    for start in range(0, len(items), self.decode_batch_size):
                        for position, datapoint, gotit, events, ready_event in self._decode_group(
                            items[start:start + self.decode_batch_size],
                            rgb_decoder,
                            depth_decoder,
                        ):
                            outputs[position] = (datapoint, gotit, events, ready_event)
                        while next_output < len(outputs) and outputs[next_output] is not None:
                            self.ready.put(outputs[next_output])
                            next_output += 1
                while next_output < len(outputs):
                    self.ready.put(outputs[next_output])
                    next_output += 1
                if len(window) < self.queue_depth:
                    break
        except BaseException as error:
            self.ready.put(error)
        finally:
            self.ready.put(self.finished)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.ready.get()
        if item is self.finished:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        current = torch.cuda.current_stream(self.device)
        datapoint, gotit, events, ready_event = item
        if datapoint is not None:
            current.wait_event(ready_event)
            _record_stream(datapoint, current)
            if events is not None:
                events[2].synchronize()
                self.last_timing = (
                    events[0].elapsed_time(events[1]),
                    events[0].elapsed_time(events[2]),
                )
            if self.last_timing is not None:
                decode_ms, prepare_ms = self.last_timing
                for metadata in datapoint.sample_metadata:
                    metadata["gpu_image_decode_ms"] = decode_ms
                    metadata["gpu_jpeg_decode_ms"] = decode_ms
                    metadata["gpu_prepare_total_ms"] = prepare_ms
        return datapoint, gotit


class CudaPrefetchLoader:
    def __init__(
        self,
        loader,
        device: torch.device | None = None,
        timing_interval: int = 0,
        queue_depth: int = 4,
        decode_batch_size: int = 2,
    ):
        if timing_interval < 0:
            raise ValueError("timing_interval must be non-negative")
        if queue_depth < 1 or decode_batch_size < 1:
            raise ValueError("CUDA queue and decode batch sizes must be positive")
        self.loader = loader
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.timing_interval = timing_interval
        self.queue_depth = queue_depth
        self.decode_batch_size = decode_batch_size

    def __iter__(self):
        return self.iter_from(iter(self.loader))

    def iter_from(self, source):
        return _CudaPrefetchIterator(
            source,
            self.device,
            self.timing_interval,
            self.queue_depth,
            self.decode_batch_size,
        )

    def __len__(self):
        return len(self.loader)

    def state_dict(self):
        return self.loader.state_dict()

    def load_state_dict(self, state_dict):
        return self.loader.load_state_dict(state_dict)
