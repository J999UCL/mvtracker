"""Batched VGGT-Omega depth preprocessing for MVTracker datasets."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


FORMAT = "mvtracker_estimated_depth"
SCHEMA_VERSION = 1
PROVIDER = "vggt_omega"
VGGT_OMEGA_SOURCE_REVISION = "39a0cb8af88554f15ddcb5354cd52bde588fa014"
VGGT_OMEGA_CHECKPOINT_REVISION = "ba9db085d6b7349b738fa2e37d198bb4dd077954"


@dataclass(frozen=True)
class ImageTransform:
    source_hw: tuple[int, int]
    crop_xywh: tuple[int, int, int, int]
    model_hw: tuple[int, int]


@dataclass(frozen=True)
class SceneDescription:
    name: str
    frame_count: int
    view_ids: tuple[int, ...]
    resolution_hw: tuple[int, int]
    source_fingerprint: str


class SceneSource(ABC):
    """Common input contract for a synchronized multi-view scene."""

    def __init__(self, root: Path, view_ids: Sequence[int] | None = None) -> None:
        self.root = root
        self._requested_view_ids = None if view_ids is None else tuple(view_ids)

    @property
    @abstractmethod
    def description(self) -> SceneDescription:
        raise NotImplementedError

    @abstractmethod
    def load_rgb(self, view_id: int, frame_index: int) -> Image.Image:
        raise NotImplementedError

    @abstractmethod
    def extrinsics_w2c(self, frame_indices: Sequence[int]) -> np.ndarray:
        """Return float32 camera transforms shaped [B, V, 4, 4]."""
        raise NotImplementedError


def _numeric_view_dirs(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )


def _select_views(available: Sequence[int], requested: Sequence[int] | None) -> tuple[int, ...]:
    selected = tuple(available if requested is None else requested)
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"requested views are absent: {missing}")
    if len(selected) < 2:
        raise ValueError("metric camera-baseline scaling requires at least two views")
    if len(set(selected)) != len(selected):
        raise ValueError("view IDs must be unique")
    return selected


def _stat_fingerprint(paths: Iterable[Path]) -> str:
    records = []
    for path in sorted(set(paths)):
        stat = path.stat()
        records.append((str(path), stat.st_size, stat.st_mtime_ns))
    payload = json.dumps(records, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


class TapVid3DSceneSource(SceneSource):
    """Read one sequence in the raw multi-view TAPVid-3D format."""

    def __init__(self, root: Path, view_ids: Sequence[int] | None = None) -> None:
        super().__init__(root, view_ids)
        tracks = np.load(root / "tracks_xyz.npy", mmap_mode="r", allow_pickle=False)
        if tracks.ndim != 3 or tracks.shape[-1] != 3:
            raise ValueError(f"{root / 'tracks_xyz.npy'}: expected [F, P, 3]")
        available = [int(path.name) for path in _numeric_view_dirs(root)]
        selected = _select_views(available, view_ids)
        self._jpeg_arrays = {
            view: np.load(root / str(view) / "images_jpeg_bytes.npy", allow_pickle=True)
            for view in selected
        }
        self._extrinsics = {
            view: np.load(root / str(view) / "extrinsics_w2c.npy", mmap_mode="r")
            for view in selected
        }
        frame_count = int(tracks.shape[0])
        if any(array.shape != (frame_count,) for array in self._jpeg_arrays.values()):
            raise ValueError(f"{root}: JPEG arrays do not match the scene frame count")
        first = self.load_rgb(selected[0], 0)
        resolution = (first.height, first.width)
        fingerprint_paths = [root / "tracks_xyz.npy"]
        for view in selected:
            view_root = root / str(view)
            fingerprint_paths.extend(
                [view_root / "images_jpeg_bytes.npy", view_root / "extrinsics_w2c.npy"]
            )
            extrinsics = self._extrinsics[view]
            if extrinsics.shape != (frame_count, 4, 4):
                raise ValueError(f"{view_root / 'extrinsics_w2c.npy'}: expected [F, 4, 4]")
        self._description = SceneDescription(
            name=root.name,
            frame_count=frame_count,
            view_ids=selected,
            resolution_hw=resolution,
            source_fingerprint=_stat_fingerprint(fingerprint_paths),
        )

    @property
    def description(self) -> SceneDescription:
        return self._description

    def load_rgb(self, view_id: int, frame_index: int) -> Image.Image:
        encoded = np.asarray(self._jpeg_arrays[view_id][frame_index], dtype=np.uint8)
        with Image.open(io.BytesIO(encoded.tobytes())) as image:
            return image.convert("RGB")

    def extrinsics_w2c(self, frame_indices: Sequence[int]) -> np.ndarray:
        return np.stack(
            [
                np.stack(
                    [
                        self._extrinsics[view][frame]
                        for view in self.description.view_ids
                    ]
                )
                for frame in frame_indices
            ]
        ).astype(np.float32)


class MVKubricSceneSource(SceneSource):
    """Read RGB and camera metadata from one native MV-Kubric scene."""

    def __init__(self, root: Path, view_ids: Sequence[int] | None = None) -> None:
        super().__init__(root, view_ids)
        view_dirs = sorted(
            (path for path in root.glob("view_*") if path.is_dir()),
            key=lambda path: int(path.name.removeprefix("view_")),
        )
        available = [int(path.name.removeprefix("view_")) for path in view_dirs]
        selected = _select_views(available, view_ids)
        self._frame_paths: dict[int, list[Path]] = {}
        self._extrinsics: dict[int, np.ndarray] = {}
        fingerprint_paths = []
        frame_count = None
        resolution = None
        for view in selected:
            view_root = root / f"view_{view}"
            frames = sorted(view_root.glob("rgba_*.png"))
            if not frames:
                raise ValueError(f"{view_root}: no rgba_*.png frames")
            frame_count = len(frames) if frame_count is None else frame_count
            if len(frames) != frame_count:
                raise ValueError(f"{root}: views have different frame counts")
            metadata_path = view_root / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            camera = metadata["camera"]
            positions = np.asarray(camera["positions"], dtype=np.float64)
            quaternions = np.asarray(camera["quaternions"], dtype=np.float64)
            if positions.shape != (frame_count, 3) or quaternions.shape != (frame_count, 4):
                raise ValueError(f"{metadata_path}: camera animation does not match RGB frames")
            rotations = _quaternion_wxyz_to_rotation_matrix(quaternions)
            camera_to_world = np.zeros((frame_count, 4, 4), dtype=np.float64)
            camera_to_world[:, :3, :3] = rotations
            camera_to_world[:, :3, 3] = positions
            camera_to_world[:, 3, 3] = 1.0
            world_to_camera = np.linalg.inv(camera_to_world)
            cv_conversion = np.diag([1.0, -1.0, -1.0, 1.0])
            self._extrinsics[view] = (cv_conversion @ world_to_camera).astype(np.float32)
            self._frame_paths[view] = frames
            fingerprint_paths.extend([metadata_path, *frames])
            with Image.open(frames[0]) as image:
                current_resolution = (image.height, image.width)
            resolution = current_resolution if resolution is None else resolution
            if current_resolution != resolution:
                raise ValueError(f"{root}: views have different image resolutions")
        self._description = SceneDescription(
            name=root.name,
            frame_count=int(frame_count),
            view_ids=selected,
            resolution_hw=resolution,
            source_fingerprint=_stat_fingerprint(fingerprint_paths),
        )

    @property
    def description(self) -> SceneDescription:
        return self._description

    def load_rgb(self, view_id: int, frame_index: int) -> Image.Image:
        with Image.open(self._frame_paths[view_id][frame_index]) as image:
            if image.mode == "RGBA":
                background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, image)
            return image.convert("RGB")

    def extrinsics_w2c(self, frame_indices: Sequence[int]) -> np.ndarray:
        return np.stack(
            [
                np.stack([self._extrinsics[view][frame] for view in self.description.view_ids])
                for frame in frame_indices
            ]
        )


def _quaternion_wxyz_to_rotation_matrix(quaternions: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64)
    quaternions = quaternions / np.linalg.norm(quaternions, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(quaternions, -1, 0)
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    ).reshape(quaternions.shape[:-1] + (3, 3))


def _crop_supported_aspect(image: Image.Image) -> tuple[Image.Image, tuple[int, int, int, int]]:
    width, height = image.size
    aspect = height / width
    if aspect < 0.5:
        crop_width = min(width, max(1, int(round(height / 0.5))))
        left = max((width - crop_width) // 2, 0)
        return image.crop((left, 0, left + crop_width, height)), (left, 0, crop_width, height)
    if aspect > 2.0:
        crop_height = min(height, max(1, int(round(width * 2.0))))
        top = max((height - crop_height) // 2, 0)
        return image.crop((0, top, width, top + crop_height)), (0, top, width, crop_height)
    return image, (0, 0, width, height)


def _balanced_shape(height: int, width: int, resolution: int = 512, patch_size: int = 16) -> tuple[int, int]:
    aspect = height / width
    token_count = (resolution // patch_size) ** 2
    width_patches_float = np.sqrt(token_count / aspect)
    height_patches_float = token_count / width_patches_float
    width_patches = max(1, int(np.round(width_patches_float)))
    height_patches = max(1, int(np.round(height_patches_float)))
    return height_patches * patch_size, width_patches * patch_size


def preprocess_images(
    images: Sequence[Image.Image],
    *,
    resolution: int = 512,
    patch_size: int = 16,
) -> tuple[torch.Tensor, tuple[ImageTransform, ...]]:
    """Apply VGGT-Omega's official balanced crop and resize policy."""
    tensors = []
    transforms = []
    for source in images:
        source = source.convert("RGB")
        source_hw = (source.height, source.width)
        cropped, crop = _crop_supported_aspect(source)
        model_hw = _balanced_shape(cropped.height, cropped.width, resolution, patch_size)
        resized = cropped.resize((model_hw[1], model_hw[0]), Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.float32).copy() / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
        transforms.append(ImageTransform(source_hw, crop, model_hw))
    shapes = {tuple(tensor.shape[-2:]) for tensor in tensors}
    if len(shapes) != 1:
        raise ValueError(f"selected views preprocess to different shapes: {sorted(shapes)}")
    return torch.stack(tensors), tuple(transforms)


def _resize_to_source(array: np.ndarray, transform: ImageTransform) -> np.ndarray:
    source_h, source_w = transform.source_hw
    left, top, crop_w, crop_h = transform.crop_xywh
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))[None, None]
    resized = F.interpolate(tensor, size=(crop_h, crop_w), mode="bilinear", align_corners=True)[0, 0].numpy()
    restored = np.zeros((source_h, source_w), dtype=np.float32)
    restored[top : top + crop_h, left : left + crop_w] = resized
    return restored


def _intrinsics_to_source(intrinsics: np.ndarray, transform: ImageTransform) -> np.ndarray:
    model_h, model_w = transform.model_hw
    left, top, crop_w, crop_h = transform.crop_xywh
    scale_x = (crop_w - 1) / max(model_w - 1, 1)
    scale_y = (crop_h - 1) / max(model_h - 1, 1)
    result = np.asarray(intrinsics, dtype=np.float32).copy()
    result[0, 0] *= scale_x
    result[1, 1] *= scale_y
    result[0, 2] = result[0, 2] * scale_x + left
    result[1, 2] = result[1, 2] * scale_y + top
    return result


def camera_centres(extrinsics_w2c: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics_w2c, dtype=np.float64)
    rotation = extrinsics[..., :3, :3]
    translation = extrinsics[..., :3, 3]
    return -np.einsum("...ji,...j->...i", rotation, translation)


def metric_scale_from_camera_baselines(
    predicted_w2c: np.ndarray,
    known_w2c: np.ndarray,
) -> tuple[float, float]:
    """Recover metric scale and RMS baseline residual for one timestamp."""
    predicted = camera_centres(predicted_w2c)
    known = camera_centres(known_w2c)
    first, second = np.triu_indices(len(predicted), k=1)
    predicted_distances = np.linalg.norm(predicted[first] - predicted[second], axis=-1)
    known_distances = np.linalg.norm(known[first] - known[second], axis=-1)
    valid = (
        np.isfinite(predicted_distances)
        & np.isfinite(known_distances)
        & (predicted_distances > 1e-8)
        & (known_distances > 1e-8)
    )
    if not np.any(valid):
        raise ValueError("camera baselines do not define a finite positive metric scale")
    scale = float(np.median(known_distances[valid] / predicted_distances[valid]))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid camera-baseline scale: {scale}")
    residual = float(
        np.sqrt(np.mean(np.square(scale * predicted_distances[valid] - known_distances[valid])))
    )
    return scale, residual


def scale_extrinsics(extrinsics_w2c: np.ndarray, scale: float) -> np.ndarray:
    result = np.eye(4, dtype=np.float32)[None].repeat(len(extrinsics_w2c), axis=0)
    result[:, :3, :4] = np.asarray(extrinsics_w2c, dtype=np.float32)
    result[:, :3, 3] *= scale
    return result


def depth_edges(depth: np.ndarray, rtol: float = 0.03, kernel_size: int = 3) -> np.ndarray:
    """VGGT-Omega's published local relative depth-jump filter."""
    depth = np.asarray(depth)
    original_shape = depth.shape
    flat = depth.reshape(-1, *original_shape[-2:])
    pad = kernel_size // 2
    padded = np.pad(flat, ((0, 0), (pad, pad), (pad, pad)), mode="edge")
    depth_max = np.full_like(flat, -np.inf)
    depth_min = np.full_like(flat, np.inf)
    for y in range(kernel_size):
        for x in range(kernel_size):
            window = padded[:, y : y + flat.shape[-2], x : x + flat.shape[-1]]
            depth_max = np.maximum(depth_max, window)
            depth_min = np.minimum(depth_min, window)
    relative_jump = (depth_max - depth_min) / np.maximum(np.abs(flat), 1e-6)
    return (relative_jump > rtol).reshape(original_shape)


def cleaned_depth_mask(
    depth: np.ndarray,
    confidence: np.ndarray,
    *,
    confidence_percentile: float = 20.0,
    edge_rtol: float = 0.03,
) -> np.ndarray:
    """Apply the official confidence percentile and depth-edge policy."""
    depth = np.asarray(depth)
    confidence = np.asarray(confidence)
    valid = np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
    if not np.any(valid):
        return np.zeros(depth.shape, dtype=bool)
    threshold = np.percentile(confidence[valid], confidence_percentile)
    return valid & (confidence >= threshold) & (confidence > 1e-5) & ~depth_edges(depth, edge_rtol)


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(checkpoint_path: Path, device: torch.device):
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega().eval()
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    return model.to(device)


def _manifest_matches(path: Path, expected: dict) -> bool:
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True or not all(
        manifest.get(key) == value for key, value in expected.items()
    ):
        return False
    return all((path / filename).is_file() for filename in manifest.get("arrays", {}))


def _emit(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def _model_batch(
    model,
    images: torch.Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from vggt_omega.utils.pose_enc import encoding_to_camera

    with torch.inference_mode():
        predictions = model(images.to(device, non_blocking=True))
        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"], predictions["images"].shape[-2:]
        )
    return tuple(
        value.detach().float().cpu().numpy()
        for value in (predictions["depth"], predictions["depth_conf"], intrinsics, extrinsics)
    )


def preprocess_scene(
    source: SceneSource,
    output_root: Path,
    model,
    checkpoint_path: Path,
    checkpoint_hash: str,
    *,
    batch_size: int,
    device: torch.device,
    image_resolution: int = 512,
    confidence_percentile: float = 20.0,
    edge_rtol: float = 0.03,
) -> dict:
    description = source.description
    preprocessing = {"mode": "balanced", "image_resolution": image_resolution, "patch_size": 16}
    cleaning = {
        "confidence_percentile": confidence_percentile,
        "depth_edge_relative_tolerance": edge_rtol,
    }
    expected_manifest = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "provider": PROVIDER,
        "source_sequence": description.name,
        "source_fingerprint": description.source_fingerprint,
        "view_ids": list(description.view_ids),
        "frame_count": description.frame_count,
        "resolution_hw": list(description.resolution_hw),
        "checkpoint_sha256": checkpoint_hash,
        "model_source_revision": VGGT_OMEGA_SOURCE_REVISION,
        "checkpoint_revision": VGGT_OMEGA_CHECKPOINT_REVISION,
        "preprocessing": preprocessing,
        "cleaning": cleaning,
    }
    target = output_root / description.name
    if _manifest_matches(target, expected_manifest):
        _emit("scene_skipped", scene=description.name, reason="matching_manifest")
        return {"scene": description.name, "skipped": True}
    staging = output_root / f".{description.name}.tmp-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)

    views = len(description.view_ids)
    frames = description.frame_count
    height, width = description.resolution_hw
    depth_output = np.lib.format.open_memmap(
        staging / "depth.npy", mode="w+", dtype=np.float32, shape=(views, frames, height, width)
    )
    mask_output = np.lib.format.open_memmap(
        staging / "cleaned_mask.npy", mode="w+", dtype=np.bool_, shape=(views, frames, height, width)
    )
    scales_output = np.lib.format.open_memmap(
        staging / "scales.npy", mode="w+", dtype=np.float32, shape=(frames,)
    )
    intrinsics_output = np.lib.format.open_memmap(
        staging / "predicted_intrinsics.npy", mode="w+", dtype=np.float32, shape=(views, frames, 3, 3)
    )
    extrinsics_output = np.lib.format.open_memmap(
        staging / "predicted_extrinsics_w2c.npy", mode="w+", dtype=np.float32, shape=(views, frames, 4, 4)
    )

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    residuals = []
    coverages = []
    for first in range(0, frames, batch_size):
        frame_indices = list(range(first, min(first + batch_size, frames)))
        batch_tensors = []
        batch_transforms = []
        for frame in frame_indices:
            images = [source.load_rgb(view, frame) for view in description.view_ids]
            tensors, transforms = preprocess_images(images, resolution=image_resolution)
            batch_tensors.append(tensors)
            batch_transforms.append(transforms)
        images = torch.stack(batch_tensors)
        predicted_depth, confidence, intrinsics, predicted_w2c = _model_batch(model, images, device)
        if predicted_depth.shape[-1] == 1:
            predicted_depth = predicted_depth[..., 0]
        if confidence.shape[-1] == 1:
            confidence = confidence[..., 0]
        known_w2c = source.extrinsics_w2c(frame_indices)
        for batch_index, frame in enumerate(frame_indices):
            scale, residual = metric_scale_from_camera_baselines(
                predicted_w2c[batch_index], known_w2c[batch_index]
            )
            restored_depth = []
            restored_confidence = []
            restored_intrinsics = []
            for view_index, transform in enumerate(batch_transforms[batch_index]):
                restored_depth.append(
                    _resize_to_source(predicted_depth[batch_index, view_index] * scale, transform)
                )
                restored_confidence.append(
                    _resize_to_source(confidence[batch_index, view_index], transform)
                )
                restored_intrinsics.append(
                    _intrinsics_to_source(intrinsics[batch_index, view_index], transform)
                )
            restored_depth = np.stack(restored_depth)
            restored_confidence = np.stack(restored_confidence)
            clean_mask = cleaned_depth_mask(
                restored_depth,
                restored_confidence,
                confidence_percentile=confidence_percentile,
                edge_rtol=edge_rtol,
            )
            depth_output[:, frame] = restored_depth
            mask_output[:, frame] = clean_mask
            scales_output[frame] = scale
            intrinsics_output[:, frame] = np.stack(restored_intrinsics)
            extrinsics_output[:, frame] = scale_extrinsics(predicted_w2c[batch_index], scale)
            residuals.append(residual)
            coverages.append(float(clean_mask.mean()))
        _emit(
            "batch_complete",
            scene=description.name,
            first_frame=frame_indices[0],
            last_frame=frame_indices[-1],
            batch_size=len(frame_indices),
            elapsed_seconds=time.perf_counter() - started,
        )

    for output in (depth_output, mask_output, scales_output, intrinsics_output, extrinsics_output):
        output.flush()
    duration = time.perf_counter() - started
    manifest = {
        **expected_manifest,
        "model_source_revision": VGGT_OMEGA_SOURCE_REVISION,
        "checkpoint_revision": VGGT_OMEGA_CHECKPOINT_REVISION,
        "checkpoint_file": checkpoint_path.name,
        "arrays": {
            "depth.npy": {"shape": [views, frames, height, width], "dtype": "float32"},
            "cleaned_mask.npy": {"shape": [views, frames, height, width], "dtype": "bool"},
            "scales.npy": {"shape": [frames], "dtype": "float32"},
            "predicted_intrinsics.npy": {"shape": [views, frames, 3, 3], "dtype": "float32"},
            "predicted_extrinsics_w2c.npy": {"shape": [views, frames, 4, 4], "dtype": "float32"},
        },
        "metrics": {
            "duration_seconds": duration,
            "timestamps_per_second": frames / duration,
            "mean_camera_baseline_residual": float(np.mean(residuals)),
            "max_camera_baseline_residual": float(np.max(residuals)),
            "mean_cleaned_coverage": float(np.mean(coverages)),
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        },
        "complete": True,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if target.exists():
        shutil.rmtree(target)
    os.replace(staging, target)
    _emit("scene_complete", scene=description.name, **manifest["metrics"])
    return manifest["metrics"]


def profile_batch_sizes(
    source: SceneSource,
    model,
    *,
    batch_sizes: Sequence[int],
    device: torch.device,
    image_resolution: int = 512,
    max_vram_fraction: float = 0.9,
) -> int:
    """Measure candidate timestamp batch sizes and return the largest safe one."""
    results = []
    total_memory = torch.cuda.get_device_properties(device).total_memory
    for batch_size in batch_sizes:
        frame_indices = [index % source.description.frame_count for index in range(batch_size)]
        tensors = []
        for frame in frame_indices:
            images = [source.load_rgb(view, frame) for view in source.description.view_ids]
            tensors.append(preprocess_images(images, resolution=image_resolution)[0])
        images = torch.stack(tensors)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        try:
            _model_batch(model, images, device)
            torch.cuda.synchronize(device)
            duration = time.perf_counter() - started
            peak_allocated = torch.cuda.max_memory_allocated(device)
            peak_reserved = torch.cuda.max_memory_reserved(device)
            safe = peak_reserved <= total_memory * max_vram_fraction
            result = {
                "batch_size": batch_size,
                "duration_seconds": duration,
                "timestamps_per_second": batch_size / duration,
                "peak_cuda_memory_bytes": int(peak_allocated),
                "peak_cuda_reserved_bytes": int(peak_reserved),
                "safe": safe,
            }
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            result = {"batch_size": batch_size, "oom": True, "safe": False}
        results.append(result)
        _emit("batch_profile", scene=source.description.name, **result)
    safe_sizes = [result["batch_size"] for result in results if result["safe"]]
    if not safe_sizes:
        raise RuntimeError("no profiled batch size stayed within the requested VRAM limit")
    selected = max(safe_sizes)
    _emit("batch_profile_selected", scene=source.description.name, batch_size=selected)
    return selected
