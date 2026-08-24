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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


FORMAT = "mvtracker_estimated_depth"
SCHEMA_VERSION = 2
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


@dataclass(frozen=True)
class SimilarityAlignment:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    residual: float


@dataclass(frozen=True)
class TemporalChunkResult:
    depth: np.ndarray
    cleaned_mask: np.ndarray
    intrinsics: np.ndarray
    extrinsics_w2c: np.ndarray
    scale: float
    alignment_residual: float
    cleaned_coverage: float


@dataclass(frozen=True)
class TemporalBatchTimings:
    load_preprocess_seconds: float
    model_seconds: float
    postprocess_seconds: float
    total_seconds: float
    scene_count: int
    image_count: int


@dataclass(frozen=True)
class TemporalChunkBatchResult:
    scenes: tuple[TemporalChunkResult, ...]
    timings: TemporalBatchTimings


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


class PackedJpegSceneSource(SceneSource):
    """Read the packed JPEG cache used by DIEGESIS and Syn4D.

    The RGB cache and camera arrays are intentionally separate for DIEGESIS:
    the former lives in ``TAPVid3D_MVTracker_cache`` while the latter remains
    in the raw sequence tree.  Syn4D stores both under the same scene root.
    """

    def __init__(
        self,
        root: Path,
        *,
        camera_root: Path | None = None,
        view_ids: Sequence[int] | None = None,
    ) -> None:
        super().__init__(root, view_ids)
        root = Path(root)
        camera_root = root if camera_root is None else Path(camera_root)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        available = [
            int(path.name.removeprefix("view_"))
            for path in root.glob("view_*")
            if path.is_dir()
        ]
        selected = _select_views(sorted(available), view_ids)
        self._jpeg_handles: dict[int, int] = {}
        self._jpeg_offsets: dict[int, np.ndarray] = {}
        self._extrinsics: dict[int, np.ndarray] = {}
        fingerprint_paths = [manifest_path]
        for view in selected:
            view_root = root / f"view_{view}"
            byte_path = view_root / "jpeg_bytes.bin"
            offsets_path = view_root / "jpeg_offsets.npy"
            camera_path = camera_root / str(view) / "extrinsics_w2c.npy"
            self._jpeg_handles[view] = os.open(byte_path, os.O_RDONLY)
            self._jpeg_offsets[view] = np.load(offsets_path, mmap_mode="r", allow_pickle=False)
            self._extrinsics[view] = np.load(camera_path, mmap_mode="r", allow_pickle=False)
            fingerprint_paths.extend([byte_path, offsets_path, camera_path])
        frame_count = int(manifest.get("frames", manifest.get("frame_count", 0)))
        if frame_count <= 0:
            frame_count = int(self._jpeg_offsets[selected[0]].shape[0] - 1)
        resolution_values = manifest.get("cache_resolution") or manifest.get("resolution_hw")
        if resolution_values is None:
            raise ValueError(f"{manifest_path}: missing cache_resolution/resolution_hw")
        resolution = tuple(int(value) for value in resolution_values)
        if len(resolution) != 2:
            raise ValueError(f"{manifest_path}: expected two resolution values")
        resolution = (resolution[0], resolution[1])
        for view in selected:
            offsets = self._jpeg_offsets[view]
            extrinsics = self._extrinsics[view]
            if offsets.shape != (frame_count + 1,):
                raise ValueError(f"{root}/view_{view}/jpeg_offsets.npy: invalid frame count")
            if extrinsics.shape != (frame_count, 4, 4):
                raise ValueError(f"{camera_root}/{view}/extrinsics_w2c.npy: expected [F,4,4]")
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
        offsets = self._jpeg_offsets[view_id]
        start, end = (int(offsets[frame_index]), int(offsets[frame_index + 1]))
        encoded = os.pread(self._jpeg_handles[view_id], end - start, start)
        with Image.open(io.BytesIO(encoded)) as image:
            return image.convert("RGB")

    def extrinsics_w2c(self, frame_indices: Sequence[int]) -> np.ndarray:
        return np.stack(
            [
                np.stack([self._extrinsics[view][frame] for view in self.description.view_ids])
                for frame in frame_indices
            ]
        ).astype(np.float32)

    def __del__(self):
        for handle in getattr(self, "_jpeg_handles", {}).values():
            try:
                os.close(handle)
            except OSError:
                pass


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


def align_camera_centres_sim3(
    predicted_w2c: np.ndarray,
    known_w2c: np.ndarray,
) -> SimilarityAlignment:
    """Fit ``known = scale * rotation @ predicted + translation``."""
    predicted = camera_centres(predicted_w2c).reshape(-1, 3)
    known = camera_centres(known_w2c).reshape(-1, 3)
    if predicted.shape != known.shape or len(predicted) < 2:
        raise ValueError("camera alignment requires corresponding camera centres")
    if not np.isfinite(predicted).all() or not np.isfinite(known).all():
        raise ValueError("camera alignment received non-finite camera centres")
    predicted_mean = predicted.mean(axis=0)
    known_mean = known.mean(axis=0)
    predicted_centered = predicted - predicted_mean
    known_centered = known - known_mean
    predicted_variance = np.mean(np.sum(predicted_centered * predicted_centered, axis=1))
    if predicted_variance <= 1e-12:
        raise ValueError("predicted camera centres have zero variance")
    covariance = known_centered.T @ predicted_centered / len(predicted)
    left, singular_values, right_t = np.linalg.svd(covariance)
    signs = np.ones(3, dtype=np.float64)
    if np.linalg.det(left @ right_t) < 0:
        signs[-1] = -1
    rotation = left @ np.diag(signs) @ right_t
    scale = float(np.sum(singular_values * signs) / predicted_variance)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid camera-centre similarity scale: {scale}")
    translation = known_mean - scale * (rotation @ predicted_mean)
    aligned = scale * np.einsum("ij,nj->ni", rotation, predicted) + translation
    residual = float(np.sqrt(np.mean(np.sum((aligned - known) ** 2, axis=1))))
    return SimilarityAlignment(
        scale=scale,
        rotation=rotation.astype(np.float32),
        translation=translation.astype(np.float32),
        residual=residual,
    )


def apply_similarity_to_extrinsics(
    predicted_w2c: np.ndarray,
    alignment: SimilarityAlignment,
) -> np.ndarray:
    """Express scaled predicted cameras as rigid W2C matrices in the known world."""
    predicted = np.asarray(predicted_w2c, dtype=np.float32)
    rotation = predicted[..., :3, :3] @ alignment.rotation.T
    translation = (
        alignment.scale * predicted[..., :3, 3]
        - np.einsum("...ij,j->...i", rotation, alignment.translation)
    )
    result = np.broadcast_to(np.eye(4, dtype=np.float32), predicted.shape[:-2] + (4, 4)).copy()
    result[..., :3, :3] = rotation
    result[..., :3, 3] = translation
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


def _postprocess_temporal_chunk(
    source: SceneSource,
    frame_indices: Sequence[int],
    sequence_transforms: Sequence[ImageTransform],
    predicted_depth: np.ndarray,
    confidence: np.ndarray,
    intrinsics: np.ndarray,
    predicted_w2c: np.ndarray,
    *,
    confidence_percentile: float,
    edge_rtol: float,
) -> TemporalChunkResult:
    description = source.description
    views = len(description.view_ids)
    height, width = description.resolution_hw
    if predicted_depth.shape[-1] == 1:
        predicted_depth = predicted_depth[..., 0]
    if confidence.shape[-1] == 1:
        confidence = confidence[..., 0]
    known_w2c = source.extrinsics_w2c(frame_indices)
    alignment = align_camera_centres_sim3(predicted_w2c, known_w2c.reshape(-1, 4, 4))
    aligned_w2c = apply_similarity_to_extrinsics(predicted_w2c, alignment)
    depth = np.stack(
        [
            _resize_to_source(value * alignment.scale, transform)
            for value, transform in zip(predicted_depth, sequence_transforms)
        ]
    ).reshape(len(frame_indices), views, height, width)
    restored_confidence = np.stack(
        [
            _resize_to_source(value, transform)
            for value, transform in zip(confidence, sequence_transforms)
        ]
    ).reshape(len(frame_indices), views, height, width)
    restored_intrinsics = np.stack(
        [
            _intrinsics_to_source(value, transform)
            for value, transform in zip(intrinsics, sequence_transforms)
        ]
    ).reshape(len(frame_indices), views, 3, 3)
    cleaned_mask = cleaned_depth_mask(
        depth,
        restored_confidence,
        confidence_percentile=confidence_percentile,
        edge_rtol=edge_rtol,
    )
    return TemporalChunkResult(
        depth=depth,
        cleaned_mask=cleaned_mask,
        intrinsics=restored_intrinsics,
        extrinsics_w2c=aligned_w2c.reshape(len(frame_indices), views, 4, 4),
        scale=alignment.scale,
        alignment_residual=alignment.residual,
        cleaned_coverage=float(cleaned_mask.mean()),
    )


def infer_temporal_chunks(
    sources: Sequence[SceneSource],
    frame_indices: Sequence[int],
    model,
    *,
    device: torch.device,
    image_resolution: int,
    loader_workers: int = 1,
    confidence_percentile: float = 20.0,
    edge_rtol: float = 0.03,
) -> TemporalChunkBatchResult:
    """Infer one homogeneous batch of timestamp-major multi-view scenes."""
    sources = tuple(sources)
    frame_indices = tuple(frame_indices)
    if not sources:
        raise ValueError("sources must not be empty")
    if not frame_indices:
        raise ValueError("frame_indices must not be empty")
    if loader_workers < 1:
        raise ValueError("loader_workers must be positive")
    view_counts = {len(source.description.view_ids) for source in sources}
    resolutions = {source.description.resolution_hw for source in sources}
    if len(view_counts) != 1 or len(resolutions) != 1:
        raise ValueError("batched scenes must have matching view counts and resolutions")

    started = time.perf_counter()

    def prepare(item):
        source, frame = item
        images = [source.load_rgb(view, frame) for view in source.description.view_ids]
        return preprocess_images(images, resolution=image_resolution)

    tasks = [(source, frame) for source in sources for frame in frame_indices]
    if loader_workers == 1:
        prepared = list(map(prepare, tasks))
    else:
        with ThreadPoolExecutor(max_workers=loader_workers) as executor:
            prepared = list(executor.map(prepare, tasks))
    batch_tensors = []
    batch_transforms = []
    frames_per_scene = len(frame_indices)
    for scene_index in range(len(sources)):
        scene_prepared = prepared[
            scene_index * frames_per_scene : (scene_index + 1) * frames_per_scene
        ]
        batch_tensors.append(torch.cat([item[0] for item in scene_prepared]))
        batch_transforms.append(
            tuple(transform for item in scene_prepared for transform in item[1])
        )
    images = torch.stack(batch_tensors)
    if device.type == "cuda":
        images = images.pin_memory()
    load_preprocess_seconds = time.perf_counter() - started

    model_started = time.perf_counter()
    predicted_depth, confidence, intrinsics, predicted_w2c = _model_batch(
        model, images, device
    )
    model_seconds = time.perf_counter() - model_started

    postprocess_started = time.perf_counter()
    results = tuple(
        _postprocess_temporal_chunk(
            source,
            frame_indices,
            transforms,
            predicted_depth[index],
            confidence[index],
            intrinsics[index],
            predicted_w2c[index],
            confidence_percentile=confidence_percentile,
            edge_rtol=edge_rtol,
        )
        for index, (source, transforms) in enumerate(zip(sources, batch_transforms))
    )
    postprocess_seconds = time.perf_counter() - postprocess_started
    total_seconds = time.perf_counter() - started
    return TemporalChunkBatchResult(
        scenes=results,
        timings=TemporalBatchTimings(
            load_preprocess_seconds=load_preprocess_seconds,
            model_seconds=model_seconds,
            postprocess_seconds=postprocess_seconds,
            total_seconds=total_seconds,
            scene_count=len(sources),
            image_count=int(images.shape[0] * images.shape[1]),
        ),
    )


def infer_temporal_chunk(
    source: SceneSource,
    frame_indices: Sequence[int],
    model,
    *,
    device: torch.device,
    image_resolution: int,
    loader_workers: int = 1,
    confidence_percentile: float = 20.0,
    edge_rtol: float = 0.03,
) -> TemporalChunkResult:
    """Run one timestamp-major VGGT-Omega sequence and align it to known cameras."""
    return infer_temporal_chunks(
        [source],
        frame_indices,
        model,
        device=device,
        image_resolution=image_resolution,
        loader_workers=loader_workers,
        confidence_percentile=confidence_percentile,
        edge_rtol=edge_rtol,
    ).scenes[0]


def preprocess_scene(
    source: SceneSource,
    output_root: Path,
    model,
    checkpoint_path: Path,
    checkpoint_hash: str,
    *,
    temporal_chunk_size: int,
    device: torch.device,
    image_resolution: int = 512,
    confidence_percentile: float = 20.0,
    edge_rtol: float = 0.03,
) -> dict:
    if temporal_chunk_size < 1:
        raise ValueError("temporal_chunk_size must be positive")
    description = source.description
    preprocessing = {
        "mode": "balanced",
        "image_resolution": image_resolution,
        "patch_size": 16,
        "temporal_chunk_size": temporal_chunk_size,
        "sequence_order": "timestamp_major_views_minor",
    }
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
    chunk_records = []
    for first in range(0, frames, temporal_chunk_size):
        frame_indices = list(range(first, min(first + temporal_chunk_size, frames)))
        result = infer_temporal_chunk(
            source,
            frame_indices,
            model,
            device=device,
            image_resolution=image_resolution,
            confidence_percentile=confidence_percentile,
            edge_rtol=edge_rtol,
        )
        for chunk_index, frame in enumerate(frame_indices):
            depth_output[:, frame] = result.depth[chunk_index]
            mask_output[:, frame] = result.cleaned_mask[chunk_index]
            scales_output[frame] = result.scale
            intrinsics_output[:, frame] = result.intrinsics[chunk_index]
            extrinsics_output[:, frame] = result.extrinsics_w2c[chunk_index]
        residuals.append(result.alignment_residual)
        coverages.append(result.cleaned_coverage)
        chunk_records.append(
            {
                "first_frame": frame_indices[0],
                "last_frame": frame_indices[-1],
                "sequence_length": len(frame_indices) * views,
                "scale": result.scale,
                "camera_center_alignment_rmse_m": result.alignment_residual,
                "cleaned_coverage": result.cleaned_coverage,
            }
        )
        _emit(
            "temporal_chunk_complete",
            scene=description.name,
            first_frame=frame_indices[0],
            last_frame=frame_indices[-1],
            temporal_chunk_size=len(frame_indices),
            sequence_length=len(frame_indices) * views,
            alignment_scale=result.scale,
            camera_center_alignment_rmse_m=result.alignment_residual,
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
        "temporal_chunks": chunk_records,
        "metrics": {
            "duration_seconds": duration,
            "timestamps_per_second": frames / duration,
            "mean_camera_center_alignment_rmse_m": float(np.mean(residuals)),
            "max_camera_center_alignment_rmse_m": float(np.max(residuals)),
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


def profile_temporal_chunk_sizes(
    source: SceneSource,
    model,
    *,
    temporal_chunk_sizes: Sequence[int],
    device: torch.device,
    image_resolution: int = 512,
    max_vram_fraction: float = 0.9,
) -> int:
    """Measure candidate temporal reconstruction sizes and return the largest safe one."""
    if not temporal_chunk_sizes or any(size < 1 for size in temporal_chunk_sizes):
        raise ValueError("temporal_chunk_sizes must contain positive integers")
    results = []
    total_memory = torch.cuda.get_device_properties(device).total_memory
    views = len(source.description.view_ids)
    for temporal_chunk_size in temporal_chunk_sizes:
        frame_indices = [
            index % source.description.frame_count for index in range(temporal_chunk_size)
        ]
        sequence_tensors = []
        for frame in frame_indices:
            images = [source.load_rgb(view, frame) for view in source.description.view_ids]
            sequence_tensors.extend(preprocess_images(images, resolution=image_resolution)[0])
        images = torch.stack(sequence_tensors).unsqueeze(0)
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
                "temporal_chunk_size": temporal_chunk_size,
                "sequence_length": temporal_chunk_size * views,
                "duration_seconds": duration,
                "timestamps_per_second": temporal_chunk_size / duration,
                "peak_cuda_memory_bytes": int(peak_allocated),
                "peak_cuda_reserved_bytes": int(peak_reserved),
                "safe": safe,
            }
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            result = {
                "temporal_chunk_size": temporal_chunk_size,
                "sequence_length": temporal_chunk_size * views,
                "oom": True,
                "safe": False,
            }
        results.append(result)
        _emit("temporal_chunk_profile", scene=source.description.name, **result)
    safe_sizes = [
        result["temporal_chunk_size"] for result in results if result["safe"]
    ]
    if not safe_sizes:
        raise RuntimeError(
            "no profiled temporal chunk size stayed within the requested VRAM limit"
        )
    selected = max(safe_sizes)
    _emit(
        "temporal_chunk_profile_selected",
        scene=source.description.name,
        temporal_chunk_size=selected,
    )
    return selected
