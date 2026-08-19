"""Indexed MV-Kubric WebDataset access.

MV-Kubric samples are assembled from one scene metadata record and the
selected scene/view media records.  WIDS provides random access to those
records; the existing encoded-sample decoder still owns GPU decoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np
import torch

from mvtracker.datasets.kubric_multiview_dataset import KubricMultiViewDataset
from mvtracker.datasets.tapvid3d_multiview_dataset import (
    EncodedTapVid3DSample,
    SamplePlan,
    _preselect_motion_tracks,
    _project,
    _sample_depth_patch_operations,
    _sample_rgb_augmentation,
    _sample_tracks,
    _scene_transform,
    _spatial_transform,
    _visible_path_lengths,
    collate_encoded_tapvid3d,
)


WEB_DATASET_FORMAT = "mvtracker-kubric-webdataset"
SOURCE_VIEW_COUNT = 10
WIDS_DESCRIPTOR_FIELD = "wids_descriptor"


def _bytes_from_value(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tobytes()
    if hasattr(value, "getvalue"):
        return value.getvalue()
    return bytes(value)


def _component(record: Mapping[str, Any], name: str) -> Any:
    key = name if name.startswith(".") else f".{name}"
    try:
        return record[key]
    except KeyError as error:
        raise KeyError(f"WIDS sample {record.get('__key__')!r} lacks {key}") from error


def _npz_component(record: Mapping[str, Any], name: str) -> dict[str, np.ndarray]:
    with np.load(BytesIO(_bytes_from_value(_component(record, name))), allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _packed_frames(record: Mapping[str, Any], name: str) -> tuple[bytes, ...]:
    packed = _npz_component(record, name)
    encoded = np.asarray(packed["bytes"], dtype=np.uint8).reshape(-1)
    offsets = np.asarray(packed["offsets"], dtype=np.int64).reshape(-1)
    if len(offsets) < 2 or offsets[0] != 0:
        raise ValueError(f"WIDS component {name} has invalid frame offsets")
    if offsets[-1] != encoded.size or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"WIDS component {name} is truncated or unsorted")
    return tuple(bytes(encoded[start:end]) for start, end in zip(offsets[:-1], offsets[1:]))


@dataclass(frozen=True)
class KubricSceneMetadata:
    name: str
    tracks_3d: np.ndarray
    visibility: np.ndarray
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    sensor_widths: np.ndarray
    focal_lengths: np.ndarray
    invalid_frame_indices: tuple[int, ...]
    resolution_hw: tuple[int, int]

    @property
    def frame_count(self) -> int:
        return int(self.tracks_3d.shape[0])

    @property
    def view_count(self) -> int:
        return int(self.visibility.shape[0])


class KubricWebDatasetCatalog:
    """Read the small catalog without opening any WIDS shard."""

    def __init__(self, manifest_path: str | Path):
        manifest_path = Path(manifest_path)
        self.root = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != WEB_DATASET_FORMAT:
            raise ValueError(f"{manifest_path}: unsupported WebDataset format")
        scenes = manifest.get("scenes")
        if not isinstance(scenes, dict) or not scenes:
            raise ValueError(f"{manifest_path}: manifest has no scenes")
        descriptor = manifest.get(WIDS_DESCRIPTOR_FIELD)
        if not isinstance(descriptor, str) or not descriptor:
            raise ValueError(f"{manifest_path}: missing {WIDS_DESCRIPTOR_FIELD}")
        self.descriptor_path = (self.root / descriptor).resolve()
        if not self.descriptor_path.is_file():
            raise FileNotFoundError(f"WIDS descriptor is missing: {self.descriptor_path}")
        self.scenes = scenes
        self.source_fingerprint = str(manifest.get("source_fingerprint", "webdataset"))

    def scene(self, name: str):
        try:
            entry = dict(self.scenes[name])
        except KeyError as error:
            raise KeyError(f"scene {name!r} is not in the WebDataset catalog") from error
        view_map = entry.get("views")
        if not isinstance(view_map, dict):
            raise ValueError(f"scene {name!r}: catalog has no view map")
        view_names = entry.get("view_names")
        if view_names is None:
            view_names = [f"view_{int(view):02d}" for view in sorted(map(int, view_map))]
        entry["view_names"] = list(view_names)
        entry.setdefault("n_frames", 24)
        entry.setdefault("invalid_frame_indices", [])
        if len(view_names) != SOURCE_VIEW_COUNT:
            raise ValueError(f"scene {name!r}: expected ten views, got {len(view_names)}")
        return entry, {}


class _WidsRecordStore:
    """One WIDS reader per materialization thread."""

    def __init__(self, descriptor_path: str | Path, *, reader_factory: Callable[[str], Any] | None = None):
        self.descriptor_path = str(Path(descriptor_path).resolve())
        self._reader_factory = reader_factory
        self._local = threading.local()

    def _reader(self):
        reader = getattr(self._local, "reader", None)
        if reader is None:
            if self._reader_factory is None:
                try:
                    import wids
                except ImportError as error:
                    raise RuntimeError("MV-Kubric WebDataset loading requires the wids package") from error
                reader = wids.ShardListDataset(self.descriptor_path, transformations=[])
            else:
                reader = self._reader_factory(self.descriptor_path)
            self._local.reader = reader
        return reader

    def get(self, index: int) -> Mapping[str, Any]:
        return self._reader()[int(index)]


def _scene_metadata(record: Mapping[str, Any], scene_name: str) -> KubricSceneMetadata:
    meta = _npz_component(record, "meta.npz")
    tracks_3d = np.asarray(meta["tracks_3d"], dtype=np.float32)
    visibility = np.asarray(meta["visibility"], dtype=np.bool_)
    if visibility.ndim != 3 or visibility.shape[1:] != tracks_3d.shape[:2]:
        raise ValueError(f"scene {scene_name}: metadata visibility does not match tracks_3d")
    if visibility.shape[0] != SOURCE_VIEW_COUNT:
        raise ValueError(f"scene {scene_name}: expected ten views, got {visibility.shape[0]}")
    for key in ("intrinsics", "extrinsics", "sensor_widths", "focal_lengths"):
        if np.asarray(meta[key]).shape[0] != SOURCE_VIEW_COUNT:
            raise ValueError(f"scene {scene_name}: metadata {key} does not contain ten views")
    return KubricSceneMetadata(
        name=scene_name,
        tracks_3d=tracks_3d,
        visibility=visibility,
        intrinsics=np.asarray(meta["intrinsics"], dtype=np.float32),
        extrinsics=np.asarray(meta["extrinsics"], dtype=np.float32),
        sensor_widths=np.asarray(meta["sensor_widths"], dtype=np.float32),
        focal_lengths=np.asarray(meta["focal_lengths"], dtype=np.float32),
        invalid_frame_indices=tuple(int(value) for value in np.asarray(meta.get("invalid_frame_indices", ())).reshape(-1)),
        resolution_hw=tuple(int(value) for value in np.asarray(meta["resolution_hw"]).reshape(-1)),
    )


class DaliKubricMultiViewDataset(KubricMultiViewDataset):
    """MV-Kubric sampler using indexed scene/view records and live sampling."""

    collate_fn = staticmethod(collate_encoded_tapvid3d)
    requires_cuda_prefetch = True

    def __init__(self, *args, webdataset_root: str, webdataset_split: str = "train", **kwargs):
        manifest_path = Path(webdataset_root) / webdataset_split / "manifest.json"
        self.catalog = KubricWebDatasetCatalog(manifest_path)
        self.webdataset_root = str(webdataset_root)
        self.webdataset_split = webdataset_split
        self._records = _WidsRecordStore(self.catalog.descriptor_path)
        kwargs["data_root"] = str(Path(webdataset_root))
        kwargs["metadata_catalog"] = self.catalog
        kwargs["metadata_index_root"] = None
        super().__init__(*args, **kwargs)

    def _scene(self, scene_name: str) -> KubricSceneMetadata:
        entry, _ = self.catalog.scene(scene_name)
        metadata_index = entry.get("metadata_index")
        if metadata_index is None:
            raise ValueError(f"scene {scene_name!r}: catalog has no metadata_index")
        return _scene_metadata(self._records.get(int(metadata_index)), scene_name)

    def _media_indices(self, scene_name: str, views: tuple[int, ...]) -> tuple[int, ...]:
        entry, _ = self.catalog.scene(scene_name)
        view_map = entry["views"]
        return tuple(int(view_map[str(view)]["media_index"]) for view in views)

    def plan_sample(self, index) -> SamplePlan | None:
        request = index if hasattr(index, "virtual_index") else None
        virtual_index = request.virtual_index if request is not None else int(index)
        scene_index = (
            request.scene_index
            if request is not None and request.scene_index is not None
            else virtual_index % self.real_len
        )
        if not 0 <= scene_index < self.real_len:
            raise IndexError(f"scene index {scene_index} is outside [0, {self.real_len})")
        scene_name = self.seq_names[scene_index]
        scene = self._scene(scene_name)
        if scene.frame_count != 24 or self.seq_len != 24:
            raise ValueError("MV-Kubric indexed loader requires exactly 24 frames (0..23)")
        if scene.invalid_frame_indices:
            return None
        seed = int(self.seed + virtual_index) if self.seed is not None else None
        rng = np.random.RandomState(seed)
        available_views = list(range(scene.view_count))
        if self.enable_variable_num_views_augs:
            probabilities = np.asarray(tuple(self.enable_variable_num_views_augs__n_views_probability.get(i, 0.0) for i in range(1, scene.view_count + 1)), dtype=np.float64)
            probabilities /= probabilities.sum()
            view_count = int(request.view_count) if request is not None and request.view_count is not None else int(rng.choice(np.arange(1, scene.view_count + 1), p=probabilities))
        else:
            view_count = len(available_views) if self.num_views == -1 else int(self.num_views)
        if not 1 <= view_count <= scene.view_count:
            raise ValueError(f"requested view count {view_count} is unavailable in {scene_name}")
        views = tuple(sorted(rng.choice(available_views, view_count, replace=False).tolist()))
        frame_indices = np.arange(24, dtype=np.int64)
        tracks_all = scene.tracks_3d
        visibility_all = scene.visibility[list(views)]
        preselected = _preselect_motion_tracks(tracks_all, visibility_all, rng, ratio_dynamic=float(self.ratio_dynamic), ratio_very_dynamic=float(self.ratio_very_dynamic), maximum=self.max_tracks_to_preload)
        if not len(preselected):
            return None
        tracks = tracks_all[:, preselected]
        visibility = visibility_all[:, :, preselected]
        intrinsics = np.repeat(scene.intrinsics[list(views), None], 24, axis=1).astype(np.float32)
        extrinsics = scene.extrinsics[list(views)].astype(np.float32)
        projected_xy, camera_z = _project(tracks, extrinsics, intrinsics)
        visibility &= np.isfinite(projected_xy).all(axis=-1) & np.isfinite(camera_z) & (camera_z > 0)
        source_size = tuple(scene.resolution_hw)
        augment = bool(self.augmentation_probability > 0 and rng.rand() <= self.augmentation_probability)
        apply_rgb_aug = bool(augment and self.enable_rgb_augs)
        rgb_augmentation = None
        if apply_rgb_aug:
            rgb_augmentation, visibility = _sample_rgb_augmentation(np.concatenate([projected_xy, camera_z[..., None]], axis=-1), visibility, *source_size, rng, eraser_probability=self.eraser_aug_prob, eraser_max=self.eraser_max, eraser_bounds=self.eraser_bounds, replace_probability=self.replace_aug_prob, replace_max=self.replace_max, replace_bounds=self.replace_bounds)
        output_size = tuple(self.crop_size) if self.enable_cropping_augs else source_size
        xy, visibility, intrinsics, theta = _spatial_transform(projected_xy, visibility, intrinsics, source_size, output_size, rng, self.enable_cropping_augs)
        transformed = np.concatenate([xy, camera_z[..., None]], axis=-1)
        apply_depth_aug = bool(augment and self.enable_depth_augs)
        depth_operations = ()
        if apply_depth_aug:
            depth_operations, visibility = _sample_depth_patch_operations(transformed, visibility, *output_size, rng, eraser_probability=self.eraser_aug_prob, eraser_max=self.eraser_max, eraser_bounds=self.eraser_bounds, replace_probability=self.replace_aug_prob, replace_max=self.replace_max, replace_bounds=self.replace_bounds)
        trajectory_cap = self.traj_per_sample
        if self.enable_variable_num_views_augs:
            trajectory_cap = int(trajectory_cap * self.enable_variable_num_views_augs__trajpersample_adjustment_factor.get(view_count, 1.0))
        selected, query_points = _sample_tracks(tracks, xy, camera_z, visibility, trajectory_cap, rng, augment_this_datapoint=augment, enable_variable_trajpersample_augs=self.enable_variable_trajpersample_augs, sample_index=virtual_index)
        if not len(selected):
            return None
        selected_global = preselected[selected]
        selected_tracks = tracks[:, selected]
        xy_z = transformed[:, :, selected]
        selected_visibility = visibility[:, :, selected]
        full_movement = _visible_path_lengths(tracks_all[:, selected_global], scene.visibility[:, :, selected_global])
        window_movement = full_movement.copy()
        depth_scale = 1.0
        if self.enable_scene_transform_augs:
            selected_tracks, query_points, xy_z, extrinsics, depth_scale = _scene_transform(selected_tracks, query_points, xy_z, extrinsics, rng)
        if self.enable_camera_params_noise_augs:
            intrinsics += rng.normal(0, 0.001, size=intrinsics.shape)
            extrinsics += rng.normal(0, 0.001, size=extrinsics.shape)
        media_indices = self._media_indices(scene_name, views)
        metadata = {"virtual_index": virtual_index, "scene_index": scene_index, "scene_name": scene_name, "seed": int(seed or 0), "window_start": 0, "window_end_exclusive": 24, "selected_views": list(views), "media_record_indices": list(media_indices), "depth_source": "gt", "gotit": True, "apply_rgb_aug": apply_rgb_aug, "apply_depth_aug": apply_depth_aug, "motion_track_count": int(len(selected_global)), "motion_full_mean_m": float(full_movement.mean()), "motion_full_median_m": float(np.median(full_movement)), "motion_full_p90_m": float(np.quantile(full_movement, 0.9)), "motion_window_mean_m": float(window_movement.mean()), "motion_window_median_m": float(np.median(window_movement)), "motion_window_p90_m": float(np.quantile(window_movement, 0.9))}
        return SamplePlan(dataset="kubric-dali", virtual_index=virtual_index, scene_index=scene_index, sequence=scene_name, seed=int(seed or 0), frame_indices=frame_indices, views=views, preselected_track_indices=preselected.copy(), selected_track_indices=selected.copy(), selected_global_track_indices=selected_global.copy(), track_count=int(len(selected)), query_points_3d=query_points, trajectory=xy_z, trajectory_3d=selected_tracks, visibility=selected_visibility, intrinsics=intrinsics, extrinsics=extrinsics, theta=theta, source_size=source_size, output_size=output_size, image_codec="dali", depth_source="gt", rgb_sources=(), depth_sources=(), apply_rgb_aug=apply_rgb_aug, rgb_augmentation=rgb_augmentation, apply_depth_aug=apply_depth_aug, depth_patch_operations=depth_operations, augmentation_seed=int(seed or 0), depth_scale=depth_scale, max_depth=float(self.max_depth), depth_sensor_widths=tuple(float(scene.sensor_widths[view]) for view in views), depth_focal_lengths=tuple(float(scene.focal_lengths[view]) for view in views), metadata=metadata, media_record_indices=media_indices)

    def materialize_sample(self, plan: SamplePlan):
        started = time.perf_counter()
        if len(plan.media_record_indices) != len(plan.views):
            raise ValueError("SamplePlan media record count does not match selected views")
        rgb_sources: list[bytes] = []
        depth_sources: list[bytes] = []
        for record_index in plan.media_record_indices:
            record = self._records.get(record_index)
            rgb = _packed_frames(record, "rgb.npz")
            depth = _packed_frames(record, "depth.npz")
            if len(rgb) != 24 or len(depth) != 24:
                raise ValueError(f"media record {record_index}: expected 24 RGB/depth frames")
            rgb_sources.extend(rgb)
            depth_sources.extend(depth)
        metadata = dict(plan.metadata)
        metadata["worker_prepare_seconds"] = time.perf_counter() - started
        metadata["media_record_count"] = len(plan.media_record_indices)
        metadata["encoded_bytes"] = sum(map(len, rgb_sources)) + sum(map(len, depth_sources))
        sample = EncodedTapVid3DSample(jpeg_bytes=tuple(rgb_sources), depth=None, theta=torch.from_numpy(plan.theta), intrs=torch.from_numpy(plan.intrinsics), extrs=torch.from_numpy(plan.extrinsics), trajectory=torch.from_numpy(plan.trajectory), trajectory_3d=torch.from_numpy(plan.trajectory_3d), visibility=torch.from_numpy(plan.visibility), valid=torch.ones((24, plan.track_count), dtype=torch.float32), query_points_3d=torch.from_numpy(plan.query_points_3d), seq_name=plan.sequence, metadata=metadata, output_size=plan.output_size, apply_rgb_aug=plan.apply_rgb_aug, rgb_augmentation=plan.rgb_augmentation, apply_depth_aug=plan.apply_depth_aug, augmentation_seed=plan.augmentation_seed, depth_scale=plan.depth_scale, track_upscaling_factor=1.0 / plan.depth_scale, max_depth=plan.max_depth, depth_patch_operations=plan.depth_patch_operations, image_codec=plan.image_codec, depth_bytes=tuple(depth_sources), depth_sensor_widths=plan.depth_sensor_widths, depth_focal_lengths=plan.depth_focal_lengths)
        return sample, True


__all__ = ["DaliKubricMultiViewDataset", "KubricSceneMetadata", "KubricWebDatasetCatalog"]
