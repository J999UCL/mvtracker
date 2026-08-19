"""MV-Kubric scene streaming through NVIDIA DALI WebDataset readers.

The WebDataset sample is one complete scene.  Metadata and all six encoded
views are read once by DALI; MVTracker's existing CPU sampling policy then
chooses the window, views and trajectories without reopening native files.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from mvtracker.datasets.kubric_multiview_dataset import (
    KubricMultiViewDataset,
    _legal_contiguous_window_starts,
)
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
WEB_DATASET_VERSION = 1
SOURCE_VIEW_COUNT = 10
COMPONENTS = ("meta.npz",) + tuple(
    f"rgb{view}.npz" for view in range(SOURCE_VIEW_COUNT)
) + tuple(
    f"depth{view}.npz" for view in range(SOURCE_VIEW_COUNT)
)


@dataclass(frozen=True)
class KubricSceneRecord:
    name: str
    tracks_3d: np.ndarray
    visibility: np.ndarray
    intrinsics: np.ndarray
    extrinsics: np.ndarray
    sensor_widths: np.ndarray
    focal_lengths: np.ndarray
    invalid_frame_indices: tuple[int, ...]
    resolution_hw: tuple[int, int]
    rgb_frames: tuple[tuple[bytes, ...], ...]
    depth_frames: tuple[tuple[bytes, ...], ...]

    @property
    def frame_count(self) -> int:
        return int(self.tracks_3d.shape[0])

    @property
    def view_count(self) -> int:
        return len(self.rgb_frames)


class KubricWebDatasetCatalog:
    """Small scene catalogue used by the shared Kubric sampler."""

    def __init__(self, manifest_path: str | Path):
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != WEB_DATASET_FORMAT:
            raise ValueError(f"{manifest_path}: unsupported WebDataset format")
        if int(manifest.get("version", -1)) != WEB_DATASET_VERSION:
            raise ValueError(f"{manifest_path}: unsupported WebDataset version")
        scenes = manifest.get("scenes")
        if not isinstance(scenes, dict) or not scenes:
            raise ValueError(f"{manifest_path}: manifest has no scenes")
        self.root = manifest_path.parent
        self.scenes = scenes
        self.source_fingerprint = str(manifest.get("source_fingerprint", "webdataset"))

    def scene(self, name: str):
        entry = dict(self.scenes[name])
        entry.setdefault("view_names", [f"view_{view}" for view in range(SOURCE_VIEW_COUNT)])
        entry.setdefault("n_frames", int(entry.get("frame_count", 24)))
        entry.setdefault("invalid_frame_indices", [])
        return entry, {}


def _bytes_from_dali(value) -> bytes:
    """Convert a batch-one DALI byte tensor to immutable bytes."""
    if hasattr(value, "at"):
        value = value.at(0)
    if hasattr(value, "as_array"):
        value = value.as_array()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return bytes(np.asarray(value, dtype=np.uint8).reshape(-1).tolist())


def _npz_bytes(value) -> dict[str, np.ndarray]:
    with np.load(BytesIO(_bytes_from_dali(value)), allow_pickle=False) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _packed_frames(value) -> tuple[bytes, ...]:
    packed = _npz_bytes(value)
    encoded = np.asarray(packed["bytes"], dtype=np.uint8).reshape(-1)
    offsets = np.asarray(packed["offsets"], dtype=np.int64)
    if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0:
        raise ValueError("packed WebDataset frames have invalid offsets")
    if offsets[-1] != encoded.size or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("packed WebDataset frames are truncated")
    return tuple(bytes(encoded[int(start) : int(end)]) for start, end in zip(offsets[:-1], offsets[1:]))


def _record_from_outputs(outputs: Sequence[Any]) -> KubricSceneRecord:
    if len(outputs) != len(COMPONENTS):
        raise ValueError(f"expected {len(COMPONENTS)} DALI components, got {len(outputs)}")
    meta = _npz_bytes(outputs[0])
    tracks_3d = np.asarray(meta["tracks_3d"], dtype=np.float32)
    visibility = np.asarray(meta["visibility"], dtype=np.bool_)
    intrinsics = np.asarray(meta["intrinsics"], dtype=np.float32)
    extrinsics = np.asarray(meta["extrinsics"], dtype=np.float32)
    if visibility.ndim != 3 or visibility.shape[1:] != tracks_3d.shape[:2]:
        raise ValueError("WebDataset metadata visibility does not match tracks_3d")
    view_count = visibility.shape[0]
    if view_count != SOURCE_VIEW_COUNT:
        raise ValueError(
            f"WebDataset scene must contain {SOURCE_VIEW_COUNT} views, got {view_count}"
        )
    for name, array in (
        ("intrinsics", intrinsics),
        ("extrinsics", extrinsics),
        ("sensor_widths", meta["sensor_widths"]),
        ("focal_lengths", meta["focal_lengths"]),
    ):
        if np.asarray(array).shape[0] != SOURCE_VIEW_COUNT:
            raise ValueError(f"WebDataset metadata {name} does not contain ten views")
    rgb = tuple(_packed_frames(outputs[1 + view]) for view in range(view_count))
    depth_offset = 1 + SOURCE_VIEW_COUNT
    depth = tuple(_packed_frames(outputs[depth_offset + view]) for view in range(view_count))
    if len(rgb) != view_count or any(len(frames) != tracks_3d.shape[0] for frames in rgb):
        raise ValueError("WebDataset RGB frame count does not match tracks_3d")
    scene_name_value = np.asarray(meta["scene_name"]).reshape(-1)[0].item()
    return KubricSceneRecord(
        name=str(scene_name_value),
        tracks_3d=tracks_3d,
        visibility=visibility,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        sensor_widths=np.asarray(meta["sensor_widths"], dtype=np.float32),
        focal_lengths=np.asarray(meta["focal_lengths"], dtype=np.float32),
        invalid_frame_indices=tuple(int(value) for value in np.asarray(meta.get("invalid_frame_indices", ())).reshape(-1)),
        resolution_hw=tuple(int(value) for value in np.asarray(meta["resolution_hw"]).reshape(-1)),
        rgb_frames=rgb,
        depth_frames=depth,
    )


class DaliKubricSceneStream(Iterator[KubricSceneRecord]):
    """Batch-one DALI WebDataset reader with indexed read-ahead."""

    def __init__(
        self,
        dataset_root: str | Path,
        *,
        split: str = "train",
        shard_id: int = 0,
        num_shards: int = 1,
        num_threads: int = 4,
        prefetch_queue_depth: int = 2,
        initial_fill: int = 32,
        random_shuffle: bool = True,
    ):
        try:
            import nvidia.dali.fn as fn
            from nvidia.dali import Pipeline
        except ImportError as error:
            raise RuntimeError("DALI WebDataset loading requires nvidia-dali-cuda120") from error
        root = Path(dataset_root) / split
        paths = sorted(root.glob("*.tar"))
        indices = sorted(root.glob("*.idx"))
        if not paths or len(paths) != len(indices):
            raise FileNotFoundError(f"{root}: expected matching WebDataset .tar and .idx shards")
        index_by_stem = {path.stem: path for path in indices}
        index_paths = [index_by_stem[path.stem] for path in paths]

        class _ScenePipeline(Pipeline):
            def __init__(self):
                super().__init__(batch_size=1, num_threads=num_threads, device_id=None,
                                 seed=0, exec_pipelined=True,
                                 prefetch_queue_depth=prefetch_queue_depth)

            def define_graph(self):
                return fn.readers.webdataset(
                    paths=[str(path) for path in paths],
                    index_paths=[str(path) for path in index_paths],
                    ext=list(COMPONENTS),
                    missing_component_behavior="error",
                    shard_id=shard_id,
                    num_shards=num_shards,
                    random_shuffle=random_shuffle,
                    initial_fill=initial_fill,
                    shuffle_after_epoch=True,
                )

        self._pipeline = _ScenePipeline()
        self._pipeline.build()

    def __iter__(self):
        return self

    def __next__(self) -> KubricSceneRecord:
        outputs = self._pipeline.run()
        return _record_from_outputs(outputs)


class DaliKubricMultiViewDataset(KubricMultiViewDataset):
    """MV-Kubric sampler backed by one DALI scene stream per DDP rank."""

    collate_fn = staticmethod(collate_encoded_tapvid3d)
    requires_cuda_prefetch = True
    requires_dali_scene_stream = True

    def __init__(self, *args, webdataset_root: str, webdataset_split: str = "train", **kwargs):
        manifest = Path(webdataset_root) / webdataset_split / "manifest.json"
        self.catalog = KubricWebDatasetCatalog(manifest)
        self.webdataset_root = str(webdataset_root)
        kwargs["data_root"] = str(Path(webdataset_root))
        kwargs["metadata_catalog"] = self.catalog
        kwargs["metadata_index_root"] = None
        super().__init__(*args, **kwargs)
        self._scene_stream: DaliKubricSceneStream | None = None

    def attach_scene_stream(self, stream: DaliKubricSceneStream) -> None:
        self._scene_stream = stream

    def _next_scene(self) -> KubricSceneRecord:
        if self._scene_stream is None:
            raise RuntimeError("DALI MV-Kubric dataset has no attached scene stream")
        return next(self._scene_stream)

    def plan_sample(self, index) -> SamplePlan | None:
        request = index if hasattr(index, "virtual_index") else None
        virtual_index = request.virtual_index if request is not None else int(index)
        scene = self._next_scene()
        scene_index = self.seq_names.index(scene.name) if scene.name in self.seq_names else virtual_index % self.real_len
        rng = np.random.RandomState(int(self.seed + virtual_index) if self.seed is not None else None)
        available_views = list(range(scene.view_count))
        if self.enable_variable_num_views_augs:
            probabilities = np.asarray(tuple(self.enable_variable_num_views_augs__n_views_probability.get(i, 0.0) for i in range(1, scene.view_count + 1)), dtype=np.float64)
            probabilities /= probabilities.sum()
            view_count = int(request.view_count) if request is not None and request.view_count is not None else int(rng.choice(np.arange(1, scene.view_count + 1), p=probabilities))
        else:
            view_count = len(available_views) if self.num_views == -1 else int(self.num_views)
        if not 1 <= view_count <= scene.view_count:
            raise ValueError(f"requested view count {view_count} is unavailable in {scene.name}")
        views = sorted(rng.choice(available_views, view_count, replace=False).tolist())
        legal = _legal_contiguous_window_starts(scene.frame_count, self.seq_len, scene.invalid_frame_indices)
        if not len(legal):
            return None
        start = int(rng.choice(legal))
        frame_indices = np.arange(start, start + self.seq_len)
        tracks_all = scene.tracks_3d
        visibility_all = scene.visibility[views]
        preselected = _preselect_motion_tracks(
            tracks_all, visibility_all, rng, ratio_dynamic=float(self.ratio_dynamic),
            ratio_very_dynamic=float(self.ratio_very_dynamic), maximum=self.max_tracks_to_preload,
        )
        if not len(preselected):
            return None
        tracks = tracks_all[frame_indices][:, preselected]
        visibility = visibility_all[:, frame_indices][:, :, preselected]
        intrinsics = np.repeat(scene.intrinsics[views, None], self.seq_len, axis=1).astype(np.float32)
        extrinsics = scene.extrinsics[views][:, frame_indices].astype(np.float32)
        projected_xy, camera_z = _project(tracks, extrinsics, intrinsics)
        visibility &= np.isfinite(projected_xy).all(axis=-1) & np.isfinite(camera_z) & (camera_z > 0)
        source_size = tuple(scene.resolution_hw)
        augment = bool(self.augmentation_probability > 0 and rng.rand() <= self.augmentation_probability)
        apply_rgb_aug = bool(augment and self.enable_rgb_augs)
        rgb_augmentation = None
        if apply_rgb_aug:
            rgb_augmentation, visibility = _sample_rgb_augmentation(
                np.concatenate([projected_xy, camera_z[..., None]], axis=-1), visibility,
                *source_size, rng, eraser_probability=self.eraser_aug_prob,
                eraser_max=self.eraser_max, eraser_bounds=self.eraser_bounds,
                replace_probability=self.replace_aug_prob, replace_max=self.replace_max,
                replace_bounds=self.replace_bounds,
            )
        output_size = tuple(self.crop_size) if self.enable_cropping_augs else source_size
        xy, visibility, intrinsics, theta = _spatial_transform(
            projected_xy, visibility, intrinsics, source_size, output_size, rng, self.enable_cropping_augs
        )
        transformed = np.concatenate([xy, camera_z[..., None]], axis=-1)
        apply_depth_aug = bool(augment and self.enable_depth_augs)
        depth_operations = ()
        if apply_depth_aug:
            depth_operations, visibility = _sample_depth_patch_operations(
                transformed, visibility, *output_size, rng,
                eraser_probability=self.eraser_aug_prob, eraser_max=self.eraser_max,
                eraser_bounds=self.eraser_bounds, replace_probability=self.replace_aug_prob,
                replace_max=self.replace_max, replace_bounds=self.replace_bounds,
            )
        trajectory_cap = self.traj_per_sample
        if self.enable_variable_num_views_augs:
            trajectory_cap = int(trajectory_cap * self.enable_variable_num_views_augs__trajpersample_adjustment_factor.get(view_count, 1.0))
        selected, query_points = _sample_tracks(
            tracks, xy, camera_z, visibility, trajectory_cap, rng,
            augment_this_datapoint=augment,
            enable_variable_trajpersample_augs=self.enable_variable_trajpersample_augs,
            sample_index=virtual_index,
        )
        if not len(selected):
            return None
        selected_global = preselected[selected]
        selected_tracks = tracks[:, selected]
        xy_z = transformed[:, :, selected]
        selected_visibility = visibility[:, :, selected]
        full_movement = _visible_path_lengths(tracks_all[:, selected_global], scene.visibility[:, :, selected_global])
        window_movement = _visible_path_lengths(tracks_all[frame_indices][:, selected_global], scene.visibility[:, frame_indices][:, :, selected_global])
        depth_scale = 1.0
        if self.enable_scene_transform_augs:
            selected_tracks, query_points, xy_z, extrinsics, depth_scale = _scene_transform(selected_tracks, query_points, xy_z, extrinsics, rng)
        if self.enable_camera_params_noise_augs:
            intrinsics += rng.normal(0, 0.001, size=intrinsics.shape)
            extrinsics += rng.normal(0, 0.001, size=extrinsics.shape)
        metadata = {
            "virtual_index": virtual_index, "scene_index": scene_index, "scene_name": scene.name,
            "seed": int(self.seed or 0) + virtual_index, "window_start": int(frame_indices[0]),
            "window_end_exclusive": int(frame_indices[-1] + 1), "selected_views": views,
            "depth_source": "gt", "requested_view_count": request.view_count if request is not None else None,
            "gotit": True, "apply_rgb_aug": apply_rgb_aug, "apply_depth_aug": apply_depth_aug,
            "motion_track_count": int(len(selected_global)),
            "motion_full_mean_m": float(full_movement.mean()), "motion_full_median_m": float(np.median(full_movement)),
            "motion_full_p90_m": float(np.quantile(full_movement, 0.9)),
            "motion_window_mean_m": float(window_movement.mean()), "motion_window_median_m": float(np.median(window_movement)),
            "motion_window_p90_m": float(np.quantile(window_movement, 0.9)),
        }
        rgb_sources = tuple(frame for view in views for frame in scene.rgb_frames[view][frame_indices[0] : frame_indices[-1] + 1])
        depth_sources = tuple(frame for view in views for frame in scene.depth_frames[view][frame_indices[0] : frame_indices[-1] + 1])
        return SamplePlan(
            dataset="kubric-dali", virtual_index=virtual_index, scene_index=scene_index,
            sequence=scene.name, seed=int(self.seed or 0) + virtual_index,
            frame_indices=frame_indices.copy(), views=tuple(views), preselected_track_indices=preselected.copy(),
            selected_track_indices=selected.copy(), selected_global_track_indices=selected_global.copy(), track_count=int(len(selected)),
            query_points_3d=query_points, trajectory=xy_z, trajectory_3d=selected_tracks,
            visibility=selected_visibility, intrinsics=intrinsics, extrinsics=extrinsics, theta=theta,
            source_size=source_size, output_size=output_size, image_codec="nvimagecodec", depth_source="gt",
            rgb_sources=rgb_sources, depth_sources=depth_sources, apply_rgb_aug=apply_rgb_aug,
            rgb_augmentation=rgb_augmentation, apply_depth_aug=apply_depth_aug,
            depth_patch_operations=depth_operations, augmentation_seed=int(self.seed or 0) + virtual_index,
            depth_scale=depth_scale, max_depth=float(self.max_depth),
            depth_sensor_widths=tuple(float(scene.sensor_widths[view]) for view in views),
            depth_focal_lengths=tuple(float(scene.focal_lengths[view]) for view in views), metadata=metadata,
        )

    def materialize_sample(self, plan: SamplePlan):
        sample = EncodedTapVid3DSample(
            jpeg_bytes=tuple(torch.frombuffer(bytearray(value), dtype=torch.uint8) for value in plan.rgb_sources),
            depth=None, theta=torch.from_numpy(plan.theta), intrs=torch.from_numpy(plan.intrinsics),
            extrs=torch.from_numpy(plan.extrinsics), trajectory=torch.from_numpy(plan.trajectory),
            trajectory_3d=torch.from_numpy(plan.trajectory_3d), visibility=torch.from_numpy(plan.visibility),
            valid=torch.ones((self.seq_len, plan.track_count), dtype=torch.float32),
            query_points_3d=torch.from_numpy(plan.query_points_3d), seq_name=plan.sequence, metadata=dict(plan.metadata),
            output_size=plan.output_size, apply_rgb_aug=plan.apply_rgb_aug, rgb_augmentation=plan.rgb_augmentation,
            apply_depth_aug=plan.apply_depth_aug, augmentation_seed=plan.augmentation_seed,
            depth_scale=plan.depth_scale, track_upscaling_factor=1.0 / plan.depth_scale,
            max_depth=plan.max_depth, depth_patch_operations=plan.depth_patch_operations,
            image_codec=plan.image_codec, depth_bytes=tuple(plan.depth_sources),
            depth_sensor_widths=plan.depth_sensor_widths, depth_focal_lengths=plan.depth_focal_lengths,
        )
        sample.metadata["gotit"] = True
        return sample, True


__all__ = [
    "COMPONENTS",
    "DaliKubricMultiViewDataset",
    "DaliKubricSceneStream",
    "KubricSceneRecord",
    "KubricWebDatasetCatalog",
    "SOURCE_VIEW_COUNT",
]
