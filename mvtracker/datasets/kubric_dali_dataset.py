"""MV-Kubric live sampling backed by DALI's native WebDataset reader."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
import torch

from mvtracker.datasets.estimated_depth import sample_depth_source
from mvtracker.datasets.kubric_dali_stream import (
    KubricDaliSceneBundle,
    KubricDaliSceneGroup,
    KubricDaliSceneStream,
)
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


def _npz_bytes(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(BytesIO(payload), allow_pickle=False) as packed:
        return {key: np.asarray(packed[key]) for key in packed.files}


def _packed_frames(payload: bytes) -> tuple[bytes, ...]:
    packed = _npz_bytes(payload)
    encoded = np.asarray(packed["bytes"], dtype=np.uint8).reshape(-1)
    offsets = np.asarray(packed["offsets"], dtype=np.int64).reshape(-1)
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
    """Read the scene catalog without opening a data shard."""

    def __init__(self, manifest_path: str | Path):
        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != WEB_DATASET_FORMAT:
            raise ValueError(f"{manifest_path}: unsupported WebDataset format")
        self.scenes = manifest["scenes"]
        self.source_fingerprint = str(manifest.get("source_fingerprint", "webdataset"))

    def scene(self, name: str):
        entry = dict(self.scenes[name])
        view_map = entry["views"]
        entry["view_names"] = entry.get("view_names") or [
            f"view_{int(view):02d}" for view in sorted(map(int, view_map))
        ]
        entry.setdefault("n_frames", 24)
        entry.setdefault("invalid_frame_indices", [])
        return entry, {}


def _scene_metadata(bundle: KubricDaliSceneBundle) -> KubricSceneMetadata:
    meta = _npz_bytes(bundle.metadata_npz)
    return KubricSceneMetadata(
        name=bundle.scene_name,
        tracks_3d=np.asarray(meta["tracks_3d"], dtype=np.float32),
        visibility=np.asarray(meta["visibility"], dtype=np.bool_),
        intrinsics=np.asarray(meta["intrinsics"], dtype=np.float32),
        extrinsics=np.asarray(meta["extrinsics"], dtype=np.float32),
        sensor_widths=np.asarray(meta["sensor_widths"], dtype=np.float32),
        focal_lengths=np.asarray(meta["focal_lengths"], dtype=np.float32),
        invalid_frame_indices=tuple(
            int(value)
            for value in np.asarray(meta.get("invalid_frame_indices", ())).reshape(-1)
        ),
        resolution_hw=tuple(int(value) for value in np.asarray(meta["resolution_hw"]).reshape(-1)),
    )


class DaliKubricMultiViewDataset(KubricMultiViewDataset):
    """Apply the existing live sampler to scenes streamed directly by DALI."""

    collate_fn = staticmethod(collate_encoded_tapvid3d)
    requires_cuda_prefetch = True
    _scene_reuse_passes = 1
    _fixed_views = None
    _seed_by_scene = False
    _stream_start_offset = 0

    def __init__(
        self,
        *args,
        webdataset_root: str,
        webdataset_split: str = "train",
        stream_rank: int = 0,
        stream_world_size: int = 1,
        stream_seed: int | None = None,
        stream_scenes_per_batch: int = 4,
        stream_repeat: bool = True,
        stream_shuffle_shards: bool = True,
        stream_include_scene_ids: tuple[str, ...] | None = None,
        stream_allow_empty: bool = False,
        scene_reuse_passes: int = 1,
        fixed_views: tuple[int, ...] | None = None,
        seed_by_scene: bool = False,
        stream_start_request_cursor: int = 0,
        depth_provider: str = "gt",
        **kwargs,
    ):
        manifest_path = Path(webdataset_root) / webdataset_split / "manifest.json"
        self.catalog = KubricWebDatasetCatalog(manifest_path)
        kwargs["data_root"] = str(Path(webdataset_root))
        kwargs["metadata_catalog"] = self.catalog
        kwargs["metadata_index_root"] = None
        super().__init__(*args, **kwargs)
        resolved_stream_seed = self.seed if stream_seed is None else stream_seed
        requests_per_group = int(stream_scenes_per_batch) * int(scene_reuse_passes)
        self.stream = KubricDaliSceneStream(
            manifest_path,
            rank=stream_rank,
            world_size=stream_world_size,
            seed=int(0 if resolved_stream_seed is None else resolved_stream_seed),
            scenes_per_batch=stream_scenes_per_batch,
            repeat=stream_repeat,
            shuffle_shards=stream_shuffle_shards,
            include_scene_ids=stream_include_scene_ids,
            allow_empty=stream_allow_empty,
            start_group_index=int(stream_start_request_cursor) // requests_per_group,
        )
        if int(scene_reuse_passes) != 1:
            raise ValueError("MV-Kubric DALI scenes must be consumed once per epoch")
        self._scene_reuse_passes = 1
        self._fixed_views = fixed_views
        self._seed_by_scene = bool(seed_by_scene)
        self.depth_provider = str(depth_provider)
        self._stream_start_offset = int(stream_start_request_cursor) % requests_per_group
        self._streamed_scenes: deque[
            tuple[KubricDaliSceneBundle, KubricDaliSceneGroup, int, int]
        ] = deque()

    def _next_scene(self):
        if not self._streamed_scenes:
            group = self.stream.next_scene_group()
            for reuse_pass in range(self._scene_reuse_passes):
                self._streamed_scenes.extend(
                    (scene, group, position, reuse_pass)
                    for position, scene in enumerate(group.scenes)
                )
            while self._stream_start_offset:
                self._streamed_scenes.popleft()
                self._stream_start_offset -= 1
        return self._streamed_scenes.popleft()

    def plan_sample(self, index) -> SamplePlan | None:
        request = index if hasattr(index, "virtual_index") else None
        virtual_index = request.virtual_index if request is not None else int(index)
        bundle, group, scene_position, reuse_pass = self._next_scene()
        scene = _scene_metadata(bundle)
        expected_scene = (
            getattr(request, "expected_scene", None) if request is not None else None
        )
        if expected_scene is not None:
            if scene.name != expected_scene:
                raise RuntimeError(
                    "DALI recipe scene diverged: "
                    f"expected {expected_scene!r}, got {scene.name!r}"
                )
        if scene.invalid_frame_indices:
            return None
        scene_index = self.seq_names.index(scene.name)
        seed_index = scene_index if self._seed_by_scene else virtual_index
        seed = int(self.seed + seed_index) if self.seed is not None else None
        rng = np.random.RandomState(seed)
        depth_source = sample_depth_source(
            rng,
            variable=getattr(self, "enable_variable_depth_type_augs", False),
            replay_depth_source=(
                getattr(request, "depth_source", None) if request is not None else None
            ),
        )

        if self._fixed_views is not None:
            views = self._fixed_views
            view_count = len(views)
        elif self.enable_variable_num_views_augs:
            probabilities = np.asarray(
                tuple(
                    self.enable_variable_num_views_augs__n_views_probability.get(i, 0.0)
                    for i in range(1, scene.view_count + 1)
                ),
                dtype=np.float64,
            )
            probabilities /= probabilities.sum()
            view_count = (
                int(request.view_count)
                if request is not None and request.view_count is not None
                else int(rng.choice(np.arange(1, scene.view_count + 1), p=probabilities))
            )
            views = tuple(
                sorted(rng.choice(scene.view_count, view_count, replace=False).tolist())
            )
        else:
            view_count = scene.view_count if self.num_views == -1 else int(self.num_views)
            views = tuple(sorted(rng.choice(scene.view_count, view_count, replace=False).tolist()))

        frame_indices = np.arange(24, dtype=np.int64)
        tracks_all = scene.tracks_3d
        visibility_all = scene.visibility[list(views)]
        preselected = _preselect_motion_tracks(
            tracks_all,
            visibility_all,
            rng,
            ratio_dynamic=float(self.ratio_dynamic),
            ratio_very_dynamic=float(self.ratio_very_dynamic),
            maximum=self.max_tracks_to_preload,
        )
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
            rgb_augmentation, visibility = _sample_rgb_augmentation(
                np.concatenate([projected_xy, camera_z[..., None]], axis=-1),
                visibility,
                *source_size,
                rng,
                eraser_probability=self.eraser_aug_prob,
                eraser_max=self.eraser_max,
                eraser_bounds=self.eraser_bounds,
                replace_probability=self.replace_aug_prob,
                replace_max=self.replace_max,
                replace_bounds=self.replace_bounds,
            )
        output_size = tuple(self.crop_size) if self.enable_cropping_augs else source_size
        xy, visibility, intrinsics, theta = _spatial_transform(
            projected_xy,
            visibility,
            intrinsics,
            source_size,
            output_size,
            rng,
            self.enable_cropping_augs,
        )
        transformed = np.concatenate([xy, camera_z[..., None]], axis=-1)
        apply_depth_aug = bool(augment and self.enable_depth_augs)
        depth_operations = ()
        if apply_depth_aug:
            depth_operations, visibility = _sample_depth_patch_operations(
                transformed,
                visibility,
                *output_size,
                rng,
                eraser_probability=self.eraser_aug_prob,
                eraser_max=self.eraser_max,
                eraser_bounds=self.eraser_bounds,
                replace_probability=self.replace_aug_prob,
                replace_max=self.replace_max,
                replace_bounds=self.replace_bounds,
            )

        trajectory_cap = self.traj_per_sample
        if self.enable_variable_num_views_augs:
            trajectory_cap = int(
                trajectory_cap
                * self.enable_variable_num_views_augs__trajpersample_adjustment_factor.get(
                    view_count, 1.0
                )
            )
        selected, query_points = _sample_tracks(
            tracks,
            xy,
            camera_z,
            visibility,
            trajectory_cap,
            rng,
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
        movement = _visible_path_lengths(
            tracks_all[:, selected_global], scene.visibility[:, :, selected_global]
        )
        depth_scale = 1.0
        if self.enable_scene_transform_augs:
            selected_tracks, query_points, xy_z, extrinsics, depth_scale = _scene_transform(
                selected_tracks, query_points, xy_z, extrinsics, rng
            )
        if self.enable_camera_params_noise_augs:
            intrinsics += rng.normal(0, 0.001, size=intrinsics.shape)
            extrinsics += rng.normal(0, 0.001, size=extrinsics.shape)

        rgb_sources = tuple(
            frame for view in views for frame in _packed_frames(bundle.rgb_npz[view])
        )
        depth_sources = tuple(
            frame for view in views for frame in _packed_frames(bundle.depth_npz[view])
        )
        metadata = {
            "virtual_index": virtual_index,
            "scene_index": scene_index,
            "scene_name": scene.name,
            "seed": int(seed or 0),
            "window_start": 0,
            "window_end_exclusive": 24,
            "selected_views": list(views),
            "depth_source": depth_source,
            "gotit": True,
            "record_store": "dali-webdataset",
            "dali_batch_index": group.batch_index,
            "dali_read_seconds": group.read_seconds,
            "dali_payload_bytes": group.payload_bytes,
            "dali_scene_position": scene_position,
            "dali_reuse_pass": reuse_pass,
            "apply_rgb_aug": apply_rgb_aug,
            "apply_depth_aug": apply_depth_aug,
            "motion_track_count": int(len(selected_global)),
            "motion_full_mean_m": float(movement.mean()),
            "motion_full_median_m": float(np.median(movement)),
            "motion_full_p90_m": float(np.quantile(movement, 0.9)),
            "motion_full_static_count": int((movement < 0.01).sum()),
            "motion_full_dynamic_count": int((movement > 0.1).sum()),
            "motion_full_very_dynamic_count": int((movement > 2.0).sum()),
            "motion_window_mean_m": float(movement.mean()),
            "motion_window_median_m": float(np.median(movement)),
            "motion_window_p90_m": float(np.quantile(movement, 0.9)),
            "motion_window_static_count": int((movement < 0.01).sum()),
            "motion_window_dynamic_count": int((movement > 0.1).sum()),
            "motion_window_very_dynamic_count": int((movement > 2.0).sum()),
            "motion_full_dynamic_window_static_count": 0,
        }
        return SamplePlan(
            dataset="kubric-dali",
            virtual_index=virtual_index,
            scene_index=scene_index,
            sequence=scene.name,
            seed=int(seed or 0),
            frame_indices=frame_indices,
            views=views,
            preselected_track_indices=preselected.copy(),
            selected_track_indices=selected.copy(),
            selected_global_track_indices=selected_global.copy(),
            track_count=int(len(selected)),
            query_points_3d=query_points,
            trajectory=xy_z,
            trajectory_3d=selected_tracks,
            visibility=selected_visibility,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            theta=theta,
            source_size=source_size,
            output_size=output_size,
            image_codec="dali",
            depth_source=depth_source,
            rgb_sources=rgb_sources,
            depth_sources=depth_sources,
            apply_rgb_aug=apply_rgb_aug,
            rgb_augmentation=rgb_augmentation,
            apply_depth_aug=apply_depth_aug,
            depth_patch_operations=depth_operations,
            augmentation_seed=int(seed or 0),
            depth_scale=depth_scale,
            max_depth=float(self.max_depth),
            depth_sensor_widths=tuple(float(scene.sensor_widths[view]) for view in views),
            depth_focal_lengths=tuple(float(scene.focal_lengths[view]) for view in views),
            metadata=metadata,
        )

    def materialize_sample(self, plan: SamplePlan):
        payload_depth_source = "gt" if self.depth_provider == "gt" else "estimated"
        if plan.depth_source != payload_depth_source:
            raise RuntimeError(
                f"planned {plan.depth_source} depth but DALI payload contains "
                f"{payload_depth_source} depth"
            )
        started = time.perf_counter()
        metadata = dict(plan.metadata)
        metadata["worker_prepare_seconds"] = time.perf_counter() - started
        metadata["encoded_bytes"] = sum(map(len, plan.rgb_sources)) + sum(map(len, plan.depth_sources))
        sample = EncodedTapVid3DSample(
            jpeg_bytes=plan.rgb_sources,
            depth=None,
            theta=torch.from_numpy(plan.theta),
            intrs=torch.from_numpy(plan.intrinsics),
            extrs=torch.from_numpy(plan.extrinsics),
            trajectory=torch.from_numpy(plan.trajectory),
            trajectory_3d=torch.from_numpy(plan.trajectory_3d),
            visibility=torch.from_numpy(plan.visibility),
            valid=torch.ones((24, plan.track_count), dtype=torch.float32),
            query_points_3d=torch.from_numpy(plan.query_points_3d),
            seq_name=plan.sequence,
            metadata=metadata,
            output_size=plan.output_size,
            apply_rgb_aug=plan.apply_rgb_aug,
            rgb_augmentation=plan.rgb_augmentation,
            apply_depth_aug=plan.apply_depth_aug,
            augmentation_seed=plan.augmentation_seed,
            depth_scale=plan.depth_scale,
            track_upscaling_factor=1.0 / plan.depth_scale,
            max_depth=plan.max_depth,
            depth_patch_operations=plan.depth_patch_operations,
            image_codec=plan.image_codec,
            depth_bytes=plan.depth_sources,
            depth_sensor_widths=plan.depth_sensor_widths,
            depth_focal_lengths=plan.depth_focal_lengths,
        )
        return sample, True


class DaliKubricValidationDataset(DaliKubricMultiViewDataset):
    """Finite, unaugmented MV-Kubric validation stream for CUDA decoding."""

    dali_loader_num_workers = 0

    def __init__(
        self,
        *args,
        webdataset_root: str,
        include_scene_ids: tuple[str, ...] | list[str] | None = None,
        views: tuple[int, ...] = (0, 1, 2, 3),
        stream_rank: int = 0,
        stream_world_size: int = 1,
        stream_seed: int = 72,
        **kwargs,
    ):
        selected_scenes = (
            None if include_scene_ids is None else tuple(map(str, include_scene_ids))
        )
        fixed_views = tuple(map(int, views))
        kwargs.update(
            include_scene_ids=selected_scenes,
            views_to_return=list(fixed_views),
            num_views=-1,
            augmentation_probability=0.0,
            enable_rgb_augs=False,
            enable_depth_augs=False,
            enable_cropping_augs=False,
            enable_variable_trajpersample_augs=False,
            enable_scene_transform_augs=False,
            enable_camera_params_noise_augs=False,
            enable_variable_depth_type_augs=False,
            enable_variable_num_views_augs=False,
            normalize_scene_following_vggt=False,
            enable_variable_vggt_crop_size_augs=False,
        )
        super().__init__(
            *args,
            webdataset_root=webdataset_root,
            webdataset_split="validation",
            stream_rank=stream_rank,
            stream_world_size=stream_world_size,
            stream_seed=stream_seed,
            stream_scenes_per_batch=1,
            stream_repeat=True,
            stream_shuffle_shards=False,
            stream_include_scene_ids=selected_scenes,
            stream_allow_empty=True,
            scene_reuse_passes=1,
            fixed_views=fixed_views,
            seed_by_scene=True,
            **kwargs,
        )
        self.virtual_len = self.stream.local_scene_count
        self.local_scene_names = self.stream.local_scene_names

    def __getitem__(self, index):
        plan = self.plan_sample(int(index))
        if plan is None:
            raise RuntimeError("MV-Kubric validation scene did not produce a sample")
        return self.materialize_sample(plan)


__all__ = [
    "DaliKubricMultiViewDataset",
    "DaliKubricValidationDataset",
    "KubricSceneMetadata",
    "KubricWebDatasetCatalog",
]
