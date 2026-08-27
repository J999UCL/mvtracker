"""MV-Kubric live sampling backed by DALI's native WebDataset reader."""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from io import BytesIO
import json
import os
from pathlib import Path
import threading
import time
from typing import Any, Mapping

import numpy as np
import torch

from mvtracker.datasets.estimated_depth import sample_depth_source
from mvtracker.datasets.io_cache import discard_file_range
from mvtracker.datasets.kubric_dali_stream import (
    KubricDaliSceneBundle,
    KubricDaliSceneGroup,
    KubricDaliSceneStream,
    KubricDaliSceneOrder,
)
from mvtracker.datasets.kubric_multiview_dataset import KubricMultiViewDataset
from mvtracker.preprocessing.mvkubric_webdataset import (
    META_COMPONENT,
    RGB_COMPONENT,
    RECORD_LOCATOR_FORMAT,
    parse_dali_index,
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
    _apply_spatial_trace,
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
class IndexedReadStats:
    requested_bytes: int
    read_bytes: int
    seconds: float
    record_count: int


class _IndexedRecordStore:
    """Read global WebDataset records directly from indexed TAR byte ranges."""

    local_cache_bytes = 0

    def __init__(self, locator_path: str | Path):
        started = time.perf_counter()
        locator_path = Path(locator_path).resolve()
        self.root = locator_path.parent
        with np.load(locator_path, allow_pickle=False) as locator:
            if str(locator["format"].item()) != RECORD_LOCATOR_FORMAT:
                raise ValueError(f"{locator_path}: unsupported record locator format")
            self.shard_paths = tuple(self.root / str(path) for path in locator["shards"])
            self.keys = tuple(str(key) for key in locator["keys"])
            self.record_shards = np.asarray(locator["record_shards"], dtype=np.int32).copy()
            self.component_names = tuple(str(name) for name in locator["component_names"])
            self.offsets = np.asarray(locator["offsets"], dtype=np.int64).copy()
            self.sizes = np.asarray(locator["sizes"], dtype=np.int64).copy()
        self._fds: dict[int, int] = {}
        self._fd_lock = threading.Lock()
        self._pid = os.getpid()
        self.build_seconds = time.perf_counter() - started

    def __len__(self) -> int:
        return len(self.keys)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_fds"] = {}
        state["_fd_lock"] = None
        state["_pid"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._fds = {}
        self._fd_lock = threading.Lock()
        self._pid = os.getpid()

    def _fd(self, shard: int) -> int:
        pid = os.getpid()
        with self._fd_lock:
            if pid != self._pid:
                for descriptor in self._fds.values():
                    os.close(descriptor)
                self._fds.clear()
                self._pid = pid
            descriptor = self._fds.get(shard)
            if descriptor is None:
                descriptor = os.open(self.shard_paths[shard], os.O_RDONLY)
                self._fds[shard] = descriptor
            return descriptor

    def read_many(
        self,
        indices: tuple[int, ...] | list[int],
        components: set[str] | None = None,
    ) -> tuple[tuple[Mapping[str, Any], ...], IndexedReadStats]:
        started = time.perf_counter()
        requested = tuple(int(index) for index in indices)
        records: list[dict[str, Any]] = []
        reads: list[tuple[int, int, int, int, str]] = []
        requested_bytes = 0
        for position, index in enumerate(requested):
            if not 0 <= index < len(self):
                raise IndexError(f"record index {index} is outside [0, {len(self)})")
            records.append({"__key__": self.keys[index]})
            shard = int(self.record_shards[index])
            for component, offset, size in zip(
                self.component_names, self.offsets[index], self.sizes[index]
            ):
                if offset >= 0 and (components is None or component in components):
                    reads.append((shard, int(offset), int(size), position, component))
                    requested_bytes += int(size)
        read_bytes = 0
        for shard, offset, size, position, component in sorted(reads):
            payload = os.pread(self._fd(shard), size, offset)
            if len(payload) != size:
                raise OSError(
                    f"short indexed read from {self.shard_paths[shard]} at {offset}: "
                    f"expected {size} bytes, read {len(payload)}"
                )
            records[position][f".{component}"] = payload
            read_bytes += len(payload)
            discard_file_range(self._fd(shard), offset, size)
        return tuple(records), IndexedReadStats(
            requested_bytes=requested_bytes,
            read_bytes=read_bytes,
            seconds=time.perf_counter() - started,
            record_count=len(requested),
        )

    def read(self, index: int) -> tuple[Mapping[str, Any], IndexedReadStats]:
        records, stats = self.read_many((index,))
        return records[0], stats


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
        locator = manifest.get("record_locator")
        self.record_locator_path = (
            None if locator is None else (manifest_path.parent / str(locator)).resolve()
        )

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
        name=bundle.scene_name or str(np.asarray(meta["scene_name"]).item()),
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
    """Plan arbitrary scenes and read selected views from indexed TAR ranges."""

    collate_fn = staticmethod(collate_encoded_tapvid3d)
    requires_cuda_prefetch = True
    _scene_reuse_passes = 1
    _fixed_views = None

    def execution_recipe_source(self, plan: SamplePlan) -> dict[str, Any]:
        entry, _ = self.catalog.scene(plan.sequence)
        return {"metadata_index": int(entry["metadata_index"])}
    _seed_by_scene = False
    _stream_start_offset = 0

    def _scene_index(self, scene_name: str) -> int:
        indices = getattr(self, "_scene_indices", None)
        if indices is None:
            indices = {
                name: index for index, name in enumerate(self.seq_names)
            }
            self._scene_indices = indices
        return int(indices[scene_name])

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
        sequential_stream: bool = False,
        **kwargs,
    ):
        manifest_path = Path(webdataset_root) / webdataset_split / "manifest.json"
        self.catalog = KubricWebDatasetCatalog(manifest_path)
        kwargs["data_root"] = str(Path(webdataset_root))
        kwargs["metadata_catalog"] = self.catalog
        kwargs["metadata_index_root"] = None
        kwargs["validate_scene_directories"] = False
        super().__init__(*args, **kwargs)
        self._scene_indices = {
            scene_name: index for index, scene_name in enumerate(self.seq_names)
        }
        if int(scene_reuse_passes) != 1:
            raise ValueError("MV-Kubric DALI scenes must be consumed once per epoch")
        self._scene_reuse_passes = 1
        self._fixed_views = fixed_views
        self._seed_by_scene = bool(seed_by_scene)
        self.depth_provider = str(depth_provider)
        self._sequential_stream = bool(sequential_stream)
        self._streamed_scenes: deque[
            tuple[KubricDaliSceneBundle, KubricDaliSceneGroup, int, int]
        ] = deque()
        if self._sequential_stream:
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
            self._stream_start_offset = int(stream_start_request_cursor) % requests_per_group
        else:
            locator = self.catalog.record_locator_path
            if locator is None or not locator.is_file():
                raise FileNotFoundError(f"record locator is missing from {manifest_path}")
            self._records = _IndexedRecordStore(locator)
            self.stream = self._records

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
        if not getattr(self, "_sequential_stream", True):
            return self._plan_indexed_sample(index)
        return self._plan_stream_sample(index)

    def _plan_stream_sample(self, index) -> SamplePlan | None:
        request = index if hasattr(index, "virtual_index") else None
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
        plan = self._plan_scene_metadata(request or int(index), scene)
        if plan is None:
            return None
        rgb_sources = tuple(
            frame for view in plan.views for frame in _packed_frames(bundle.rgb_npz[view])
        )
        depth_sources = tuple(
            frame for view in plan.views for frame in _packed_frames(bundle.depth_npz[view])
        )
        metadata = {
            **plan.metadata,
            "record_store": "dali-webdataset",
            "dali_batch_index": group.batch_index,
            "dali_read_seconds": group.read_seconds,
            "dali_payload_bytes": group.payload_bytes,
            "dali_scene_position": scene_position,
            "dali_reuse_pass": reuse_pass,
        }
        return replace(
            plan,
            rgb_sources=rgb_sources,
            depth_sources=depth_sources,
            metadata=metadata,
        )

    def _plan_indexed_sample(self, index) -> SamplePlan | None:
        request = index if hasattr(index, "virtual_index") else None
        virtual_index = int(request.virtual_index) if request is not None else int(index)
        expected_scene = (
            getattr(request, "expected_scene", None) if request is not None else None
        )
        requested_index = (
            getattr(request, "scene_index", None) if request is not None else None
        )
        if expected_scene is not None:
            scene_name = str(expected_scene)
            if scene_name not in self.seq_names:
                raise KeyError(f"planned scene {scene_name!r} is unavailable")
            scene_index = self._scene_index(scene_name)
            if requested_index is not None and int(requested_index) != scene_index:
                raise RuntimeError(
                    f"planned scene/index diverged: {scene_name!r} != {requested_index}"
                )
        elif requested_index is not None:
            scene_index = int(requested_index)
            if not 0 <= scene_index < self.real_len:
                raise IndexError(f"scene index {scene_index} is outside [0, {self.real_len})")
            scene_name = self.seq_names[scene_index]
        else:
            scene_index = virtual_index % self.real_len
            scene_name = self.seq_names[scene_index]

        entry, _ = self.catalog.scene(scene_name)
        metadata_index = int(entry["metadata_index"])
        record, metadata_read = self._records.read(metadata_index)
        expected_key = f"scene-{scene_name}"
        if record["__key__"] != expected_key:
            raise RuntimeError(
                f"metadata locator diverged: expected {expected_key!r}, got {record['__key__']!r}"
            )
        scene = _scene_metadata(
            KubricDaliSceneBundle(scene_name, record[f".{META_COMPONENT}"], (), ())
        )
        plan = self._plan_scene_metadata(request or virtual_index, scene)
        if plan is None:
            return None
        media_indices = tuple(
            int(entry["views"][str(view)]["media_index"])
            for view in plan.views
        )
        metadata = {
            **plan.metadata,
            "record_store": "indexed-webdataset",
            "media_record_indices": list(media_indices),
            "indexed_metadata_requested_bytes": metadata_read.requested_bytes,
            "indexed_metadata_read_bytes": metadata_read.read_bytes,
            "indexed_metadata_read_seconds": metadata_read.seconds,
        }
        return replace(plan, media_record_indices=media_indices, metadata=metadata)

    def _plan_scene_metadata(self, index, scene: KubricSceneMetadata) -> SamplePlan | None:
        request = index if hasattr(index, "virtual_index") else None
        virtual_index = request.virtual_index if request is not None else int(index)
        if scene.invalid_frame_indices:
            return None
        scene_index = self._scene_index(scene.name)
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
        spatial_trace: list[np.ndarray] = []
        xy, visibility, intrinsics, theta = _spatial_transform(
            projected_xy,
            visibility,
            intrinsics,
            source_size,
            output_size,
            rng,
            self.enable_cropping_augs,
            trace_out=spatial_trace,
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
        post_selection_rng_state = rng.get_state()
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
            "record_store": "metadata-only",
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
            rgb_sources=(),
            depth_sources=(),
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
            post_selection_rng_state=post_selection_rng_state,
            spatial_transform=spatial_trace[0],
        )

    def materialize_sample(self, plan: SamplePlan, runtime_depth=None):
        payload_depth_source = "gt" if self.depth_provider == "gt" else "estimated"
        if runtime_depth is None and plan.depth_source != payload_depth_source:
            raise RuntimeError(
                f"planned {plan.depth_source} depth but DALI payload contains "
                f"{payload_depth_source} depth"
            )
        started = time.perf_counter()
        rgb_sources = plan.rgb_sources
        depth_sources = plan.depth_sources
        metadata = dict(plan.metadata)
        if plan.media_record_indices:
            if len(plan.media_record_indices) != len(plan.views):
                raise ValueError("media record count does not match selected views")
            records, media_read = self._records.read_many(
                plan.media_record_indices,
                components={RGB_COMPONENT} if runtime_depth is not None else None,
            )
            rgb_payloads: list[bytes] = []
            depth_payloads: list[bytes] = []
            for view, record in zip(plan.views, records):
                expected_key = f"scene-{plan.sequence}-view-{view:02d}"
                if record["__key__"] != expected_key:
                    raise RuntimeError(
                        f"media locator diverged: expected {expected_key!r}, "
                        f"got {record['__key__']!r}"
                    )
                rgb_payloads.extend(_packed_frames(record[".rgb.npz"]))
                if runtime_depth is None:
                    depth_payloads.extend(_packed_frames(record[".depth.npz"]))
            rgb_sources = tuple(rgb_payloads)
            depth_sources = tuple(depth_payloads)
            metadata.update(
                indexed_media_requested_bytes=media_read.requested_bytes,
                indexed_media_read_bytes=media_read.read_bytes,
                indexed_media_read_seconds=media_read.seconds,
                indexed_requested_bytes=(
                    int(metadata["indexed_metadata_requested_bytes"])
                    + media_read.requested_bytes
                ),
                indexed_read_bytes=(
                    int(metadata["indexed_metadata_read_bytes"])
                    + media_read.read_bytes
                ),
                indexed_read_seconds=(
                    float(metadata["indexed_metadata_read_seconds"])
                    + media_read.seconds
                ),
            )
        metadata["worker_prepare_seconds"] = time.perf_counter() - started
        metadata["encoded_bytes"] = sum(map(len, rgb_sources)) + sum(map(len, depth_sources))
        runtime_depth_tensor = None
        if runtime_depth is not None:
            estimated_depths = runtime_depth.depth
            cleaned_mask = runtime_depth.cleaned_mask
            expected = (len(plan.views), len(plan.frame_indices), *plan.source_size)
            if estimated_depths.shape != expected:
                raise ValueError(
                    f"runtime depth shape {estimated_depths.shape} does not match {expected}"
                )
            if plan.depth_source == "estimated_cleaned":
                estimated_depths = estimated_depths * cleaned_mask
            runtime_depth_tensor = torch.from_numpy(
                np.asarray(estimated_depths, dtype=np.float32).copy()
            )[:, :, None]
            metadata["runtime_depth_ready_wait_seconds"] = float(
                runtime_depth.ready_wait_seconds
            )
            metadata["runtime_depth_read_seconds"] = float(runtime_depth.read_seconds)
            metadata["runtime_depth_delete_seconds"] = float(runtime_depth.delete_seconds)
            metadata["runtime_depth_bytes"] = int(runtime_depth.byte_count)
        sample = EncodedTapVid3DSample(
            jpeg_bytes=rgb_sources,
            depth=runtime_depth_tensor,
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
            depth_bytes=depth_sources,
            depth_sensor_widths=plan.depth_sensor_widths,
            depth_focal_lengths=plan.depth_focal_lengths,
        )
        return sample, True

    def execution_plan(self, execution_record) -> SamplePlan:
        """Read selected MV-Kubric metadata without invoking its planner."""
        from .training_recipe import unpack_execution_mask

        record = execution_record.recipe
        trace = execution_record.trace
        if getattr(self, "_records", None) is None:
            scene = self._recipe_metadata[record.scene]
            metadata_read = IndexedReadStats(0, 0, 0.0, 0)
        else:
            metadata_record, metadata_read = self._records.read(
                int(trace["source"]["metadata_index"])
            )
            expected_key = f"scene-{record.scene}"
            if metadata_record["__key__"] != expected_key:
                raise RuntimeError(
                    f"metadata locator diverged: expected {expected_key!r}, got {metadata_record['__key__']!r}"
                )
            scene = _scene_metadata(
                KubricDaliSceneBundle(
                    record.scene,
                    metadata_record[f".{META_COMPONENT}"],
                    (),
                    (),
                )
            )
        frames = np.asarray(record.frames, dtype=np.int64)
        views = tuple(record.views)
        selected_global = np.asarray(record.tracks, dtype=np.int64)
        tracks = np.asarray(
            scene.tracks_3d[np.ix_(frames, selected_global)], dtype=np.float32
        )
        intrinsics = np.repeat(
            scene.intrinsics[list(views), None], len(frames), axis=1
        ).astype(np.float32)
        extrinsics = scene.extrinsics[list(views)][:, frames].astype(np.float32)
        xy, camera_z = _project(tracks, extrinsics, intrinsics)
        theta = np.asarray(trace["theta"], dtype=np.float32)
        xy, intrinsics = _apply_spatial_trace(
            xy, intrinsics, np.asarray(trace["spatial_transform"], dtype=np.float64)
        )
        trajectory = np.concatenate([xy, camera_z[..., None]], axis=-1)
        query_times = np.asarray(trace["query_times"], dtype=np.int64)
        query_points = np.concatenate(
            [query_times[:, None].astype(np.float32), tracks[query_times, np.arange(record.track_count)]],
            axis=1,
        )
        rng = np.random.RandomState()
        rng.set_state(trace["post_selection_rng_state"])
        depth_scale = 1.0
        if self.enable_scene_transform_augs:
            tracks, query_points, trajectory, extrinsics, depth_scale = _scene_transform(
                tracks, query_points, trajectory, extrinsics, rng
            )
        if self.enable_camera_params_noise_augs:
            intrinsics += rng.normal(0, 0.001, size=intrinsics.shape)
            extrinsics += rng.normal(0, 0.001, size=extrinsics.shape)
        augmentation = record.augmentation
        metadata = {
            **trace["metadata"],
            "indexed_metadata_requested_bytes": metadata_read.requested_bytes,
            "indexed_metadata_read_bytes": metadata_read.read_bytes,
            "indexed_metadata_read_seconds": metadata_read.seconds,
        }
        return SamplePlan(
            dataset=str(trace["dataset"]),
            virtual_index=int(record.request["virtual_index"]),
            scene_index=record.scene_index,
            sequence=record.scene,
            seed=record.seed,
            frame_indices=frames,
            views=views,
            preselected_track_indices=selected_global,
            selected_track_indices=np.arange(record.track_count, dtype=np.int64),
            selected_global_track_indices=selected_global,
            track_count=record.track_count,
            query_points_3d=query_points.astype(np.float32, copy=False),
            trajectory=trajectory.astype(np.float32, copy=False),
            trajectory_3d=tracks.astype(np.float32, copy=False),
            visibility=unpack_execution_mask(trace["visibility"]),
            intrinsics=intrinsics.astype(np.float32, copy=False),
            extrinsics=extrinsics.astype(np.float32, copy=False),
            theta=theta,
            source_size=tuple(trace["source_size"]),
            output_size=tuple(trace["output_size"]),
            image_codec=str(trace["image_codec"]),
            depth_source=record.depth_source,
            rgb_sources=(),
            depth_sources=(),
            apply_rgb_aug=bool(augmentation["apply_rgb"]),
            rgb_augmentation=augmentation["rgb"],
            apply_depth_aug=bool(augmentation["apply_depth"]),
            depth_patch_operations=tuple(tuple(item) for item in augmentation["depth_patch_operations"]),
            augmentation_seed=int(augmentation["seed"]),
            depth_scale=float(depth_scale),
            max_depth=float(trace["max_depth"]),
            depth_sensor_widths=tuple(trace["depth_sensor_widths"]),
            depth_focal_lengths=tuple(trace["depth_focal_lengths"]),
            metadata=metadata,
            media_record_indices=tuple(trace["media_record_indices"]),
            post_selection_rng_state=trace["post_selection_rng_state"],
        )

    def materialize_recipe_record(self, execution_record, runtime_depth=None):
        return self.materialize_sample(
            self.execution_plan(execution_record), runtime_depth=runtime_depth
        )


class DaliKubricRecipePlanner(DaliKubricMultiViewDataset):
    """CPU-only MV-Kubric planner following the exact live DALI scene order."""

    requires_cuda_prefetch = False

    def __init__(
        self,
        *args,
        webdataset_root: str,
        webdataset_split: str = "train",
        stream_world_size: int = 1,
        stream_seed: int | None = None,
        stream_scenes_per_batch: int = 4,
        stream_shuffle_shards: bool = True,
        stream_include_scene_ids: tuple[str, ...] | None = None,
        stream_start_request_cursor: int = 0,
        fixed_views: tuple[int, ...] | None = None,
        seed_by_scene: bool = False,
        **kwargs,
    ):
        manifest_path = Path(webdataset_root) / webdataset_split / "manifest.json"
        self.catalog = KubricWebDatasetCatalog(manifest_path)
        kwargs["data_root"] = str(Path(webdataset_root))
        kwargs["metadata_index_root"] = None
        kwargs["metadata_catalog"] = self.catalog
        kwargs["validate_scene_directories"] = False
        KubricMultiViewDataset.__init__(self, *args, **kwargs)
        self._scene_indices = {
            scene_name: index for index, scene_name in enumerate(self.seq_names)
        }
        resolved_seed = self.seed if stream_seed is None else stream_seed
        self._recipe_world_size = int(stream_world_size)
        self._recipe_start_cursor = int(stream_start_request_cursor)
        self._recipe_scenes_per_batch = int(stream_scenes_per_batch)
        self._recipe_orders = tuple(
            KubricDaliSceneOrder(
                manifest_path,
                rank=rank,
                world_size=self._recipe_world_size,
                seed=int(0 if resolved_seed is None else resolved_seed),
                shuffle_shards=stream_shuffle_shards,
                include_scene_ids=stream_include_scene_ids,
            )
            for rank in range(self._recipe_world_size)
        )
        self._recipe_metadata = {}
        self._fixed_views = fixed_views
        self._seed_by_scene = bool(seed_by_scene)

    def preload_recipe_metadata(self, workers: int = 16) -> None:
        shards = tuple(
            dict.fromkeys(
                shard
                for order in self._recipe_orders
                for shard in order.assigned
            )
        )
        selected = set(self.seq_names)

        def load_shard(shard):
            records = parse_dali_index(shard.index)
            scenes = {}
            with shard.archive.open("rb") as archive:
                for record in records:
                    component = next(
                        (
                            item
                            for item in record.components
                            if item.extension == META_COMPONENT
                        ),
                        None,
                    )
                    if component is None:
                        continue
                    archive.seek(component.offset)
                    payload = archive.read(component.size)
                    scene = _scene_metadata(
                        KubricDaliSceneBundle("", payload, (), ())
                    )
                    if scene.name in selected:
                        scenes[scene.name] = scene
            return scenes

        started = time.perf_counter()
        print(
            "RECIPE_METADATA event=start "
            f"scenes={len(selected)} shards={len(shards)} workers={workers}",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            futures = {
                executor.submit(load_shard, shard): shard
                for shard in shards
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                self._recipe_metadata.update(future.result())
                if completed % 25 == 0 or completed == len(shards):
                    elapsed = time.perf_counter() - started
                    print(
                        "RECIPE_METADATA event=progress "
                        f"shards={completed}/{len(shards)} "
                        f"scenes={len(self._recipe_metadata)}/{len(selected)} "
                        f"rate={len(self._recipe_metadata) / max(elapsed, 1e-9):.1f}_scenes_per_second",
                        flush=True,
                    )

    def resolve_recipe_request(self, request, *, rank=None, local_cursor=None):
        if rank is None:
            rank = int(request.virtual_index) % self._recipe_world_size
        if local_cursor is None:
            local_cursor = int(request.virtual_index) // self._recipe_world_size
        cursor = self._recipe_start_cursor + int(local_cursor)
        order = self._recipe_orders[int(rank)]
        first_epoch = order.recipe_scene_names(
            0, self._recipe_scenes_per_batch
        )
        epoch, position = divmod(cursor, len(first_epoch))
        scene_name = order.recipe_scene_names(
            epoch, self._recipe_scenes_per_batch
        )[position]
        return replace(
            request,
            scene_index=self._scene_index(scene_name),
            expected_scene=scene_name,
        )

    def plan_sample(self, request) -> SamplePlan | None:
        resolved = self.resolve_recipe_request(request)
        scene = self._recipe_metadata[resolved.expected_scene]
        return self._plan_scene_metadata(resolved, scene)


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
            sequential_stream=True,
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
    "DaliKubricRecipePlanner",
    "DaliKubricValidationDataset",
    "IndexedReadStats",
    "KubricSceneMetadata",
    "KubricWebDatasetCatalog",
]
