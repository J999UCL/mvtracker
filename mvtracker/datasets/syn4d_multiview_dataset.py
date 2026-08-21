"""Syn4D sequence-cache loader for mixed MVTracker training."""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from mvtracker.datasets.kubric_multiview_dataset import _legal_contiguous_window_starts
from mvtracker.datasets.tapvid3d_multiview_dataset import (
    SamplePlan,
    TapVid3DMultiViewDataset,
    _project,
    _sample_depth_patch_operations,
    _sample_rgb_augmentation,
    _sample_tracks,
    _scene_transform,
    _spatial_transform,
    _visible_path_lengths,
)


_DATASET_PREFIX = "syn4d-multiview-"
_SPLITS = {"training": "train", "validation": "validation", "test": "test"}
class _MappedSequence:
    def __init__(self, root: Path):
        self.root = root
        self.arrays: dict[Path, np.ndarray] = {}
        self.lock = threading.Lock()

    def read(self, relative: str | Path, index=()) -> np.ndarray:
        path = self.root / relative
        with self.lock:
            array = self.arrays.get(path)
            if array is None:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                self.arrays[path] = array
        return np.array(array[index], copy=True)

    def close(self) -> None:
        for array in self.arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()


class _SequenceMmapCache:
    """Keep a fixed number of sequence stores open across lookahead workers."""

    def __init__(self, root: Path, maximum: int):
        if maximum < 1:
            raise ValueError("mmap_cache_sequences must be positive")
        self.root = root
        self.maximum = int(maximum)
        self.stores: OrderedDict[str, tuple[_MappedSequence, int]] = OrderedDict()
        self.condition = threading.Condition()

    @contextmanager
    def use(self, sequence: str):
        with self.condition:
            while sequence not in self.stores and len(self.stores) >= self.maximum:
                idle = next(
                    (name for name, (_, users) in self.stores.items() if users == 0),
                    None,
                )
                if idle is None:
                    self.condition.wait()
                    continue
                store, _ = self.stores.pop(idle)
                store.close()
            if sequence in self.stores:
                store, users = self.stores.pop(sequence)
            else:
                store, users = _MappedSequence(self.root / sequence), 0
            self.stores[sequence] = (store, users + 1)
        try:
            yield store
        finally:
            with self.condition:
                current, users = self.stores[sequence]
                self.stores[sequence] = (current, users - 1)
                self.condition.notify_all()


def _preselect_tracks(
    movement: np.ndarray,
    eligible: np.ndarray,
    rng: np.random.RandomState,
    *,
    ratio_dynamic: float,
    ratio_very_dynamic: float,
    maximum: int | None,
) -> np.ndarray:
    candidates = np.flatnonzero(eligible)
    static = candidates[movement[candidates] < 0.01]
    very_dynamic = candidates[movement[candidates] > 2.0]
    dynamic = candidates[movement[candidates] > 0.1]
    ratio_static = 1.0 - ratio_dynamic - ratio_very_dynamic
    target = min(
        len(candidates),
        int(len(dynamic) / ratio_dynamic) if ratio_dynamic else len(candidates),
        int(len(very_dynamic) / ratio_very_dynamic)
        if ratio_very_dynamic
        else len(candidates),
        int(len(static) / ratio_static) if ratio_static else len(candidates),
    )
    if maximum is not None:
        target = min(target, int(maximum))
    dynamic_count = int(target * ratio_dynamic)
    very_dynamic_count = int(target * ratio_very_dynamic)
    static_count = target - dynamic_count - very_dynamic_count
    selected = [
        rng.choice(dynamic, dynamic_count, replace=False),
        rng.choice(very_dynamic, very_dynamic_count, replace=False),
        rng.choice(static, static_count, replace=False),
    ]
    result = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
    rng.shuffle(result)
    return result.astype(np.int64, copy=False)


class Syn4DMultiViewDataset(TapVid3DMultiViewDataset):
    """Sample explicit Syn4D world tracks and indexed JPEG media."""

    def __init__(
        self,
        *args,
        data_root: str,
        view_count_probabilities: Sequence[float] | None = None,
        mmap_cache_sequences: int = 4,
        **kwargs,
    ):
        probabilities = tuple(view_count_probabilities or (1 / 6,) * 6)
        super().__init__(
            *args,
            data_root=data_root,
            raw_root=data_root,
            view_count_probabilities=probabilities,
            **kwargs,
        )
        self._sequence_cache = _SequenceMmapCache(
            Path(data_root), mmap_cache_sequences
        )

    def _load_manifest(self, sequence: str) -> dict[str, Any]:
        root = Path(self.data_root) / sequence
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        manifest["frame_count"] = int(manifest["frames"])
        manifest["point_count"] = int(manifest["tracks"])
        manifest["views"] = list(range(int(manifest["views"])))
        manifest["resolution_hw"] = list(manifest["cache_resolution"])
        return manifest

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
            raise ValueError(f"Unsupported Syn4D dataset name: {dataset_name}")
        requested = dataset_name[len(_DATASET_PREFIX):]
        if requested not in _SPLITS:
            raise ValueError(f"Unsupported Syn4D split: {requested}")
        kwargs = TapVid3DMultiViewDataset.from_name(
            f"tapvid3d-multiview-{requested}",
            dataset_root,
            training_args,
            fabric,
            just_return_kwargs=True,
            include_scene_ids=include_scene_ids,
            exclude_scene_ids=exclude_scene_ids,
        )
        datasets_cfg = getattr(training_args, "datasets", {}) if training_args else {}
        kwargs.pop("raw_root")
        kwargs.update(
            data_root=os.path.join(dataset_root, _SPLITS[requested]),
            num_views=int(datasets_cfg.get("syn4d_num_views", 6)),
            view_count_probabilities=tuple(
                datasets_cfg.get("syn4d_view_count_probabilities", (1 / 6,) * 6)
            ),
            mmap_cache_sequences=int(datasets_cfg.get("syn4d_mmap_cache_sequences", 4)),
            enable_variable_depth_type_augs=False,
            estimated_depth_root=None,
            estimated_depth_provider=None,
        )
        if requested == "training" and kwargs.get("enable_variable_num_views_augs"):
            kwargs["num_views"] = None
        if just_return_kwargs:
            return kwargs
        return Syn4DMultiViewDataset(**kwargs)

    def plan_sample(self, index) -> SamplePlan | None:
        request = index if hasattr(index, "virtual_index") else None
        virtual_index = request.virtual_index if request is not None else int(index)
        scene_index = (
            request.scene_index
            if request is not None and request.scene_index is not None
            else virtual_index % self.real_len
        )
        sequence = self.seq_names[scene_index]
        seed = (
            int(torch.randint(0, 2**32 - 1, ()).item())
            if self.seed is None
            else int(self.seed + virtual_index if self.add_index_to_seed else self.seed)
        )
        rng = np.random.RandomState(seed)
        manifest = self._manifest(sequence)
        frame_count = int(manifest["frame_count"])
        views_available = list(manifest["views"])

        if self.enable_variable_num_views_augs:
            maximum = min(6, len(views_available))
            if request is not None and request.view_count is not None:
                view_count = int(request.view_count)
            else:
                probabilities = np.asarray(
                    self.view_count_probabilities[:maximum], dtype=np.float64
                )
                probabilities /= probabilities.sum()
                view_count = int(rng.choice(np.arange(1, maximum + 1), p=probabilities))
            if not 1 <= view_count <= maximum:
                raise ValueError(f"requested view count must be in [1, {maximum}]")
        else:
            view_count = len(views_available) if self.num_views == -1 else int(self.num_views)
        views = sorted(rng.choice(views_available, view_count, replace=False).tolist())
        legal = _legal_contiguous_window_starts(frame_count, self.seq_len)
        if not len(legal):
            raise ValueError(f"{sequence}: fewer than {self.seq_len} frames")
        start = int(rng.choice(legal))
        stop = start + self.seq_len
        frame_indices = np.arange(start, stop)

        with self._sequence_cache.use(sequence) as store:
            track_valid = store.read("track_valid.npy", np.s_[start:stop])
            visibility = np.stack(
                [store.read(f"{view}/visibility.npy", np.s_[start:stop]) for view in views]
            )
            visibility &= track_valid[None]
            any_view = visibility.any(axis=0)
            eligible = (any_view.sum(axis=0) >= 2) & (
                any_view[0] | any_view[len(any_view) // 2]
            )
            movement = store.read("motion_path_length.npy")
            preselected = _preselect_tracks(
                movement,
                eligible,
                rng,
                ratio_dynamic=float(self.ratio_dynamic),
                ratio_very_dynamic=float(self.ratio_very_dynamic),
                maximum=self.max_tracks_to_preload,
            )
            if not len(preselected):
                return None
            tracks = store.read(
                "tracks_xyz.npy", (slice(start, stop), preselected, slice(None))
            )
            selected_valid = track_valid[:, preselected]
            selected_visibility = visibility[:, :, preselected]
            intrinsics = np.stack(
                [store.read(f"{view}/intrinsics.npy", np.s_[start:stop]) for view in views]
            ).astype(np.float32, copy=False)
            extrinsics = np.stack(
                [store.read(f"{view}/extrinsics_w2c.npy", np.s_[start:stop, :3, :4]) for view in views]
            ).astype(np.float32, copy=False)

        xy, camera_z = _project(tracks, extrinsics, intrinsics)
        selected_visibility &= (
            np.isfinite(xy).all(axis=-1)
            & np.isfinite(camera_z)
            & (camera_z > 0)
        )
        augment = bool(
            self.augmentation_probability > 0
            and rng.rand() <= self.augmentation_probability
        )
        apply_rgb_aug = bool(augment and self.enable_rgb_augs)
        rgb_augmentation = None
        source_size = tuple(int(value) for value in manifest["resolution_hw"])
        if apply_rgb_aug:
            rgb_augmentation, selected_visibility = _sample_rgb_augmentation(
                np.concatenate([xy, camera_z[..., None]], axis=-1),
                selected_visibility,
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
        xy, selected_visibility, intrinsics, theta = _spatial_transform(
            xy,
            selected_visibility,
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
            depth_operations, selected_visibility = _sample_depth_patch_operations(
                transformed,
                selected_visibility,
                *output_size,
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
            selected_visibility,
            min(int(self.traj_per_sample), 2048),
            rng,
            augment_this_datapoint=augment,
            enable_variable_trajpersample_augs=self.enable_variable_trajpersample_augs,
            sample_index=virtual_index,
        )
        if not len(selected):
            return None
        selected_global = preselected[selected]
        tracks = tracks[:, selected]
        query_points = query_points.astype(np.float32, copy=False)
        trajectory = transformed[:, :, selected]
        visibility = selected_visibility[:, :, selected]
        validity = selected_valid[:, selected]
        full_movement = movement[selected_global]
        window_movement = _visible_path_lengths(tracks, visibility)
        depth_scale = 1.0
        if self.enable_scene_transform_augs:
            tracks, query_points, trajectory, extrinsics, depth_scale = _scene_transform(
                tracks, query_points, trajectory, extrinsics, rng
            )
        if self.enable_camera_params_noise_augs:
            intrinsics += rng.normal(0, 0.001, size=intrinsics.shape)
            extrinsics += rng.normal(0, 0.001, size=extrinsics.shape)
        metadata = {
            "virtual_index": virtual_index,
            "scene_index": scene_index,
            "scene_name": sequence,
            "seed": seed,
            "window_start": start,
            "window_end_exclusive": stop,
            "selected_views": views,
            "requested_view_count": request.view_count if request is not None else None,
            "depth_source": "gt",
            "gotit": True,
            "apply_rgb_aug": apply_rgb_aug,
            "apply_depth_aug": apply_depth_aug,
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
        root = Path(self.data_root) / sequence
        return SamplePlan(
            dataset="syn4d",
            virtual_index=virtual_index,
            scene_index=scene_index,
            sequence=sequence,
            seed=seed,
            frame_indices=frame_indices,
            views=tuple(views),
            preselected_track_indices=preselected,
            selected_track_indices=selected,
            selected_global_track_indices=selected_global,
            track_count=len(selected),
            query_points_3d=query_points,
            trajectory=trajectory,
            trajectory_3d=tracks.astype(np.float32, copy=False),
            visibility=visibility,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
            theta=theta,
            source_size=source_size,
            output_size=output_size,
            image_codec="jpeg",
            depth_source="gt",
            rgb_sources=tuple(
                (
                    root / f"view_{view}" / "jpeg_bytes.bin",
                    root / f"view_{view}" / "jpeg_offsets.npy",
                )
                for view in views
            ),
            depth_sources=tuple(root / str(view) / "depth.npy" for view in views),
            apply_rgb_aug=apply_rgb_aug,
            rgb_augmentation=rgb_augmentation,
            apply_depth_aug=apply_depth_aug,
            depth_patch_operations=depth_operations,
            augmentation_seed=seed,
            depth_scale=depth_scale,
            max_depth=float(self.max_depth),
            depth_sensor_widths=(),
            depth_focal_lengths=(),
            metadata=metadata,
            track_validity=validity,
        )

    def materialize_sample(self, plan: SamplePlan):
        return super().materialize_sample(plan)
