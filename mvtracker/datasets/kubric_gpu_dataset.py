"""Lossless GPU image decoding for native MV-Kubric training samples."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mvtracker.datasets.kubric_multiview_dataset import (
    KubricMultiViewDataset,
    _legal_contiguous_window_starts,
)
from mvtracker.datasets.tapvid3d_multiview_dataset import (
    EncodedTapVid3DSample,
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


class GpuDecodedKubricMultiViewDataset(KubricMultiViewDataset):
    """Prepare MV-Kubric metadata on CPU and decode PNG/TIFF batches on CUDA."""

    collate_fn = staticmethod(collate_encoded_tapvid3d)
    requires_cuda_prefetch = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.metadata_index is None:
            raise ValueError("MV-Kubric GPU decode requires kubric_metadata_index_root")
        if self.use_duster_depths or self.clean_duster_depths:
            raise ValueError("MV-Kubric GPU decode supports native GT depth only")
        if self.estimated_depth_store.enabled or self.enable_variable_depth_type_augs:
            raise ValueError("MV-Kubric GPU decode does not support estimated depth substitution")
        if self.novel_views is not None or self.normalize_scene_following_vggt:
            raise ValueError("MV-Kubric GPU decode does not support novel-view/VGGT modes")

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
        scene_path = Path(self.data_root) / sequence
        scene, arrays = self.metadata_index.scene(sequence)

        with np.load(scene_path / "tracks_3d.npz") as payload:
            tracks_all = np.asarray(payload["tracks_3d"], dtype=np.float32)
        available_views = list(range(len(scene["view_names"])))
        if self.enable_variable_num_views_augs:
            view_counts = self.enable_variable_num_views_augs__n_views_probability
            view_count = (
                int(request.view_count)
                if request is not None and request.view_count is not None
                else int(rng.choice(tuple(view_counts), p=tuple(view_counts.values())))
            )
        else:
            view_count = len(available_views) if self.num_views == -1 else int(self.num_views)
        if not 1 <= view_count <= len(available_views):
            raise ValueError(f"requested view count {view_count} is unavailable for {scene_path}")
        views = sorted(rng.choice(available_views, view_count, replace=False).tolist())

        legal = _legal_contiguous_window_starts(
            len(tracks_all), self.seq_len, scene["invalid_frame_indices"]
        )
        if not len(legal):
            raise ValueError(f"{scene_path}: no valid {self.seq_len}-frame window")
        start = int(rng.choice(legal))
        frame_indices = np.arange(start, start + self.seq_len)

        tracks_2d_all = []
        occlusion_all = []
        for view in views:
            view_path = scene_path / scene["view_names"][view]
            with np.load(view_path / "tracks_2d.npz") as payload:
                tracks_2d_all.append(np.asarray(payload["tracks_2d"]))
                occlusion_all.append(np.asarray(payload["occlusion"], dtype=np.bool_))
        visibility_all = ~np.stack(occlusion_all)
        preselected = _preselect_motion_tracks(
            tracks_all,
            visibility_all,
            rng,
            ratio_dynamic=float(self.ratio_dynamic),
            ratio_very_dynamic=float(self.ratio_very_dynamic),
            maximum=self.max_tracks_to_preload,
        )
        if not len(preselected):
            return None, False

        tracks = tracks_all[frame_indices][:, preselected]
        visibility = visibility_all[:, frame_indices][:, :, preselected]
        trajectory_2d = np.stack(tracks_2d_all)[:, frame_indices][:, :, preselected]
        intrinsics = np.repeat(
            np.asarray(arrays["intrinsics"])[views, None], self.seq_len, axis=1
        ).astype(np.float32)
        extrinsics = np.asarray(arrays["extrinsics"])[views][:, frame_indices].astype(np.float32)
        projected_xy, camera_z = _project(tracks, extrinsics, intrinsics)
        if self.perform_sanity_checks and not np.allclose(
            projected_xy, trajectory_2d, atol=1e-3
        ):
            raise ValueError(f"{scene_path}: indexed 2D tracks do not match camera projection")
        visibility &= (
            np.isfinite(projected_xy).all(axis=-1)
            & np.isfinite(camera_z)
            & (camera_z > 0)
        )

        first_rgb = scene_path / scene["view_names"][views[0]] / scene["rgba_files"][views[0]][start]
        with Image.open(first_rgb) as image:
            source_size = (int(image.height), int(image.width))
        rgb_bytes = []
        depth_bytes = []
        for view in views:
            view_path = scene_path / scene["view_names"][view]
            rgb_bytes.extend(
                (view_path / scene["rgba_files"][view][frame]).read_bytes()
                for frame in frame_indices
            )
            depth_bytes.extend(
                (view_path / scene["depth_files"][view][frame]).read_bytes()
                for frame in frame_indices
            )

        augment = bool(
            self.augmentation_probability > 0
            and rng.rand() <= self.augmentation_probability
        )
        apply_rgb_aug = bool(augment and self.enable_rgb_augs)
        rgb_augmentation = None
        if apply_rgb_aug:
            pre_crop = np.concatenate([projected_xy, camera_z[..., None]], axis=-1)
            rgb_augmentation, visibility = _sample_rgb_augmentation(
                pre_crop,
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
        transformed_trajectory = np.concatenate([xy, camera_z[..., None]], axis=-1)
        apply_depth_aug = bool(augment and self.enable_depth_augs)
        depth_operations = ()
        if apply_depth_aug:
            depth_operations, visibility = _sample_depth_patch_operations(
                transformed_trajectory,
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
            return None, False

        selected_tracks = tracks[:, selected]
        xy_z = transformed_trajectory[:, :, selected]
        selected_visibility = visibility[:, :, selected]
        selected_global = preselected[selected]
        full_movement = _visible_path_lengths(
            tracks_all[:, selected_global], visibility_all[:, :, selected_global]
        )
        window_movement = _visible_path_lengths(
            tracks_all[frame_indices][:, selected_global],
            visibility_all[:, frame_indices][:, :, selected_global],
        )
        depth_scale = 1.0
        if self.enable_scene_transform_augs:
            selected_tracks, query_points, xy_z, extrinsics, depth_scale = _scene_transform(
                selected_tracks, query_points, xy_z, extrinsics, rng
            )
        if self.enable_camera_params_noise_augs:
            intrinsics += rng.normal(0, 0.001, size=intrinsics.shape)
            extrinsics += rng.normal(0, 0.001, size=extrinsics.shape)

        sample = EncodedTapVid3DSample(
            jpeg_bytes=tuple(rgb_bytes),
            depth=None,
            theta=torch.from_numpy(theta),
            intrs=torch.from_numpy(intrinsics),
            extrs=torch.from_numpy(extrinsics),
            trajectory=torch.from_numpy(xy_z),
            trajectory_3d=torch.from_numpy(selected_tracks),
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
                "depth_source": "gt",
                "gotit": True,
                "worker_prepare_seconds": time.perf_counter() - load_started,
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
            },
            output_size=output_size,
            apply_rgb_aug=apply_rgb_aug,
            rgb_augmentation=rgb_augmentation,
            apply_depth_aug=apply_depth_aug,
            augmentation_seed=seed,
            depth_scale=depth_scale,
            track_upscaling_factor=1.0 / depth_scale,
            max_depth=float(self.max_depth),
            depth_patch_operations=depth_operations,
            image_codec="nvimagecodec",
            depth_bytes=tuple(depth_bytes),
            depth_sensor_widths=tuple(float(arrays["sensor_widths"][view]) for view in views),
            depth_focal_lengths=tuple(float(arrays["focal_lengths"][view]) for view in views),
        )
        return sample, True
