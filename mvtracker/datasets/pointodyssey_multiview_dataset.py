# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from mvtracker.datasets.kubric_multiview_dataset import KubricMultiViewDataset
from mvtracker.datasets.utils import read_json


_DATASET_PREFIX = "pointodyssey-multiview-"
_SPLITS = {
    "training": "train",
    "validation": "validation",
    "test": "test",
}
_SCHEMA_VERSION = 4
_FORMAT_NAME = "pointodyssey_mvtracker_preprocessed"
_VIEW_IDS = (0, 1, 2, 3)
_HEIGHT = 384
_WIDTH = 512
_POINT_COUNT = 2600


def _require_dict(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object.")
    return value


def _require_value(value: Any, expected: Any, context: str) -> None:
    if value != expected:
        raise ValueError(f"{context} must be {expected!r}, got {value!r}.")


def _load_npy(path: Path, expected_shape: tuple[int, ...], expected_dtype: np.dtype) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required PointOdyssey array is missing: {path}")
    try:
        array = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Could not read PointOdyssey array: {path}") from exc
    if not isinstance(array, np.ndarray):
        raise ValueError(f"{path} is not an NPY array.")
    if array.shape != expected_shape:
        raise ValueError(f"{path} has shape {array.shape}, expected {expected_shape}.")
    if array.dtype != np.dtype(expected_dtype):
        raise ValueError(f"{path} has dtype {array.dtype}, expected {np.dtype(expected_dtype)}.")
    return array


def _read_rgb_frames(view_path: Path, frame_count: int) -> torch.Tensor:
    expected_names = [f"rgba_{frame:05d}.jpg" for frame in range(frame_count)]
    actual_names = sorted(path.name for path in view_path.glob("rgba_*.jpg"))
    if actual_names != expected_names:
        raise ValueError(
            f"{view_path} does not contain the exact contiguous RGB sequence "
            f"rgba_00000.jpg through rgba_{frame_count - 1:05d}.jpg."
        )

    frames = []
    for frame, name in enumerate(expected_names):
        path = view_path / name
        try:
            with Image.open(path) as image:
                if image.format != "JPEG":
                    raise ValueError(f"decoded format is {image.format!r}, expected 'JPEG'")
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        except Exception as exc:
            raise ValueError(
                f"Could not decode prepared PointOdyssey RGB view={view_path.name} "
                f"frame={frame}: {path}"
            ) from exc
        if rgb.shape != (_HEIGHT, _WIDTH, 3):
            raise ValueError(
                f"{path} decodes to shape {rgb.shape}, expected {(_HEIGHT, _WIDTH, 3)}."
            )
        frames.append(rgb)
    return torch.from_numpy(np.stack(frames, axis=0))


def _project_tracks(
    tracks_3d: np.ndarray,
    extrinsics_w2c: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    homogeneous = np.concatenate(
        [tracks_3d, np.ones_like(tracks_3d[..., :1])],
        axis=-1,
    )
    camera = np.einsum("fij,fnj->fni", extrinsics_w2c, homogeneous)
    pixels_homogeneous = np.einsum("ij,fnj->fni", intrinsics, camera)
    with np.errstate(divide="ignore", invalid="ignore"):
        tracks_2d = pixels_homogeneous[..., :2] / pixels_homogeneous[..., 2:]
    return np.asarray(tracks_2d, dtype=np.float32)


class PointOdysseyMultiViewDataset(KubricMultiViewDataset):
    """MV-Tracker's Kubric sampling policy over prepared PointOdyssey scenes."""

    def _motion_bucket_ratios(self, total_tracks, very_dynamic_tracks):
        """Reassign an undersupplied very-dynamic quota to dynamic tracks."""
        target_tracks = total_tracks
        if self.max_tracks_to_preload is not None:
            target_tracks = min(target_tracks, self.max_tracks_to_preload)
        if self.traj_per_sample is not None:
            target_tracks = min(target_tracks, self.traj_per_sample)

        required_very_dynamic = int(np.ceil(target_tracks * self.ratio_very_dynamic))
        if very_dynamic_tracks < required_very_dynamic:
            return self.ratio_dynamic + self.ratio_very_dynamic, 0.0
        return self.ratio_dynamic, self.ratio_very_dynamic

    @staticmethod
    def from_name(
        dataset_name: str,
        dataset_root: str,
        training_args=None,
        fabric=None,
        just_return_kwargs: bool = False,
    ):
        if not dataset_name.startswith(_DATASET_PREFIX):
            raise ValueError(f"Unsupported PointOdyssey dataset name: {dataset_name!r}.")
        requested_split = dataset_name[len(_DATASET_PREFIX):]
        if requested_split not in _SPLITS:
            supported = ", ".join(f"{_DATASET_PREFIX}{name}" for name in _SPLITS)
            raise ValueError(
                f"Unsupported PointOdyssey dataset name {dataset_name!r}; expected one of: {supported}."
            )

        if requested_split == "training":
            if training_args is None or fabric is None:
                raise ValueError("PointOdyssey training requires training_args and fabric.")
            kwargs = KubricMultiViewDataset.from_name(
                "kubric-multiview-v3-training",
                dataset_root,
                training_args=training_args,
                fabric=fabric,
                just_return_kwargs=True,
            )
        else:
            kwargs = KubricMultiViewDataset.from_name(
                "kubric-multiview-v3",
                dataset_root,
                training_args=training_args,
                just_return_kwargs=True,
                subset=_SPLITS[requested_split],
            )

        unsupported = []
        if kwargs["use_duster_depths"] or kwargs["clean_duster_depths"]:
            unsupported.append("Duster depth")
        if kwargs.get("enable_variable_depth_type_augs", False):
            unsupported.append("variable-depth augmentation")
        if kwargs.get("enable_variable_num_views_augs", False):
            unsupported.append("variable-view augmentation")
        if unsupported:
            raise ValueError(
                "Prepared PointOdyssey provides fixed four-view GT depth only; disable "
                + ", ".join(unsupported)
                + "."
            )

        prepared_split = _SPLITS[requested_split]
        if requested_split == "training" and training_args.modes.debug:
            prepared_split = "validation"
        kwargs.update({
            "data_root": os.path.join(
                dataset_root,
                "PointOdyssey_MVTracker",
                prepared_split,
            ),
            "num_views": 4,
            "views_to_return": None,
            "novel_views": None,
            "use_duster_depths": False,
            "clean_duster_depths": False,
            "duster_views": None,
            "supported_duster_views_sets": None,
            "enable_variable_depth_type_augs": False,
            "enable_variable_num_views_augs": False,
        })
        if just_return_kwargs:
            return kwargs
        return PointOdysseyMultiViewDataset(**kwargs)

    @staticmethod
    def getitem_raw_datapoint(scene_path, perform_2d_projection_sanity_check=True):
        """Read the prepared contract; all Kubric sampling stays in the parent class."""
        del perform_2d_projection_sanity_check  # Tracks are projected here from the stored cameras.

        scene_path = Path(scene_path)
        metadata_path = scene_path / "scene.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Required PointOdyssey metadata is missing: {metadata_path}")
        try:
            metadata = _require_dict(read_json(str(metadata_path)), str(metadata_path))
        except Exception as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"Could not read PointOdyssey metadata: {metadata_path}") from exc

        _require_value(metadata.get("schema_version"), _SCHEMA_VERSION, f"{metadata_path}: schema_version")
        _require_value(metadata.get("format"), _FORMAT_NAME, f"{metadata_path}: format")
        _require_value(metadata.get("scene_id"), scene_path.name, f"{metadata_path}: scene_id")
        _require_value(metadata.get("split"), scene_path.parent.name, f"{metadata_path}: split")

        output = _require_dict(metadata.get("output"), f"{metadata_path}: output")
        frame_count = output.get("frame_count")
        if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
            raise ValueError(f"{metadata_path}: output.frame_count must be a positive integer.")
        _require_value(output.get("views"), list(_VIEW_IDS), f"{metadata_path}: output.views")
        _require_value(
            output.get("resolution_hw"),
            [_HEIGHT, _WIDTH],
            f"{metadata_path}: output.resolution_hw",
        )

        rgb_metadata = _require_dict(output.get("rgb"), f"{metadata_path}: output.rgb")
        _require_value(rgb_metadata.get("format"), "jpeg", f"{metadata_path}: output.rgb.format")
        invalid_frame_indices = rgb_metadata.get("invalid_frame_indices")
        if not isinstance(invalid_frame_indices, list) or any(
            isinstance(frame, bool) or not isinstance(frame, int)
            for frame in invalid_frame_indices
        ):
            raise ValueError(
                f"{metadata_path}: output.rgb.invalid_frame_indices must be a list of integers."
            )
        if invalid_frame_indices != sorted(set(invalid_frame_indices)):
            raise ValueError(
                f"{metadata_path}: output.rgb.invalid_frame_indices must be sorted and unique."
            )
        if any(frame < 0 or frame >= frame_count for frame in invalid_frame_indices):
            raise ValueError(
                f"{metadata_path}: invalid RGB frame indices must be in [0, {frame_count})."
            )

        depth_metadata = _require_dict(output.get("depth"), f"{metadata_path}: output.depth")
        _require_value(depth_metadata.get("format"), "npy", f"{metadata_path}: output.depth.format")
        _require_value(depth_metadata.get("dtype"), "float32", f"{metadata_path}: output.depth.dtype")
        _require_value(
            depth_metadata.get("semantics"),
            "optical_z_meters",
            f"{metadata_path}: output.depth.semantics",
        )
        _require_value(depth_metadata.get("invalid_value"), 0.0, f"{metadata_path}: output.depth.invalid_value")
        _require_value(depth_metadata.get("clipped"), False, f"{metadata_path}: output.depth.clipped")

        visibility_metadata = _require_dict(
            output.get("visibility"),
            f"{metadata_path}: output.visibility",
        )
        _require_value(
            visibility_metadata.get("format"),
            "npy",
            f"{metadata_path}: output.visibility.format",
        )
        _require_value(
            visibility_metadata.get("dtype"),
            "bool",
            f"{metadata_path}: output.visibility.dtype",
        )

        expected_view_names = [f"view_{view}" for view in _VIEW_IDS]
        actual_view_names = sorted(
            path.name
            for path in scene_path.iterdir()
            if path.is_dir() and path.name.startswith("view_")
        )
        if actual_view_names != expected_view_names:
            raise ValueError(
                f"{scene_path} has view directories {actual_view_names}, expected {expected_view_names}."
            )

        tracks_3d = _load_npy(
            scene_path / "tracks_3d.npy",
            (frame_count, _POINT_COUNT, 3),
            np.float32,
        )
        views = []
        for view in _VIEW_IDS:
            view_path = scene_path / f"view_{view}"
            rgba = _read_rgb_frames(view_path, frame_count)
            depth = _load_npy(
                view_path / "depth.npy",
                (frame_count, _HEIGHT, _WIDTH),
                np.float32,
            )
            intrinsics = _load_npy(view_path / "intrinsics.npy", (3, 3), np.float32)
            extrinsics = _load_npy(
                view_path / "extrinsics_w2c.npy",
                (frame_count, 3, 4),
                np.float32,
            )
            visibility = _load_npy(
                view_path / "visibility.npy",
                (frame_count, _POINT_COUNT),
                np.bool_,
            )
            tracks_2d = _project_tracks(tracks_3d, extrinsics, intrinsics)
            views.append({
                "rgba": rgba,
                "depth": torch.from_numpy(depth[..., None]),
                "intrinsics": torch.from_numpy(intrinsics),
                "extrinsics": torch.from_numpy(extrinsics),
                "tracks_2d": torch.from_numpy(tracks_2d),
                "occlusion": torch.from_numpy(~visibility),
                "view_path": str(view_path),
            })

        return {
            "tracks_3d": torch.from_numpy(tracks_3d),
            "views": views,
            "invalid_rgb_frame_indices": invalid_frame_indices,
        }
