# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import pathlib
import re
import time

import cv2
import kornia
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.spatial.transform import Rotation as R
from torch.utils.data import get_worker_info
from torchvision.transforms import ColorJitter, GaussianBlur
from torchvision.transforms import functional as F_torchvision

from mvtracker.datasets.estimated_depth import (
    ESTIMATED_DEPTH_TYPE_PROBABILITIES,
    EstimatedDepthStore,
)
from mvtracker.datasets.kubric_metadata_index import KubricMetadataIndex
from mvtracker.datasets.scene_selection import select_scene_names
from mvtracker.datasets.utils import Datapoint, read_json, read_tiff, read_png, transform_scene, add_camera_noise, \
    aug_depth


def _training_virtual_dataset_size(
    world_size,
    num_steps,
    gradient_accumulation_steps,
    physical_batch_size=1,
):
    """Reserve one distinct virtual index per training microbatch."""
    gradient_accumulation_steps = int(gradient_accumulation_steps)
    if gradient_accumulation_steps < 1:
        raise ValueError("trainer.gradient_accumulation_steps must be at least 1")
    return (
        int(world_size)
        * int(physical_batch_size)
        * gradient_accumulation_steps
        * (int(num_steps) + 1000)
    )


def _legal_contiguous_window_starts(n_frames, seq_len, invalid_frame_indices=()):
    """Return inclusive window starts whose frames do not intersect invalid indices."""
    if isinstance(n_frames, bool) or not isinstance(n_frames, (int, np.integer)) or n_frames < 0:
        raise ValueError(f"n_frames must be a non-negative integer, got {n_frames!r}.")
    if isinstance(seq_len, bool) or not isinstance(seq_len, (int, np.integer)) or seq_len <= 0:
        raise ValueError(f"seq_len must be a positive integer, got {seq_len!r}.")

    invalid_frame_indices = list(invalid_frame_indices)
    if any(isinstance(frame_idx, bool) or not isinstance(frame_idx, (int, np.integer))
           for frame_idx in invalid_frame_indices):
        raise ValueError("Invalid frame indices must be integers.")
    invalid_frame_indices = [int(frame_idx) for frame_idx in invalid_frame_indices]
    if len(set(invalid_frame_indices)) != len(invalid_frame_indices):
        raise ValueError("Invalid frame indices must be unique.")
    if any(frame_idx < 0 or frame_idx >= n_frames for frame_idx in invalid_frame_indices):
        raise ValueError(
            f"Invalid frame indices must be in [0, {n_frames}), got {invalid_frame_indices}."
        )

    if n_frames < seq_len:
        return np.empty(0, dtype=np.int64)

    starts = np.arange(n_frames - seq_len + 1, dtype=np.int64)
    legal = np.ones(starts.shape, dtype=bool)
    for frame_idx in invalid_frame_indices:
        first_affected_start = max(0, frame_idx - seq_len + 1)
        last_affected_start = min(frame_idx, n_frames - seq_len)
        legal[first_affected_start:last_affected_start + 1] = False
    return starts[legal]


def _track_cache_file(data_root, scene_name, cache_name, enabled):
    """Return the cache file path, creating its directory only when enabled."""
    if not enabled:
        return None
    cache_root = os.path.join(data_root, scene_name, "cache")
    os.makedirs(cache_root, exist_ok=True)
    return os.path.join(cache_root, f"{cache_name}.npz")


class KubricMultiViewDataset(torch.utils.data.Dataset):

    @staticmethod
    def from_name(
            dataset_name: str,
            dataset_root: str,
            training_args=None,
            fabric=None,
            just_return_kwargs: bool = False,
            subset: str = "test",
            include_scene_ids=None,
            exclude_scene_ids=(),
            dali_stream_start_request_cursor: int = 0,
    ):
        """
        Examples of evaluation datasets supported by this factory method:
        - kubric-multiview-v3
        - kubric-multiview-v3-duster0123
        - kubric-multiview-v3-duster01234567
        - kubric-multiview-v3-duster01234567cleaned
        - kubric-multiview-v3-duster01234567cleaned-views012
        - kubric-multiview-v3-duster01234567cleaned-views012-novelviews7
        - kubric-multiview-v3-duster01234567cleaned-views012-novelviews7-overfit-on-training
        - kubric-multiview-v3-duster01234567cleaned-views012-novelviews7-overfit-on-training-single
        - kubric-multiview-v3-duster01234567cleaned-views012-novelviews7-overfit-on-training-2dpt-single
        - kubric-multiview-v3-duster01234567cleaned-views012-novelviews7-overfit-on-training-2dpt-single-cached
        - kubric-multiview-v3-noise1.23cm

        Example of a training dataset:
        - kubric-multiview-v3-training
        """
        # Parse the dataset name, chunk by chunk
        non_parsed = dataset_name.replace("kubric-multiview-v3", "", 1)

        if non_parsed.startswith("-noise"):
            match = re.match(r"-noise([\d.]+)cm", non_parsed)
            assert match is not None
            depth_noise_std = float(match.group(1))
            depth_noise_std = depth_noise_std / 13  # real-world cm to kubric's metric unit
            non_parsed = non_parsed.replace(match.group(0), "", 1)
        else:
            depth_noise_std = 0.0

        # Validation split names are configuration labels. The mixed trainer
        # supplies the actual scene IDs to the canonical Kubric factory.
        for validation_label in ("-validation-subset", "-validation-full"):
            if non_parsed.startswith(validation_label):
                non_parsed = non_parsed.replace(validation_label, "", 1)
                break

        if non_parsed.startswith("-duster"):
            match = re.match(r"-duster(\d+)(cleaned)?", non_parsed)
            assert match is not None
            duster_views = list(map(int, match.group(1)))
            use_duster = True
            use_duster_cleaned = match.group(2) is not None
            non_parsed = non_parsed.replace(match.group(0), "", 1)
        else:
            use_duster = False
            use_duster_cleaned = False
            duster_views = None

        if non_parsed.startswith("-views"):
            match = re.match(r"-views(\d+)", non_parsed)
            assert match is not None
            views = list(map(int, match.group(1)))
            if duster_views is not None:
                assert all(v in duster_views for v in views)
            non_parsed = non_parsed.replace(match.group(0), "", 1)
        else:
            views = duster_views

        if non_parsed.startswith("-novelviews"):
            match = re.match(r"-novelviews(\d+)", non_parsed)
            assert match is not None
            novel_views = list(map(int, match.group(1)))
            non_parsed = non_parsed.replace(match.group(0), "", 1)
        else:
            novel_views = None

        if non_parsed.startswith("-training"):
            training = True
            non_parsed = non_parsed.replace("-training", "", 1)
            assert training_args is not None
            assert fabric is not None
        else:
            training = False

        if non_parsed.startswith("-overfit-on-training"):
            overfit_on_train = True
            non_parsed = non_parsed.replace("-overfit-on-training", "", 1)
            assert not training, "Either ...-training or ...-overfit-on-training[-single][-2dpt]"
            assert training_args is not None
            expected_training_dset_name = (dataset_name.replace("-overfit-on-training", "-training")
                                           .replace("-single", "").replace("2dpt", ""))
            assert training_args.datasets.train.name == expected_training_dset_name, \
                f"{expected_training_dset_name} != {training_args.datasets.train.name}"
        else:
            overfit_on_train = False

        if non_parsed.startswith("-single"):
            assert not training, "The single-point evaluation options is not relevant for a training dataset"
            single_point = True
            non_parsed = non_parsed.replace("-single", "", 1)
        else:
            single_point = False

        if non_parsed.startswith("-2dpt"):
            eval_2dpt = True
            non_parsed = non_parsed.replace("-2dpt", "", 1)
        else:
            eval_2dpt = False

        if non_parsed.startswith("-cached"):
            use_cached_tracks = True
            non_parsed = non_parsed.replace("-cached", "", 1)
        else:
            use_cached_tracks = False

        assert non_parsed == "", f"Unparsed part of the dataset name: {non_parsed}"

        kubric_kwargs = {
            "data_root": os.path.join(dataset_root, "kubric-multiview", subset),
            "seq_len": 24,
            "traj_per_sample": 512,
            "seed": 72,
            "sample_vis_1st_frame": False,
            "tune_per_scene": False,
            "max_videos": 30,
            "use_duster_depths": use_duster,
            "duster_views": duster_views,
            "clean_duster_depths": use_duster_cleaned,
            "views_to_return": views,
            "novel_views": novel_views,
            "num_views": -1 if views is not None else 4,
            "depth_noise_std": depth_noise_std,
            "ratio_dynamic": 0.5,
            "ratio_very_dynamic": 0.25,
            "use_cached_tracks": use_cached_tracks,
            # The released paper benchmark ships frozen v1 track selections.
            "cache_version": "v1" if use_cached_tracks else "v3",
            "include_scene_ids": include_scene_ids,
            "exclude_scene_ids": exclude_scene_ids,
            "metadata_index_root": None,
        }
        if training:
            kubric_kwargs["virtual_dataset_size"] = _training_virtual_dataset_size(
                world_size=fabric.world_size,
                num_steps=training_args.trainer.num_steps,
                gradient_accumulation_steps=(
                    training_args.trainer.gradient_accumulation_steps
                ),
                physical_batch_size=training_args.datasets.train.batch_size,
            )
        if training or overfit_on_train:
            kubric_kwargs["data_root"] = (
                os.path.join(training_args.datasets.root, "kubric-multiview", "train")
                if not training_args.modes.debug else
                os.path.join(training_args.datasets.root, "kubric-multiview", "validation")
            )
            kubric_kwargs["seq_len"] = training_args.datasets.train.sequence_len
            kubric_kwargs["traj_per_sample"] = training_args.datasets.train.traj_per_sample
            kubric_kwargs["max_depth"] = training_args.datasets.train.kubric_max_depth
            kubric_kwargs["tune_per_scene"] = training_args.modes.tune_per_scene
            if training:
                kubric_kwargs["max_videos"] = training_args.datasets.train.max_videos
            else:
                kubric_kwargs["max_videos"] = 30

            kubric_kwargs["augmentation_probability"] = training_args.augmentations.probability
            kubric_kwargs["enable_rgb_augs"] = training_args.augmentations.rgb
            kubric_kwargs["enable_depth_augs"] = training_args.augmentations.depth
            kubric_kwargs["enable_cropping_augs"] = training_args.augmentations.cropping
            kubric_kwargs["aug_crop_size"] = training_args.augmentations.cropping_size
            kubric_kwargs["enable_variable_trajpersample_augs"] = training_args.augmentations.variable_trajpersample
            kubric_kwargs["enable_scene_transform_augs"] = training_args.augmentations.scene_transform
            kubric_kwargs["enable_camera_params_noise_augs"] = training_args.augmentations.camera_params_noise
            kubric_kwargs["enable_variable_depth_type_augs"] = training_args.augmentations.variable_depth_type
            kubric_kwargs["enable_variable_num_views_augs"] = training_args.augmentations.variable_num_views
            kubric_kwargs["normalize_scene_following_vggt"] = training_args.augmentations.normalize_scene_following_vggt
            kubric_kwargs["enable_variable_vggt_crop_size_augs"] = training_args.augmentations.variable_vggt_crop_size
            kubric_kwargs["keep_principal_point_centered"] = training_args.augmentations.keep_principal_point_centered

            estimated_depth_root = training_args.datasets.get("estimated_depth_root")
            if estimated_depth_root is not None and not os.path.isabs(estimated_depth_root):
                estimated_depth_root = os.path.join(dataset_root, estimated_depth_root)
            kubric_kwargs["estimated_depth_root"] = estimated_depth_root
            kubric_kwargs["estimated_depth_provider"] = training_args.datasets.get(
                "estimated_depth_provider"
            )
            metadata_index_root = training_args.datasets.train.get(
                "kubric_metadata_index_root"
            )
            if metadata_index_root is not None and not os.path.isabs(metadata_index_root):
                metadata_index_root = os.path.join(dataset_root, metadata_index_root)
            kubric_kwargs["metadata_index_root"] = metadata_index_root

            if training_args.modes.pretrain_only:
                kubric_kwargs["ratio_dynamic"] = 0.0
                kubric_kwargs["ratio_very_dynamic"] = 0.0

            if training_args.augmentations.variable_num_views:
                kubric_kwargs["num_views"] = None
                kubric_kwargs["views_to_return"] = None
                kubric_kwargs["duster_views"] = None
                kubric_kwargs["supported_duster_views_sets"] = [
                    [0, 1, 2, 3],
                    [0, 1, 2, 3, 4, 5, 6, 7],
                ]

        if just_return_kwargs:
            return kubric_kwargs

        mvkubric_storage = str(
            training_args.datasets.train.get("mvkubric_storage", "native")
        ) if training else "native"
        if training and mvkubric_storage == "dali_stream":
            from mvtracker.datasets.kubric_dali_dataset import DaliKubricMultiViewDataset

            webdataset_root = training_args.datasets.train.get("mvkubric_webdataset_root")
            if webdataset_root is None:
                raise ValueError(
                    "mvkubric_storage=dali_stream requires mvkubric_webdataset_root"
                )
            depth_provider = str(
                training_args.datasets.train.get(
                    "mvkubric_depth_provider",
                    training_args.datasets.get("estimated_depth_provider", "gt")
                    if training_args.datasets.get("estimated_depth_root")
                    else "gt",
                )
            )
            if depth_provider != "gt":
                estimated_webdataset_root = training_args.datasets.train.get(
                    "mvkubric_estimated_webdataset_root"
                )
                if estimated_webdataset_root is None:
                    raise ValueError(
                        "non-GT MV-Kubric DALI depth requires "
                        "mvkubric_estimated_webdataset_root"
                    )
                webdataset_root = estimated_webdataset_root
            if not os.path.isabs(str(webdataset_root)):
                webdataset_root = os.path.join(dataset_root, str(webdataset_root))
            kubric_kwargs["webdataset_root"] = webdataset_root
            kubric_kwargs["depth_provider"] = depth_provider
            kubric_kwargs["webdataset_split"] = "train"
            kubric_kwargs["stream_rank"] = int(fabric.global_rank)
            kubric_kwargs["stream_world_size"] = int(fabric.world_size)
            kubric_kwargs["stream_seed"] = int(training_args.reproducibility.seed)
            kubric_kwargs["stream_start_request_cursor"] = int(
                dali_stream_start_request_cursor
            )
            return DaliKubricMultiViewDataset(**kubric_kwargs)
        if training and mvkubric_storage not in {"native", "dali_stream"}:
            raise ValueError(f"unsupported mvkubric_storage={mvkubric_storage!r}")

        if training and bool(training_args.datasets.train.get("mvkubric_gpu_decode", False)):
            from mvtracker.datasets.kubric_gpu_dataset import (
                GpuDecodedKubricMultiViewDataset,
            )

            return GpuDecodedKubricMultiViewDataset(**kubric_kwargs)
        return KubricMultiViewDataset(**kubric_kwargs)

    def __init__(
            self,
            data_root,
            views_to_return=None,
            novel_views=None,
            use_duster_depths=False,
            clean_duster_depths=False,
            duster_views=None,
            supported_duster_views_sets=None,
            seq_len=24,
            num_views=4,
            traj_per_sample=768,
            max_depth=1000,
            sample_vis_1st_frame=False,
            ratio_dynamic=0.5,
            ratio_very_dynamic=0.25,
            depth_noise_std=0.0,

            augmentation_probability=0.0,
            enable_rgb_augs=False,
            enable_depth_augs=False,
            enable_cropping_augs=False,
            aug_crop_size=(384, 512),
            enable_variable_trajpersample_augs=False,
            enable_scene_transform_augs=False,
            enable_camera_params_noise_augs=False,
            enable_variable_depth_type_augs=False,
            enable_variable_num_views_augs=False,

            normalize_scene_following_vggt=False,
            enable_variable_vggt_crop_size_augs=False,
            keep_principal_point_centered=False,

            static_cropping=False,
            seed=None,
            tune_per_scene=False,
            max_videos=None,
            virtual_dataset_size=None,
            max_tracks_to_preload=18000,
            perform_sanity_checks=False,
            use_cached_tracks=False,
            cache_version="v3",
            estimated_depth_root=None,
            estimated_depth_provider=None,
            include_scene_ids=None,
            exclude_scene_ids=(),
            metadata_index_root=None,
            metadata_catalog=None,
            validate_scene_directories=True,
    ):
        super(KubricMultiViewDataset, self).__init__()

        self.data_root = data_root
        if metadata_catalog is not None and metadata_index_root is not None:
            raise ValueError("metadata_catalog and metadata_index_root are mutually exclusive")
        self.metadata_index = metadata_catalog or (
            KubricMetadataIndex(metadata_index_root)
            if metadata_index_root is not None
            else None
        )
        self.estimated_depth_store = EstimatedDepthStore(
            estimated_depth_root, estimated_depth_provider
        )
        self.views_to_return = views_to_return
        self.novel_views = novel_views
        self.use_duster_depths = use_duster_depths
        self.clean_duster_depths = clean_duster_depths
        self.duster_views = duster_views
        self.supported_duster_views_sets = supported_duster_views_sets
        if self.use_duster_depths:
            assert self.duster_views is not None, "When using Duster depths, duster_views must be set."
            if self.supported_duster_views_sets is None:
                self.supported_duster_views_sets = [self.duster_views]

        self.seq_len = seq_len
        self.num_views = num_views
        self.traj_per_sample = traj_per_sample
        self.sample_vis_1st_frame = sample_vis_1st_frame
        self.ratio_dynamic = ratio_dynamic
        self.ratio_very_dynamic = ratio_very_dynamic

        self.seed = seed
        self.add_index_to_seed = not tune_per_scene

        self.perform_sanity_checks = perform_sanity_checks
        self.use_cached_tracks = use_cached_tracks
        self.cache_version = cache_version
        self.cache_name = self._cache_key()
        self.max_tracks_to_preload = max_tracks_to_preload
        if self.traj_per_sample is not None and self.max_tracks_to_preload is not None:
            assert self.traj_per_sample <= self.max_tracks_to_preload, "We need to preload more tracks than we sample."

        self.depth_noise_std = depth_noise_std

        # Augmentation settings
        self.augmentation_probability = augmentation_probability
        if any([enable_rgb_augs, enable_depth_augs, enable_variable_trajpersample_augs,
                enable_scene_transform_augs, enable_camera_params_noise_augs, enable_variable_num_views_augs,
                enable_variable_depth_type_augs]):
            assert self.augmentation_probability > 0, "Augmentations are enabled, but augmentation probability is 0%."
        if self.augmentation_probability > 0:
            assert not self.use_cached_tracks, "caching tracks not supported with augs"

        self.enable_rgb_augs = enable_rgb_augs
        self.enable_depth_augs = enable_depth_augs
        self.enable_cropping_augs = enable_cropping_augs
        self.enable_variable_trajpersample_augs = enable_variable_trajpersample_augs
        self.enable_scene_transform_augs = enable_scene_transform_augs
        self.enable_camera_params_noise_augs = enable_camera_params_noise_augs
        self.enable_variable_num_views_augs = enable_variable_num_views_augs
        self.enable_variable_depth_type_augs = enable_variable_depth_type_augs
        self.enable_variable_depth_type_augs__depth_type_probability = {
            "gt": 0.70, "duster": 0.20, "duster_cleaned": 0.10,
        }
        # TODO: self.enable_seqlen_augs = enable_seqlen_augs
        if self.enable_variable_depth_type_augs:
            assert not self.use_duster_depths, "Cannot force depth type when using variable depth type augs."
            assert not self.clean_duster_depths, "Cannot force depth type when using variable depth type augs."
        self.enable_variable_num_views_augs__n_views_probability = {
            # v2
            1: 0.20,
            2: 0.10,
            3: 0.10,
            4: 0.25,
            5: 0.10,
            6: 0.25,

            # # v1
            # 1: 0.20,
            # 2: 0.10,
            # 3: 0.10,
            # 4: 0.25,
            # 5: 0.10,
            # 6: 0.05,
            # 7: 0.05,
            # 8: 0.15,
        }
        self.enable_variable_num_views_augs__trajpersample_adjustment_factor = {
            1: 1.00,
            2: 1.00,
            3: 1.00,
            4: 1.00,
            5: 0.40,
            6: 0.25,
        }
        if self.enable_variable_num_views_augs:
            assert self.num_views is None, "Cannot use enable_variable_num_views_augs with num_views != None."
            assert self.views_to_return is None, "Cannot use enable_variable_num_views_augs with views_to_return."

        # photometric augmentation
        # TODO: "Override" ColorJitter and GaussianBlur to take in a random state
        #       in forward pass so we can assure reproducibility. This affects
        #       only training as augmentation is disabled during evaluation.
        self.photo_aug = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.25 / 3.14)
        self.blur_aug = GaussianBlur(11, sigma=(0.1, 2.0))
        self.blur_aug_prob = 0.25
        self.color_aug_prob = 0.25

        # occlusion augmentation
        self.eraser_aug_prob = 0.5
        self.eraser_bounds = [2, 100]
        self.eraser_max = 10

        # occlusion augmentation
        self.replace_aug_prob = 0.5
        self.replace_bounds = [2, 100]
        self.replace_max = 10

        # spatial augmentations
        self.crop_size = aug_crop_size
        self.normalize_scene_following_vggt = normalize_scene_following_vggt
        self.enable_variable_vggt_crop_size_augs = enable_variable_vggt_crop_size_augs
        self.keep_principal_point_centered = keep_principal_point_centered

        self.max_depth = max_depth

        self.pad_bounds = [0, 45]
        self.resize_lim = [0.8, 1.2]
        self.resize_delta = 0.15
        self.max_crop_offset = 36
        if static_cropping or tune_per_scene:
            self.pad_bounds = [0, 1]
            self.resize_lim = [1.0, 1.0]
            self.resize_delta = 0.0
            self.max_crop_offset = 0

        if self.keep_principal_point_centered:
            self.pad_bounds = [0, 45]
            self.resize_lim = [1.02, 1.25]
            self.resize_delta = None
            self.max_crop_offset = None
            if static_cropping or tune_per_scene:
                self.pad_bounds = [0, 1]
                self.resize_lim = [1.04, 1.04]

        self.seq_names = (
            list(self.metadata_index.scenes)
            if self.metadata_index is not None
            else [
                fname
                for fname in os.listdir(self.data_root)
                if os.path.isdir(os.path.join(self.data_root, fname))
                   and not fname.startswith(".")
                   and not fname.startswith("_")
            ]
        )
        self.seq_names = sorted(
            self.seq_names,
            key=lambda name: (0, int(name)) if name.isdigit() else (1, name),
        )
        if validate_scene_directories:
            seq_names_clean = []
            for seq_name in self.seq_names:
                scene_path = os.path.join(self.data_root, seq_name)
                if self.metadata_index is not None:
                    scene_entry, _ = self.metadata_index.scene(seq_name)
                    view_folders = scene_entry["view_names"]
                else:
                    view_folders = [
                        d for d in os.listdir(scene_path)
                        if os.path.isdir(os.path.join(scene_path, d))
                        and (d.startswith('view_') or d.isdigit())
                    ]
                if len(view_folders) == 0:
                    logging.warning(f"Skipping {scene_path} because it has no views.")
                    continue
                if self.num_views is not None and len(view_folders) < self.num_views:
                    logging.warning(f"Skipping {scene_path} because it has {len(view_folders)} views (<{self.num_views}).")
                    continue
                seq_names_clean.append(seq_name)
            self.seq_names = seq_names_clean

        self.seq_names = select_scene_names(
            self.seq_names,
            include=include_scene_ids,
            exclude=exclude_scene_ids,
        )

        if self.supported_duster_views_sets is not None:
            supported_duster_views_sets_cleaned = []
            for s in self.supported_duster_views_sets:
                duster_views_str = ''.join(str(v) for v in s)
                if os.path.isdir(os.path.join(self.data_root, self.seq_names[0], f"duster-views-{duster_views_str}")):
                    supported_duster_views_sets_cleaned.append(s)
                else:
                    logging.warning(f"Skipping duster views set {s} because it does not exist.")
            self.supported_duster_views_sets = supported_duster_views_sets_cleaned

        if tune_per_scene:
            self.seq_names = self.seq_names[3:4]
        if max_videos is not None:
            self.seq_names = self.seq_names[:max_videos]
        logging.info("Using %d videos from %s" % (len(self.seq_names), self.data_root))

        self.real_len = len(self.seq_names)
        if virtual_dataset_size is not None:
            self.virtual_len = virtual_dataset_size
        else:
            self.virtual_len = self.real_len
        logging.info(f"Real dataset size: {self.real_len}. Virtual dataset size: {self.virtual_len}.")

        self.getitem_calls = 0

    def _cache_key(self):
        name = f"cachedtracks--seed{self.seed}-dynamic{self.ratio_dynamic}-verydynamic-{self.ratio_very_dynamic}"
        if self.views_to_return is not None:
            name += f"-views{'_'.join(map(str, self.views_to_return))}"
        if self.traj_per_sample is not None:
            name += f"-n{self.traj_per_sample}"
        if self.num_views is not None:
            name += f"-numviews{self.num_views}"
        if self.seq_len is not None:
            name += f"-t{self.seq_len}"
        if self.sample_vis_1st_frame:
            name += f"-sample_vis_1st_frame"
        return name + f"--{self.cache_version}"

    def __len__(self):
        return self.virtual_len

    def _motion_bucket_ratios(self, total_tracks, very_dynamic_tracks):
        """Return motion-bucket ratios; subclasses may opt into dataset-specific policy."""
        del total_tracks, very_dynamic_tracks
        return self.ratio_dynamic, self.ratio_very_dynamic

    def __getitem__(self, index):
        request = index if hasattr(index, "virtual_index") else None
        virtual_index = request.virtual_index if request is not None else int(index)
        scene_index = (
            request.scene_index
            if request is not None and request.scene_index is not None
            else virtual_index % self.real_len
        )
        if not 0 <= scene_index < self.real_len:
            raise IndexError(f"scene index {scene_index} is outside [0, {self.real_len})")

        sample, gotit = self._getitem_helper(scene_index, seed_index=virtual_index)

        if not gotit:
            logging.warning("warning: sampling failed")
            # fake sample, so we can still collate
            num_views = self.num_views if self.num_views is not None else 4
            h, w = 384, 512
            traj_per_sample = self.traj_per_sample if self.traj_per_sample is not None else 768
            sample = Datapoint(
                video=torch.zeros((num_views, self.seq_len, 3, h, w)),
                videodepth=torch.zeros((num_views, self.seq_len, 1, h, w)),
                segmentation=torch.zeros((num_views, self.seq_len, 1, h, w)),
                trajectory=torch.zeros((self.seq_len, traj_per_sample, 2)),
                visibility=torch.zeros((self.seq_len, traj_per_sample)),
                valid=torch.zeros((self.seq_len, traj_per_sample)),
                sample_metadata={
                    "virtual_index": virtual_index,
                    "scene_index": scene_index,
                    "scene_name": self.seq_names[scene_index],
                    "gotit": False,
                },
            )

        return sample, gotit

    def _getitem_helper(self, index, seed_index=None):
        start_time_1 = time.time()

        gotit = True

        if seed_index is None:
            seed_index = index

        # Take a new seed from torch or use self.seed if set
        # The rest of the code will use generators initialized with this seed
        if self.seed is None:
            seed = torch.randint(0, 2 ** 32 - 1, (1,)).item()
        else:
            seed = self.seed
            if self.add_index_to_seed:
                seed += seed_index
        rnd_torch = torch.Generator().manual_seed(seed)
        rnd_np = np.random.RandomState(seed=seed)

        # Indexed mode first loads only lightweight scene data. The selected views
        # are materialized after view sampling below.
        scene_path = os.path.join(self.data_root, self.seq_names[index])
        indexed_scene = None
        metadata_index = getattr(self, "metadata_index", None)
        if metadata_index is not None:
            indexed_scene, indexed_arrays = metadata_index.scene(self.seq_names[index])
            with np.load(os.path.join(scene_path, "tracks_3d.npz")) as tracks_file:
                tracks_3d = tracks_file["tracks_3d"].copy()
            datapoint = {
                "tracks_3d": torch.from_numpy(tracks_3d),
                "views": [None] * len(indexed_scene["view_names"]),
                "invalid_frame_indices": indexed_scene["invalid_frame_indices"],
            }
        else:
            datapoint = self.getitem_raw_datapoint(scene_path)

        traj3d_world = datapoint["tracks_3d"].numpy()
        views = datapoint["views"]
        invalid_frame_indices = datapoint.get("invalid_frame_indices", ())
        legal_start_indices = _legal_contiguous_window_starts(
            n_frames=traj3d_world.shape[0],
            seq_len=self.seq_len,
            invalid_frame_indices=invalid_frame_indices,
        )
        if len(legal_start_indices) == 0:
            scene_path = os.path.join(self.data_root, self.seq_names[index])
            raise ValueError(
                f"No valid {self.seq_len}-frame windows remain in scene {scene_path}; "
                f"invalid frame indices: {list(invalid_frame_indices)}."
            )

        # Take a random depth type, if enabled
        if self.enable_variable_depth_type_augs:
            assert self.use_duster_depths is False, "Cannot force depth type when using variable depth type augs."
            assert self.clean_duster_depths is False, "Cannot force depth type when using variable depth type augs."
            depth_probabilities = (
                ESTIMATED_DEPTH_TYPE_PROBABILITIES
                if self.estimated_depth_store.enabled
                else self.enable_variable_depth_type_augs__depth_type_probability
            )
            depth_type = rnd_np.choice(
                a=list(depth_probabilities.keys()),
                size=1,
                p=list(depth_probabilities.values()),
            )[0]
            use_duster_depths, clean_duster_depths = {
                "gt": (False, False),
                "duster": (True, False),
                "duster_cleaned": (True, True),
                "estimated": (False, False),
                "estimated_cleaned": (False, False),
            }[depth_type]
        else:
            depth_type = "duster_cleaned" if self.clean_duster_depths else (
                "duster" if self.use_duster_depths else "gt"
            )
            use_duster_depths = self.use_duster_depths
            clean_duster_depths = self.clean_duster_depths

        # Take a random number of views, if enabled
        all_views = sorted(list(range(len(views))))
        if self.enable_variable_num_views_augs:
            assert self.num_views is None, "Cannot use enable_variable_num_views_augs with num_views != None."
            assert self.views_to_return is None, "Cannot use enable_variable_num_views_augs with views_to_return."
            num_views = rnd_np.choice(
                a=list(self.enable_variable_num_views_augs__n_views_probability.keys()),
                size=1,
                p=list(self.enable_variable_num_views_augs__n_views_probability.values()),
            )[0]
            if use_duster_depths:
                num_views = min(num_views, max([len(s) for s in self.supported_duster_views_sets]))
                # Take only those that have the closest number of views that is greater or equal to num_views
                closest_num_views_in_supported_duster_views_set = min([
                    len(vs)
                    for vs in self.supported_duster_views_sets
                    if len(vs) >= num_views
                ])
                supported_duster_views_sets = [
                    vs
                    for vs in self.supported_duster_views_sets
                    if len(vs) == closest_num_views_in_supported_duster_views_set
                ]
                duster_views = supported_duster_views_sets[rnd_np.randint(len(supported_duster_views_sets))]
                views_to_return = rnd_np.choice(duster_views, num_views, replace=False).tolist()
            else:
                views_to_return = rnd_np.choice(all_views, num_views, replace=False).tolist()
                duster_views = views_to_return
        else:
            num_views = self.num_views
            if self.views_to_return is not None:
                assert num_views == -1, "Cannot use views_to_return with num_views != -1."
                views_to_return = self.views_to_return
            elif use_duster_depths:
                if self.duster_views is not None:
                    duster_views = self.duster_views
                else:
                    # Take only those that have the closest number of views that is greater or equal to num_views
                    closest_num_views_in_supported_duster_views_set = min([
                        len(vs)
                        for vs in self.supported_duster_views_sets
                        if len(vs) >= num_views
                    ])
                    supported_duster_views_sets = [
                        vs
                        for vs in self.supported_duster_views_sets
                        if len(vs) == closest_num_views_in_supported_duster_views_set
                    ]
                    duster_views = supported_duster_views_sets[rnd_np.randint(len(supported_duster_views_sets))]
                views_to_return = duster_views
            else:
                if num_views == -1:
                    # Take all views
                    views_to_return = all_views
                elif num_views is None:
                    # Randomly sample a number of views
                    n = rnd_np.randint(min(3, len(views)), len(views) + 1)
                    views_to_return = rnd_np.choice(all_views, n, replace=False).tolist()
                else:
                    # Take a fixed number of views
                    assert num_views > 0, "Fixed number of views must be positive."
                    assert num_views <= len(views), f"Not enough views available (idx={index})."
                    views_to_return = rnd_np.choice(all_views, num_views, replace=False).tolist()
            if self.duster_views is not None:
                duster_views = self.duster_views
            else:
                duster_views = views_to_return

        if indexed_scene is not None:
            views_needed = set(views_to_return)
            if self.novel_views is not None:
                views_needed.update(self.novel_views)
            views = self.getitem_indexed_views(
                scene_path,
                indexed_scene,
                indexed_arrays,
                sorted(views_needed),
            )

        # Extract only the data we need
        rgbs = np.stack([views[v]["rgba"][..., :3].numpy() for v in views_to_return])
        depths = np.stack([views[v]["depth"].numpy() for v in views_to_return])
        # segs = np.stack([views[v]["segmentation"].numpy() for v in views_to_return])
        segs = np.ones(((rgbs.shape[0], rgbs.shape[1], rgbs.shape[2], rgbs.shape[3], 1)), dtype=np.float32)
        intrs = np.stack([views[v]["intrinsics"].numpy() for v in views_to_return])
        intrs = intrs[:, None, :, :].repeat(rgbs.shape[1], axis=1)
        extrs = np.stack([views[v]["extrinsics"].numpy() for v in views_to_return])
        traj2d = np.stack([views[v]["tracks_2d"].numpy() for v in views_to_return])
        visibility = ~np.stack([views[v]["occlusion"].numpy() for v in views_to_return])

        novel_rgbs = None
        novel_intrs = None
        novel_extrs = None
        if self.novel_views is not None:
            novel_rgbs = np.stack([views[v]["rgba"][..., :3].numpy() for v in self.novel_views])
            novel_intrs = np.stack([views[v]["intrinsics"].numpy() for v in self.novel_views])
            novel_intrs = novel_intrs[:, None, :, :].repeat(rgbs.shape[1], axis=1)
            novel_extrs = np.stack([views[v]["extrinsics"].numpy() for v in self.novel_views])

        # Load Duster's features and estimated depths if they exist
        duster_views_str = ''.join(str(v) for v in duster_views)
        duster_root = pathlib.Path(self.data_root) / self.seq_names[index] / f'duster-views-{duster_views_str}'

        num_views, n_frames, h, w, _ = rgbs.shape
        if depth_type in {"estimated", "estimated_cleaned"}:
            estimated_depths, cleaned_mask = self.estimated_depth_store.load(
                self.seq_names[index], views_to_return
            )
            if estimated_depths.shape != (num_views, n_frames, h, w):
                raise ValueError(
                    f"estimated depth shape {estimated_depths.shape} does not match "
                    f"scene shape {(num_views, n_frames, h, w)}"
                )
            if depth_type == "estimated_cleaned":
                estimated_depths = estimated_depths * cleaned_mask
            depths = np.asarray(estimated_depths[..., None], dtype=np.float32)
        feats = None
        feat_dim = None
        feat_stride = None
        duster_outputs_exist = duster_root.exists() and (
                duster_root / f"3d_model__{n_frames - 1:05d}__scene.npz").exists()
        if use_duster_depths:
            assert duster_outputs_exist, "use_duster_depths --> duster_output_exist"
        if duster_outputs_exist:
            duster_depths = []
            duster_feats = []
            for frame_idx in range(n_frames):
                scene = np.load(duster_root / f"3d_model__{frame_idx:05d}__scene.npz")
                duster_depth = torch.from_numpy(scene["depths"])
                duster_conf = torch.from_numpy(scene["confs"])
                duster_msk = torch.from_numpy(scene["cleaned_mask"])
                duster_feat = torch.from_numpy(scene["feats"])

                if clean_duster_depths:
                    ## Filter based on the confidence
                    # conf_threshold = max(0.00001, min(0.1, torch.quantile(duster_conf.flatten(), 0.3).item()))
                    # duster_depth = duster_depth * (duster_conf > conf_threshold)

                    # Filter based on the mask
                    duster_depth = duster_depth * duster_msk

                duster_depth = F.interpolate(duster_depth[:, None], (depths.shape[2], depths.shape[3]), mode='nearest')

                duster_depths.append(duster_depth[:, 0, :, :, None])
                duster_feats.append(duster_feat)

            duster_depths = torch.stack(duster_depths, dim=1).numpy()
            feats = torch.stack(duster_feats, dim=1).numpy()

            # Extract the correct views
            assert duster_depths.shape[0] == feats.shape[0] == len(duster_views)
            duster_depths = duster_depths[[duster_views.index(v) for v in views_to_return]]
            feats = feats[[duster_views.index(v) for v in views_to_return]]

            # Reshape the features
            assert feats.ndim == 4
            assert feats.shape[0] == num_views
            assert feats.shape[1] == n_frames
            feat_stride = np.round(np.sqrt(h * w / feats.shape[2])).astype(int)
            feat_dim = feats.shape[3]
            feats = feats.reshape(num_views, n_frames, h // feat_stride, w // feat_stride, feat_dim)

            # Replace the depths with the Duster depths, if configured so
            if use_duster_depths:
                depths = duster_depths

        start_time_2 = time.time()

        # Strategically select dynamic points to track
        visible_at_t_and_t_plus_1 = (visibility[:, :-1] & visibility[:, 1:]).any(0)
        movement = np.linalg.norm(traj3d_world[1:] - traj3d_world[:-1], axis=-1)
        movement[~visible_at_t_and_t_plus_1] = 0
        movement = movement.sum(axis=0)
        assert np.isfinite(movement).all(), "Movement contains NaN or Inf values."

        static_threshold = 0.01  # < 1 cm
        dynamic_threshold = 0.1  # > 10 cm
        very_dynamic_threshold = 2.0  # > 2 m

        static_points = movement < static_threshold  # 1 cm
        dynamic_points = movement > dynamic_threshold  # 10 cm
        very_dynamic_points = movement > very_dynamic_threshold  # 2 m

        if self.perform_sanity_checks:
            logging.info(f"Movement stats: "
                         f"static: {static_points.sum()} ({static_points.mean() * 100:.2f}), "
                         f"dynamic: {dynamic_points.sum()} ({dynamic_points.mean() * 100:.2f}), "
                         f"very dynamic: {very_dynamic_points.sum()} ({very_dynamic_points.mean() * 100:.2f})"
                         f"other: {(~static_points & ~dynamic_points & ~very_dynamic_points).sum()}")

        # Sample the points according to the desired ratios if possible
        ratio_dynamic, ratio_very_dynamic = self._motion_bucket_ratios(
            total_tracks=traj3d_world.shape[1],
            very_dynamic_tracks=int(very_dynamic_points.sum()),
        )
        ratio_static = 1 - ratio_dynamic - ratio_very_dynamic
        max_tracks_to_preload = traj3d_world.shape[1]
        max_tracks_to_preload = min([
            max_tracks_to_preload,
            int(dynamic_points.sum() / ratio_dynamic) if ratio_dynamic > 0 else max_tracks_to_preload,
            int(very_dynamic_points.sum() // ratio_very_dynamic) if ratio_very_dynamic > 0 else max_tracks_to_preload,
            int(static_points.sum() / ratio_static) if ratio_static > 0 else max_tracks_to_preload,
        ])
        if self.max_tracks_to_preload is not None:
            max_tracks_to_preload = min(max_tracks_to_preload, self.max_tracks_to_preload)
        n_dynamic = min(int(max_tracks_to_preload * ratio_dynamic), dynamic_points.sum())
        n_very_dynamic = min(int(max_tracks_to_preload * ratio_very_dynamic), very_dynamic_points.sum())
        n_static = max_tracks_to_preload - n_dynamic - n_very_dynamic

        dynamic_indices = rnd_np.choice(np.where(dynamic_points)[0], n_dynamic, replace=False)
        very_dynamic_indices = rnd_np.choice(np.where(very_dynamic_points)[0], n_very_dynamic, replace=False)
        static_indices = rnd_np.choice(np.where(static_points)[0], n_static, replace=False)

        selected_indices = np.concatenate([dynamic_indices, very_dynamic_indices, static_indices])
        rnd_np.shuffle(selected_indices)

        traj3d_world = traj3d_world[:, selected_indices]
        traj2d = traj2d[:, :, selected_indices]
        visibility = visibility[:, :, selected_indices]
        if traj3d_world.shape[1] > max_tracks_to_preload:
            traj3d_world = traj3d_world[:, :max_tracks_to_preload]
            traj2d = traj2d[:, :, :max_tracks_to_preload]
            visibility = visibility[:, :, :max_tracks_to_preload]

        n_tracks = traj3d_world.shape[1]
        num_views, n_frames, h, w, _ = rgbs.shape
        assert n_frames >= self.seq_len
        assert rgbs.shape == (num_views, n_frames, h, w, 3)
        assert depths.shape == (num_views, n_frames, h, w, 1)
        assert segs.shape == (num_views, n_frames, h, w, 1)
        assert feats is None or feats.shape == (num_views, n_frames, h // feat_stride, w // feat_stride, feat_dim)
        assert intrs.shape == (num_views, n_frames, 3, 3)
        assert extrs.shape == (num_views, n_frames, 3, 4)
        assert traj2d.shape == (num_views, n_frames, n_tracks, 2)
        assert visibility.shape == (num_views, n_frames, n_tracks)
        assert traj3d_world.shape == (n_frames, n_tracks, 3)

        if novel_rgbs is not None:
            assert novel_rgbs.shape == (len(self.novel_views), n_frames, h, w, 3)
            assert novel_intrs.shape == (len(self.novel_views), n_frames, 3, 3)
            assert novel_extrs.shape == (len(self.novel_views), n_frames, 3, 4)

        if ((depths < 0.01) & (depths != 0)).mean() > 0.5:
            raise ValueError("Depth map might be invalid? Values that are too small will be ignored by SpaTracker, "
                             "but found that more than half of non-zero depths are below 0.01 in the loaded depths.")

        # Make sure our intrinsics and extrinsics work correctly
        point_3d_world = traj3d_world
        point_4d_world_homo = np.concatenate([point_3d_world, np.ones_like(point_3d_world[..., :1])], axis=-1)
        point_3d_camera = np.einsum('ABij,BCj->ABCi', extrs, point_4d_world_homo)
        if self.perform_sanity_checks:
            point_2d_pixel_homo = np.einsum('ABij,ABCj->ABCi', intrs, point_3d_camera)
            point_2d_pixel = point_2d_pixel_homo[..., :2] / point_2d_pixel_homo[..., 2:]
            point_2d_pixel_gt = traj2d
            assert np.allclose(point_2d_pixel[0, :, 0, :], point_2d_pixel_gt[0, :, 0, :], atol=1e-3), f"Proj. failed"
            assert np.allclose(point_2d_pixel, point_2d_pixel_gt, atol=1e-3), f"Point projection failed"

        # Now save the z value in traj3d_camera as usual, just if needed
        traj3d_camera = point_3d_camera
        assert traj3d_camera.shape == (num_views, n_frames, n_tracks, 3)

        # Also sanity check that pix2cam is working correctly with the intrinsics
        if self.perform_sanity_checks:
            from mvtracker.models.core.spatracker.blocks import pix2cam
            xyz = np.concatenate([traj2d, traj3d_camera[..., 2:]], axis=-1)
            pix2cam_xyz = torch.from_numpy(xyz).double()
            pix2cam_intr = torch.from_numpy(intrs).double()
            traj_3d_repro = pix2cam(pix2cam_xyz, pix2cam_intr).numpy()
            assert np.allclose(traj3d_camera, traj_3d_repro, atol=0.1)

        # If the video is too long, randomly crop self.seq_len frames
        start_ind = 0
        if self.seq_len < n_frames:
            start_ind = int(rnd_np.choice(legal_start_indices))
            rgbs = rgbs[:, start_ind: start_ind + self.seq_len]
            depths = depths[:, start_ind: start_ind + self.seq_len]
            segs = segs[:, start_ind: start_ind + self.seq_len]
            if feats is not None:
                feats = feats[:, start_ind: start_ind + self.seq_len]
            intrs = intrs[:, start_ind: start_ind + self.seq_len]
            extrs = extrs[:, start_ind: start_ind + self.seq_len]
            traj2d = traj2d[:, start_ind: start_ind + self.seq_len]
            visibility = visibility[:, start_ind: start_ind + self.seq_len]
            traj3d_camera = traj3d_camera[:, start_ind: start_ind + self.seq_len]
            traj3d_world = traj3d_world[start_ind: start_ind + self.seq_len]
            n_frames = self.seq_len

        # Add the z value to the traj2d
        traj2d_w_z = np.concatenate((traj2d[..., :], traj3d_camera[..., 2:]), axis=-1)

        start_time_3 = time.time()
        augment_this_datapoint = False
        if self.augmentation_probability > 0:
            augment_this_datapoint = rnd_np.rand() <= self.augmentation_probability
        if augment_this_datapoint and self.enable_rgb_augs:
            rgbs, visibility = self._add_photometric_augs(rgbs, traj2d_w_z, visibility, rnd_np)

        crop_size = self.crop_size
        if augment_this_datapoint and self.enable_variable_vggt_crop_size_augs:
            sizes = list(range(168, 518 + 14, 14))  # VIT-friendly sizes
            weights = np.array(sizes) ** 2  # Quadratic bias toward larger sizes
            probs = weights / weights.sum()
            shorter_side = rnd_np.choice(a=sizes, size=1, p=probs)[0]
            longer_side = max(crop_size)
            crop_size = (shorter_side, longer_side)
        if self.enable_cropping_augs and not self.keep_principal_point_centered:
            rgbs, depths, intrs, traj2d_w_z, visibility = self._add_cropping_augs(
                crop_size=crop_size,
                rgbs=rgbs,
                depths=depths,
                intrs=intrs,
                trajs=traj2d_w_z,
                visibles=visibility,
            )
            h, w = rgbs.shape[-3:-1]
        if self.enable_cropping_augs and self.keep_principal_point_centered:
            rgbs, depths, intrs, traj2d_w_z, visibility = self._add_cropping_augs_with_pp_at_center(
                crop_size=crop_size,
                rgbs=rgbs,
                depths=depths,
                intrs=intrs,
                trajs=traj2d_w_z,
                visibles=visibility,
            )
            h, w = rgbs.shape[-3:-1]

        depths[depths > self.max_depth] = 0.0
        if augment_this_datapoint and self.enable_depth_augs:
            invalid_depth_mask = depths <= 0.0
            depths = aug_depth(
                torch.from_numpy(depths).reshape(num_views * n_frames, 1, h, w),
                grid=(16, 16),
                scale=(0.99, 1.01),
                shift=(-0.001, 0.001),
                gn_kernel=(5, 5),
                gn_sigma=(2, 2),
                generator=rnd_torch,
            ).reshape(num_views, n_frames, h, w, 1).numpy()
            depths, visibility = self._rescale_and_erase_depth_patches(depths, traj2d_w_z, visibility, rnd_np)
            depths[invalid_depth_mask] = 0.0  # Restore invalid depths

        if self.depth_noise_std > 0.0:
            invalid_depth_mask = depths <= 0.0
            noise = np.random.normal(loc=0.0, scale=self.depth_noise_std, size=depths.shape)
            depths = depths + noise.astype(depths.dtype)
            depths = np.clip(depths, 0.0, self.max_depth)
            depths[invalid_depth_mask] = 0.0  # Restore invalid depths

        rgbs = torch.from_numpy(rgbs).permute(0, 1, 4, 2, 3).float()
        depths = torch.from_numpy(depths).permute(0, 1, 4, 2, 3).float()
        segs = torch.from_numpy(segs).permute(0, 1, 4, 2, 3).float()
        feats = torch.from_numpy(feats).permute(0, 1, 4, 2, 3).float() if feats is not None else None
        intrs = torch.from_numpy(intrs).float()
        extrs = torch.from_numpy(extrs).float()
        visibility = torch.from_numpy(visibility)
        traj2d = torch.from_numpy(traj2d)
        traj2d_w_z = torch.from_numpy(traj2d_w_z)
        traj3d_camera = torch.from_numpy(traj3d_camera)
        traj3d_world = torch.from_numpy(traj3d_world)
        if novel_rgbs is not None:
            novel_rgbs = torch.from_numpy(novel_rgbs).permute(0, 1, 4, 2, 3).float()
            novel_intrs = torch.from_numpy(novel_intrs).float()
            novel_extrs = torch.from_numpy(novel_extrs).float()

        # Track selection
        cache_file = _track_cache_file(
            self.data_root,
            self.seq_names[index],
            self.cache_name,
            self.use_cached_tracks,
        )

        # Check if we can use cached tracks
        use_cache = cache_file is not None and os.path.isfile(cache_file)
        if use_cache:
            cache = np.load(cache_file)
            visible_inds_sampled = torch.from_numpy(cache["track_indices"])
            traj2d_w_z = torch.from_numpy(cache["traj2d_w_z"])
            traj3d_world = torch.from_numpy(cache["traj3d_world"])
            visibility = torch.from_numpy(cache["visibility"])
            valids = torch.from_numpy(cache["valids"])
            query_points = torch.from_numpy(cache["query_points"])

        # Otherwise, sample the tracks and create query points
        else:
            # Sample the points to track
            visibile_pts_first_frame_inds = (visibility.any(0)[0]).nonzero(as_tuple=False)[:, 0]
            if self.sample_vis_1st_frame:
                visibile_pts_inds = visibile_pts_first_frame_inds
            else:
                visibile_pts_mid_frame_inds = (visibility.any(0)[self.seq_len // 2]).nonzero(as_tuple=False)[:, 0]
                visibile_pts_inds = torch.cat((visibile_pts_first_frame_inds, visibile_pts_mid_frame_inds), dim=0)
                visibile_pts_inds = torch.unique(visibile_pts_inds)
            visible_for_at_least_two_frames = (visibility.any(0).sum(0) >= 2).nonzero(as_tuple=False)[:, 0]
            visibile_pts_inds = visibile_pts_inds[torch.isin(visibile_pts_inds, visible_for_at_least_two_frames)]
            point_inds = torch.randperm(len(visibile_pts_inds), generator=rnd_torch)

            traj_per_sample = self.traj_per_sample if self.traj_per_sample is not None else len(point_inds)
            if self.enable_variable_num_views_augs:
                adj_factor = self.enable_variable_num_views_augs__trajpersample_adjustment_factor.get(num_views, 1.0)
                traj_per_sample = int(traj_per_sample * adj_factor)
            if len(point_inds) == 0 or len(point_inds) < traj_per_sample // 4:
                gotit = False
                return None, gotit
            if augment_this_datapoint and self.enable_variable_trajpersample_augs:
                if index % 20 == 0:
                    traj_per_sample = traj_per_sample // 8
                elif index % 21 == 0:
                    pass  # keep the same number of trajectories
                else:
                    low = max(1, traj_per_sample // 4)
                    high = min(len(point_inds), traj_per_sample) + 1
                    traj_per_sample = torch.randint(low=low, high=high, size=(1,), generator=rnd_torch).item()
            else:
                traj_per_sample = min(len(point_inds), traj_per_sample)
            point_inds = point_inds[:traj_per_sample]
            logging.info(
                f"[i={index:04d};seq={self.seq_names[index]};seed={seed}]"
                f"Selected {len(point_inds)}/{len(visibile_pts_inds)} tracks. "
                f"{num_views=}. "
                f"{point_inds[0]=} max_depth={self.max_depth}."
            )

            visible_inds_sampled = visibile_pts_inds[point_inds]

            n_tracks = len(visible_inds_sampled)
            traj2d = traj2d[:, :, visible_inds_sampled].float()
            traj2d_w_z = traj2d_w_z[:, :, visible_inds_sampled].float()
            traj3d_camera = traj3d_camera[:, :, visible_inds_sampled].float()
            traj3d_world = traj3d_world[:, visible_inds_sampled].float()
            visibility = visibility[:, :, visible_inds_sampled]
            valids = torch.ones((n_frames, n_tracks))

            # Create the query points
            gt_visibilities_any_view = visibility.any(dim=0)
            assert (gt_visibilities_any_view.sum(dim=0) >= 2).all(), "All points should be visible in least two frames."
            last_visible_index = (torch.arange(n_frames).unsqueeze(-1) * gt_visibilities_any_view).max(0).values
            assert gt_visibilities_any_view[last_visible_index[None, :], torch.arange(n_tracks)].all()
            gt_visibilities_any_view[last_visible_index[None, :], torch.arange(n_tracks)] = False
            assert (gt_visibilities_any_view.sum(dim=0) >= 1).all()

            if self.sample_vis_1st_frame:
                n_non_first_point_appearance_queries = 0
                n_first_point_appearance_queries = n_tracks
            else:
                n_non_first_point_appearance_queries = n_tracks // 4
                n_first_point_appearance_queries = n_tracks - n_non_first_point_appearance_queries

            first_point_appearances = torch.argmax(
                gt_visibilities_any_view[..., -n_first_point_appearance_queries:].float(), dim=0)
            non_first_point_appearances = first_point_appearances.new_zeros((n_non_first_point_appearance_queries,))
            for track_idx in range(n_tracks)[:n_non_first_point_appearance_queries]:
                # Randomly take a timestep where the point is visible
                non_zero_timesteps = torch.nonzero(gt_visibilities_any_view[:, track_idx] == 1)
                random_timestep = non_zero_timesteps[rnd_np.randint(len(non_zero_timesteps))].item()
                non_first_point_appearances[track_idx] = random_timestep

            query_points_t = torch.cat([non_first_point_appearances, first_point_appearances], dim=0)
            query_points_xyz_worldspace = traj3d_world[query_points_t, torch.arange(n_tracks)]
            query_points = torch.cat([query_points_t[:, None], query_points_xyz_worldspace], dim=1)
            assert gt_visibilities_any_view[query_points_t, torch.arange(n_tracks)].all()

            # Cache the selected tracks and query points
            if self.use_cached_tracks:
                assert cache_file is not None
                logging.warn(f"Caching tracks for {self.seq_names[index]} at {os.path.abspath(cache_file)}")
                np.savez_compressed(
                    cache_file,
                    track_indices=visible_inds_sampled.numpy(),
                    traj2d_w_z=traj2d_w_z.numpy(),
                    traj3d_world=traj3d_world.numpy(),
                    visibility=visibility.numpy(),
                    valids=valids.numpy(),
                    query_points=query_points.numpy(),
                )

        # Apply a transform to the world space
        scale = 1.0
        rot = torch.eye(3, dtype=torch.float32)
        translation = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        if self.enable_scene_transform_augs:
            rot_x_angle = rnd_np.uniform(-15, 15)
            rot_y_angle = rnd_np.uniform(-15, 15)
            rot_z_angle = 0.0
            scale = rnd_np.uniform(0.8, 1.5)
            translate_x = rnd_np.uniform(-2, 2)
            translate_y = rnd_np.uniform(-2, 2)
            translate_z = rnd_np.uniform(-2, 2)

            rot_x = R.from_euler('x', rot_x_angle, degrees=True).as_matrix()
            rot_y = R.from_euler('y', rot_y_angle, degrees=True).as_matrix()
            rot_z = R.from_euler('z', rot_z_angle, degrees=True).as_matrix()
            rot = torch.as_tensor(rot_z @ rot_y @ rot_x, dtype=torch.float32)
            T_rot = torch.eye(4)
            T_rot[:3, :3] = rot
            T_scale_and_translate = torch.tensor([
                [scale, 0.0, 0.0, translate_x],
                [0.0, scale, 0.0, translate_y],
                [0.0, 0.0, scale, translate_z],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=torch.float32)
            T = T_scale_and_translate @ T_rot

        if self.normalize_scene_following_vggt:
            assert not self.enable_scene_transform_augs, "Cannot normalize scene with scene transform augs enabled."
            extrs_square = torch.eye(4, device=extrs.device)[None, None].repeat(num_views, n_frames, 1, 1)
            extrs_square[:, :, :3, :] = extrs
            extrs_inv = torch.inverse(extrs_square)
            intrs_inv = torch.inverse(intrs)

            y, x = torch.meshgrid(
                torch.arange(h, device=extrs.device),
                torch.arange(w, device=extrs.device),
                indexing="ij",
            )
            homog = torch.stack([x, y, torch.ones_like(x)], dim=-1).float().reshape(-1, 3)
            homog = homog[None].expand(num_views, -1, -1)
            cam_points = torch.einsum("Vij, VNj->VNi", intrs_inv[:, 0], homog) * depths[:, 0].reshape(num_views, -1, 1)
            cam_points_h = torch.cat([cam_points, torch.ones_like(cam_points[..., :1])], dim=-1)
            world_points_h = torch.einsum("Vij, VNj->VNi", extrs_inv[:, 0], cam_points_h)

            world_points_in_first = torch.einsum("ij, VNj->VNi", extrs[0, 0], world_points_h)

            mask = (depths[:, 0] > 0).reshape(num_views, -1)
            valid_points = world_points_in_first[mask]
            avg_dist = valid_points.norm(dim=1).mean()
            scale = 1.0 / avg_dist

            depths *= scale
            traj3d_world *= scale
            traj3d_camera *= scale
            traj2d_w_z[..., 2] *= scale
            extrs[:, :, :3, 3] *= scale

            T_first_cam_to_origin = torch.eye(4, device=extrs.device)
            T_first_cam_to_origin[:3, :4] = extrs[0, 0]
            T = T_first_cam_to_origin

        (
            depths_trans, extrs_trans, query_points_trans, traj3d_world_trans, traj2d_w_z_trans
        ) = transform_scene(scale, rot, translation, depths, extrs, query_points, traj3d_world, traj2d_w_z)
        novel_extrs_trans = transform_scene(scale, rot, translation, None, novel_extrs, None, None, None)[1]

        if self.enable_camera_params_noise_augs:
            intrs, extrs_trans = add_camera_noise(
                intrs=intrs.numpy(),
                extrs=extrs_trans.numpy(),
                noise_std_intr=0.001,
                noise_std_extr=0.001,
                rnd=rnd_np,
            )
            intrs = torch.from_numpy(intrs)
            extrs_trans = torch.from_numpy(extrs_trans)

        # Dump non-normalized tracks to disk
        if self.augmentation_probability == 0.0 and not self.enable_variable_trajpersample_augs and seed is not None:
            num_views_str = self.num_views if self.num_views is not None else "none"
            views_str = ''.join(str(v) for v in self.views_to_return) if self.views_to_return is not None else "none"
            duster_views_str = ''.join(str(v) for v in self.duster_views) if self.duster_views is not None else "none"
            sample_identifier_str = (
                f"seed-{seed:06d}"
                f"_tracks-{self.traj_per_sample}"
                f"_use-duster-depths-{self.use_duster_depths}"
                f"_clean-duster-depths-{self.clean_duster_depths}"
                f"_num-views-{num_views_str}"
                f"_views-{views_str}"
                f"_duster-views-{duster_views_str}"
                f"_ratio-dynamic-{self.ratio_dynamic}"
                f"_ratio-very-dynamic-{self.ratio_very_dynamic}"
                f"_aug-prob-{self.augmentation_probability}"
                f"_max-tracks-to-preload-{self.max_tracks_to_preload}"
            )
            datapoint_path = os.path.join(self.data_root, self.seq_names[index])
            dumped_path = os.path.join(datapoint_path, f"{sample_identifier_str}.npz")
            # if not os.path.exists(dumped_path):
            #     logging.info(f"Dumping {dumped_path}")
            #     np.savez(
            #         dumped_path,
            #         trajectories=traj3d_world.numpy(),
            #         trajectories_pixelspace=traj2d.numpy(),
            #         per_view_visibilities=visibility.numpy(),
            #         query_points_3d=query_points.numpy(),
            #         extrinsics=extrs.numpy(),
            #         intrinsics=intrs.numpy(),
            #         transform_that_would_have_been_applied=T,
            #     )

        datapoint = Datapoint(
            video=rgbs,
            videodepth=depths_trans,
            feats=feats,
            segmentation=segs,
            trajectory=traj2d_w_z_trans,
            trajectory_3d=traj3d_world_trans,
            visibility=visibility,
            valid=valids,
            seq_name=self.seq_names[index],
            intrs=intrs,
            extrs=extrs_trans,
            query_points=None,
            query_points_3d=query_points_trans,
            sample_metadata={
                "virtual_index": int(seed_index),
                "scene_index": int(index),
                "scene_name": self.seq_names[index],
                "seed": int(seed),
                "window_start": int(start_ind),
                "window_end_exclusive": int(start_ind + self.seq_len),
                "selected_views": [int(view) for view in views_to_return],
                "depth_source": depth_type,
                "gotit": True,
            },
            track_upscaling_factor=1 / scale,

            novel_video=novel_rgbs,
            novel_intrs=novel_intrs,
            novel_extrs=novel_extrs_trans,
        )

        # Log timings
        start_time_4 = time.time()
        self.getitem_calls += 1
        top_duration = start_time_2 - start_time_1
        middle_duration = start_time_3 - start_time_2
        bottom_duration = start_time_4 - start_time_3
        total_duration = start_time_4 - start_time_1
        logging.info(f"Loading {index:>06d} took {total_duration:>7.3f}s "
                     f"[top:{top_duration:>7.3f}s, middle:{middle_duration:>7.3f}s, bottom:{bottom_duration:>7.3f}s] "
                     f"Getitem calls: {self.getitem_calls:>6d}. "
                     f"n_views={num_views}, {n_tracks=:>4d}, augmented={int(augment_this_datapoint)} {rgbs.shape=}")

        min_valid_depth_ratio_threshold = 0.1
        valid_depth_ratio = (depths > 0).float().mean()
        if valid_depth_ratio < min_valid_depth_ratio_threshold:
            logging.warning(f"Skipping datapoint {index} due to too little valid depth values: "
                            f"{valid_depth_ratio * 100:.1f}% (< {min_valid_depth_ratio_threshold * 100:.1f}%)")
            return None, False

        return datapoint, gotit

    @staticmethod
    def getitem_indexed_views(scene_path, scene_entry, arrays, view_indices, frame_indices=None):
        """Load only requested native views (and optionally requested frames)."""
        n_frames = int(scene_entry["n_frames"])
        if frame_indices is None:
            frame_indices = range(n_frames)
        frame_indices = list(frame_indices)
        intrinsics = arrays["intrinsics"]
        extrinsics = arrays["extrinsics"]
        sensor_widths = arrays["sensor_widths"]
        focal_lengths = arrays["focal_lengths"]

        views = [None] * len(scene_entry["view_names"])
        for view_index in view_indices:
            view_path = os.path.join(scene_path, scene_entry["view_names"][view_index])
            rgba = torch.stack([
                torch.from_numpy(np.asarray(read_png(os.path.join(
                    view_path, scene_entry["rgba_files"][view_index][frame_index]
                ))))
                for frame_index in frame_indices
            ])
            depth_images = [
                np.asarray(read_tiff(os.path.join(
                    view_path, scene_entry["depth_files"][view_index][frame_index]
                )))
                for frame_index in frame_indices
            ]
            if depth_images[0].dtype == np.uint16:
                depth_images = [image.astype(np.int32) for image in depth_images]
            depth = torch.stack([torch.from_numpy(image) for image in depth_images])
            depth = KubricMultiViewDataset.depth_from_euclidean_to_z(
                depth=depth,
                sensor_width=sensor_widths[view_index],
                focal_length=focal_lengths[view_index],
            )
            too_far = depth > 1000
            if too_far.any():
                depth[too_far] = 0

            with np.load(os.path.join(view_path, "tracks_2d.npz")) as tracks_file:
                tracks_2d = torch.from_numpy(tracks_file["tracks_2d"].copy())
                occlusion = torch.from_numpy(tracks_file["occlusion"].copy())
            views[view_index] = {
                "rgba": rgba,
                "depth": depth,
                "tracks_2d": tracks_2d[frame_indices],
                "occlusion": occlusion[frame_indices],
                "intrinsics": torch.from_numpy(intrinsics[view_index]),
                "extrinsics": torch.from_numpy(extrinsics[view_index, frame_indices]),
                "view_path": view_path,
            }
        return views

    @staticmethod
    def getitem_raw_datapoint(scene_path, perform_2d_projection_sanity_check=True):
        # Load global scene data
        tracks_3d = torch.from_numpy(
            np.load(os.path.join(scene_path, 'tracks_3d.npz'))['tracks_3d'],
        )
        tracks_segmentation_ids = torch.from_numpy(
            np.load(os.path.join(scene_path, 'tracks_segmentation_ids.npz'))['tracks_segmentation_ids'],
        )
        tracked_objects = read_json(os.path.join(scene_path, 'tracked_objects.json'))

        if os.path.exists(os.path.join(scene_path, 'views.npz')):
            # V2 (lookat fixed to 0)
            camera_positions = torch.from_numpy(np.load(os.path.join(scene_path, 'views.npz'))['views'])
            lookat_positions = 0. * camera_positions
        elif os.path.exists(os.path.join(scene_path, 'cameras.npz')):
            # V3 (with randomized lookat)
            camera_positions = torch.from_numpy(np.load(os.path.join(scene_path, 'cameras.npz'))['camera_positions'])
            lookat_positions = torch.from_numpy(np.load(os.path.join(scene_path, 'cameras.npz'))['lookat_positions'])
        else:
            raise ValueError("No camera data found: neither views.npz nor cameras.npz exist.")

        invalid_frame_indices = []
        scene_metadata_path = os.path.join(scene_path, "scene.json")
        if os.path.isfile(scene_metadata_path):
            scene_metadata = read_json(scene_metadata_path)
            if not isinstance(scene_metadata, dict):
                raise ValueError(f"{scene_metadata_path}: root must be an object.")
            output_metadata = scene_metadata.get("output", {})
            if not isinstance(output_metadata, dict):
                raise ValueError(f"{scene_metadata_path}: 'output' must be an object.")
            rgb_metadata = output_metadata.get("rgb", {})
            if not isinstance(rgb_metadata, dict):
                raise ValueError(f"{scene_metadata_path}: 'output.rgb' must be an object.")
            invalid_frame_indices = rgb_metadata.get("invalid_frame_indices", [])
            if not isinstance(invalid_frame_indices, list):
                raise ValueError(
                    f"{scene_metadata_path}: 'output.rgb.invalid_frame_indices' must be a list."
                )

        n_frames = tracks_3d.shape[0]
        n_tracks = tracks_3d.shape[1]
        n_views = camera_positions.shape[0]
        assert tracks_3d.shape == (n_frames, n_tracks, 3)
        assert tracks_segmentation_ids.shape == (n_tracks,)
        assert camera_positions.shape == (n_views, 3)
        assert lookat_positions.shape == (n_views, 3)

        # Initialize views data
        views_data = []
        view_folders = [
            d for d in os.listdir(scene_path)
            if os.path.isdir(os.path.join(scene_path, d)) and d.startswith('view_')
        ]
        view_folders = sorted(view_folders, key=lambda x: int(x.split('_')[-1]))

        for view_folder in view_folders:
            view_path = os.path.join(scene_path, view_folder)

            # Load per-view data
            view_data = {
                'rgba': [],
                'depth': [],
                # 'segmentation': [],
            }

            frame_files = sorted(os.listdir(view_path))
            for frame_file in frame_files:
                if frame_file.startswith('rgba_'):
                    view_data['rgba'].append(read_png(os.path.join(view_path, frame_file)))
                elif frame_file.startswith('depth_'):
                    view_data['depth'].append(read_tiff(os.path.join(view_path, frame_file)))
                # elif frame_file.startswith('segmentation_'):
                #     view_data['segmentation'].append(read_png(os.path.join(view_path, frame_file)))

            assert len(view_data['rgba']) == n_frames, f"{len(view_data['rgba'])}!={n_frames}"
            assert len(view_data['depth']) == n_frames, f"{len(view_data['depth'])}!={n_frames}"
            # assert len(view_data['segmentation']) == n_frames, f"{len(view_data['segmentation'])}!={n_frames}"

            # Convert lists to torch tensors
            for key in view_data:
                if view_data[key][0].dtype == np.uint16:
                    view_data[key] = [a.astype(np.int32) for a in view_data[key]]
                view_data[key] = torch.stack([torch.from_numpy(np.array(img)) for img in view_data[key]])

            # Load additional per-view data
            view_data.update({
                'tracks_2d': torch.from_numpy(np.load(os.path.join(view_path, 'tracks_2d.npz'))['tracks_2d']),
                'occlusion': torch.from_numpy(np.load(os.path.join(view_path, 'tracks_2d.npz'))['occlusion']),
                'data_ranges': "NOT LOADED",  # read_json(os.path.join(view_path, 'data_ranges.json')),
                'metadata': read_json(os.path.join(view_path, 'metadata.json')),
                'events': "NOT LOADED",  # read_json(os.path.join(view_path, 'events.json')),
                'object_id_to_segmentation_id': read_json(os.path.join(view_path, 'object_id_to_segmentation_id.json')),
            })

            # Extracting the intrinsics
            view_data['intrinsics'] = torch.tensor(view_data['metadata']['camera']['K'], dtype=torch.float64)
            assert view_data['intrinsics'].shape == (3, 3)

            # Extracting the extrinsics
            positions = torch.tensor(view_data['metadata']['camera']['positions'], dtype=torch.float64)
            quaternions = torch.tensor(view_data['metadata']['camera']['quaternions'], dtype=torch.float64)
            rotation_matrices = kornia.geometry.quaternion_to_rotation_matrix(quaternions)
            assert positions.shape == (n_frames, 3)
            assert quaternions.shape == (n_frames, 4)
            assert rotation_matrices.shape == (n_frames, 3, 3)
            extrinsics_inv = torch.zeros((n_frames, 4, 4), dtype=torch.float64)
            extrinsics_inv[:, :3, :3] = rotation_matrices
            extrinsics_inv[:, :3, 3] = positions
            extrinsics_inv[:, 3, 3] = 1
            view_data['extrinsics'] = extrinsics_inv.inverse()
            assert torch.allclose(view_data['extrinsics'][:, 3, :3], torch.zeros(n_frames, 3, dtype=torch.float64))
            assert torch.allclose(view_data['extrinsics'][:, 3, 3], torch.ones(n_frames, dtype=torch.float64))
            view_data['extrinsics'] = view_data['extrinsics'][:, :3, :]

            # Change the intrinsics to the format
            w, h = view_data["metadata"]["metadata"]["resolution"]
            view_data['intrinsics'] = np.diag([w, h, 1]) @ view_data['intrinsics'].numpy() @ np.diag([1, -1, -1])
            view_data['extrinsics'] = np.diag([1, -1, -1]) @ view_data['extrinsics'].numpy()
            view_data['intrinsics'] = torch.from_numpy(view_data['intrinsics'])
            view_data['extrinsics'] = torch.from_numpy(view_data['extrinsics'])

            # Project one point to the image plane to check if the extrinsics are correct
            if perform_2d_projection_sanity_check:
                point_3d_world = tracks_3d[0, 0]
                point_4d_world_homo = torch.cat([point_3d_world, torch.ones(1)])
                point_2d_pixel = view_data['intrinsics'] @ view_data['extrinsics'][0] @ point_4d_world_homo
                point_2d_pixel = point_2d_pixel[:2] / point_2d_pixel[2]
                point_2d_pixel_gt = view_data["tracks_2d"][0, 0]
                assert torch.allclose(point_2d_pixel, point_2d_pixel_gt, atol=1e-3), f"Point projection failed"

            # The original depth is the euclidean distance from the camera
            # Compute the depth in z format instead (so the z coordinate in the camera space)
            view_data['depth'] = KubricMultiViewDataset.depth_from_euclidean_to_z(
                depth=view_data['depth'],
                sensor_width=view_data['metadata']['camera']['sensor_width'],
                focal_length=view_data['metadata']['camera']['focal_length'],
            )

            # Sometimes the Kubric depths contains very high values of 10e9
            # We will clip those to 10e3 to avoid problems with inf and nan
            larger_than_1000 = view_data['depth'] > 1000
            if larger_than_1000.any():
                logging.info(f"Datapoint {scene_path} has depths larger than 1000: "
                             f"{view_data['depth'][larger_than_1000]}. "
                             f"Replacing those by 0 to denote invalid depth and avoid inf and nan values later.")
                view_data['depth'][larger_than_1000] = 0

            view_data['view_path'] = view_path
            views_data.append(view_data)

        datapoint = {
            "tracks_3d": tracks_3d,
            "tracks_segmentation_ids": tracks_segmentation_ids,
            "tracked_objects": tracked_objects,
            "camera_positions": camera_positions,
            "lookat_positions": lookat_positions,
            "views": views_data,
            "invalid_frame_indices": invalid_frame_indices,
        }

        return datapoint

    @staticmethod
    def depth_from_euclidean_to_z(depth, sensor_width, focal_length):
        n_frames, h, w, _ = depth.shape
        sensor_height = sensor_width / w * h
        pixel_centers_x = (np.arange(-w / 2, w / 2, dtype=np.float32) + 0.5) / w * sensor_width
        pixel_centers_y = (np.arange(-h / 2, h / 2, dtype=np.float32) + 0.5) / h * sensor_height

        # Calculate squared distance from the center of the image
        pixel_centers_x, pixel_centers_y = np.meshgrid(pixel_centers_x, pixel_centers_y, indexing="xy")
        squared_distance_from_center = np.square(pixel_centers_x) + np.square(pixel_centers_y)

        # Calculate rescaling factor for each pixel
        z_to_eucl_rescaling = np.sqrt(1 + squared_distance_from_center / focal_length ** 2)

        # Apply the rescaling to each depth value
        z_to_eucl_rescaling = np.expand_dims(z_to_eucl_rescaling, axis=-1)  # Add a dimension for broadcasting
        depth_z = depth / z_to_eucl_rescaling
        return depth_z

    def _add_photometric_augs(
            self,
            rgbs,
            trajs,
            visibles,
            rndstate,
            eraser=True,
            replace=True,
    ):
        V, T, H, W, _ = rgbs.shape
        _, _, N, _ = trajs.shape
        assert rgbs.dtype == np.uint8
        assert rgbs.shape == (V, T, H, W, 3)
        assert trajs.shape == (V, T, N, 3)
        assert visibles.shape == (V, T, N)

        rgbs = rgbs.copy()
        visibles = visibles.copy()

        if eraser:  # eraser the specific region in the image
            for v in range(V):
                rgbs_view = rgbs[v]
                rgbs_view = [rgb.astype(np.float32) for rgb in rgbs_view]
                ############ eraser transform (per image after the first) ############
                for i in range(1, T):
                    if rndstate.rand() < self.eraser_aug_prob:
                        for _ in range(
                                rndstate.randint(1, self.eraser_max + 1)
                        ):  # number of times to occlude
                            xc = rndstate.randint(0, W)
                            yc = rndstate.randint(0, H)
                            dx = rndstate.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                            dy = rndstate.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                            x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                            x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                            y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                            y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)
                            mean_color = np.mean(rgbs_view[i][y0:y1, x0:x1, :].reshape(-1, 3), axis=0)
                            rgbs_view[i][y0:y1, x0:x1, :] = mean_color
                            occ_inds = np.logical_and(
                                np.logical_and(trajs[v, i, :, 0] >= x0, trajs[v, i, :, 0] < x1),
                                np.logical_and(trajs[v, i, :, 1] >= y0, trajs[v, i, :, 1] < y1),
                            )
                            visibles[v, i, occ_inds] = 0
                rgbs_view = [rgb.astype(np.uint8) for rgb in rgbs_view]
                rgbs[v] = np.stack(rgbs_view)

        if replace:
            for v in range(V):
                rgbs_view = rgbs[v]
                rgbs_view_alt = [
                    np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8)
                    for rgb in rgbs_view
                ]
                rgbs_view_alt = [
                    np.array(self.photo_aug(Image.fromarray(rgb)), dtype=np.uint8)
                    for rgb in rgbs_view_alt
                ]

                ############ replace transform (per image after the first) ############
                rgbs_view = [rgb.astype(np.float32) for rgb in rgbs_view]
                rgbs_view_alt = [rgb.astype(np.float32) for rgb in rgbs_view_alt]
                for i in range(1, T):
                    if rndstate.rand() < self.replace_aug_prob:
                        for _ in range(
                                rndstate.randint(1, self.replace_max + 1)
                        ):  # number of times to occlude
                            xc = rndstate.randint(0, W)
                            yc = rndstate.randint(0, H)
                            dx = rndstate.randint(self.replace_bounds[0], self.replace_bounds[1])
                            dy = rndstate.randint(self.replace_bounds[0], self.replace_bounds[1])
                            x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                            x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                            y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                            y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)

                            wid = x1 - x0
                            hei = y1 - y0
                            y00 = rndstate.randint(0, H - hei)
                            x00 = rndstate.randint(0, W - wid)
                            fr = rndstate.randint(0, T)
                            rep = rgbs_view_alt[fr][y00: y00 + hei, x00: x00 + wid, :]
                            rgbs_view[i][y0:y1, x0:x1, :] = rep

                            occ_inds = np.logical_and(
                                np.logical_and(trajs[v, i, :, 0] >= x0, trajs[v, i, :, 0] < x1),
                                np.logical_and(trajs[v, i, :, 1] >= y0, trajs[v, i, :, 1] < y1),
                            )
                            visibles[v, i, occ_inds] = 0
                rgbs_view = [rgb.astype(np.uint8) for rgb in rgbs_view]
                rgbs[v] = np.stack(rgbs_view)

        ############ photometric augmentation ############
        if rndstate.rand() < self.color_aug_prob:
            # random per-frame amount of aug
            # but shared across all views
            for i in range(T):
                fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.photo_aug.get_params(
                    self.photo_aug.brightness, self.photo_aug.contrast, self.photo_aug.saturation, self.photo_aug.hue
                )
                for v in range(V):
                    rgb = rgbs[v, i]
                    rgb = Image.fromarray(rgb)
                    for fn_id in fn_idx:
                        if fn_id == 0 and brightness_factor is not None:
                            rgb = F_torchvision.adjust_brightness(rgb, brightness_factor)
                        elif fn_id == 1 and contrast_factor is not None:
                            rgb = F_torchvision.adjust_contrast(rgb, contrast_factor)
                        elif fn_id == 2 and saturation_factor is not None:
                            rgb = F_torchvision.adjust_saturation(rgb, saturation_factor)
                        elif fn_id == 3 and hue_factor is not None:
                            rgb = F_torchvision.adjust_hue(rgb, hue_factor)
                    rgb = np.array(rgb, dtype=np.uint8)
                    rgbs[v, i] = rgb

        if rndstate.rand() < self.blur_aug_prob:
            # random per-frame amount of blur
            # but shared across all views
            for i in range(T):
                sigma = self.blur_aug.get_params(self.blur_aug.sigma[0], self.blur_aug.sigma[1])
                for v in range(V):
                    rgb = rgbs[v, i]
                    rgb = Image.fromarray(rgb)
                    rgb = F_torchvision.gaussian_blur(
                        rgb,
                        self.blur_aug.kernel_size,
                        [sigma, sigma],
                    )
                    rgb = np.array(rgb, dtype=np.uint8)
                    rgbs[v, i] = rgb

        return rgbs, visibles

    def _add_cropping_augs(self, crop_size, rgbs, depths, intrs, trajs, visibles):
        V, T, H, W, _ = rgbs.shape
        _, _, N, _ = trajs.shape
        assert rgbs.dtype == np.uint8
        assert depths.dtype == np.float32
        assert rgbs.shape == (V, T, H, W, 3)
        assert depths.shape == (V, T, H, W, 1)
        assert intrs.shape == (V, T, 3, 3)
        assert trajs.shape == (V, T, N, 3)
        assert visibles.shape == (V, T, N)

        rgbs = rgbs.copy()
        depths = depths.copy()
        intrs = intrs.copy()
        trajs = trajs.copy()
        visibles = visibles.copy()

        ############ spatial transform ############
        rgbs_new = np.zeros((V, T, crop_size[0], crop_size[1], 3), dtype=np.uint8)
        depths_new = np.zeros((V, T, crop_size[0], crop_size[1], 1), dtype=np.float32)
        for v in range(V):
            # padding
            pad_x0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
            pad_x1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
            pad_y0 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
            pad_y1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])

            rgbs_view = [np.pad(rgb, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0))) for rgb in rgbs[v]]
            depths_view = [np.pad(depth, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0))) for depth in depths[v]]
            intrs[v, :, 0, 2] += pad_x0
            intrs[v, :, 1, 2] += pad_y0
            trajs[v, :, :, 0] += pad_x0
            trajs[v, :, :, 1] += pad_y0
            H_padded, W_padded = rgbs_view[0].shape[:2]

            # scaling + stretching
            scale = np.random.uniform(self.resize_lim[0], self.resize_lim[1])
            scale_x = scale
            scale_y = scale

            scale_delta_x = 0.0
            scale_delta_y = 0.0

            for t in range(T):
                if t == 1:
                    scale_delta_x = np.random.uniform(-self.resize_delta, self.resize_delta)
                    scale_delta_y = np.random.uniform(-self.resize_delta, self.resize_delta)
                elif t > 1:
                    scale_delta_x = (
                            scale_delta_x * 0.8
                            + np.random.uniform(-self.resize_delta, self.resize_delta) * 0.2
                    )
                    scale_delta_y = (
                            scale_delta_y * 0.8
                            + np.random.uniform(-self.resize_delta, self.resize_delta) * 0.2
                    )
                scale_x = scale_x + scale_delta_x
                scale_y = scale_y + scale_delta_y

                # bring h/w closer
                scale_xy = (scale_x + scale_y) * 0.5
                scale_x = scale_x * 0.5 + scale_xy * 0.5
                scale_y = scale_y * 0.5 + scale_xy * 0.5

                # don't get too crazy
                scale_x = np.clip(scale_x, self.resize_lim[0], self.resize_lim[1])
                scale_y = np.clip(scale_y, self.resize_lim[0], self.resize_lim[1])

                H_new = int(H_padded * scale_y)
                W_new = int(W_padded * scale_x)

                # make it at least slightly bigger than the crop area,
                # so that the random cropping can add diversity
                H_new = np.clip(H_new, crop_size[0] + 10, None)
                W_new = np.clip(W_new, crop_size[1] + 10, None)
                # recompute scale in case we clipped
                scale_x = (W_new - 1) / float(W_padded - 1)
                scale_y = (H_new - 1) / float(H_padded - 1)
                rgbs_view[t] = cv2.resize(rgbs_view[t], (W_new, H_new), interpolation=cv2.INTER_LINEAR)
                depths_view[t] = cv2.resize(depths_view[t], (W_new, H_new), interpolation=cv2.INTER_NEAREST)
                intrs[v, t, 0, :] *= scale_x
                intrs[v, t, 1, :] *= scale_y
                trajs[v, t, :, 0] *= scale_x
                trajs[v, t, :, 1] *= scale_y
            ok_inds = visibles[v, 0, :] > 0
            vis_trajs = trajs[v, :, ok_inds]  # S,?,2

            if vis_trajs.shape[0] > 0:
                mid_x = np.mean(vis_trajs[:, 0, 0])
                mid_y = np.mean(vis_trajs[:, 0, 1])
            else:
                mid_y = crop_size[0] // 2
                mid_x = crop_size[1] // 2

            x0 = int(mid_x - crop_size[1] // 2)
            y0 = int(mid_y - crop_size[0] // 2)

            offset_x = 0
            offset_y = 0

            for t in range(T):
                # on each frame, shift a bit more
                if t == 1:
                    offset_x = np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1)
                    offset_y = np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1)
                elif t > 1:
                    offset_x = int(
                        offset_x * 0.8
                        + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1)
                        * 0.2
                    )
                    offset_y = int(
                        offset_y * 0.8
                        + np.random.randint(-self.max_crop_offset, self.max_crop_offset + 1)
                        * 0.2
                    )
                x0 = x0 + offset_x
                y0 = y0 + offset_y

                H_new, W_new = rgbs_view[t].shape[:2]
                if H_new == crop_size[0]:
                    y0 = 0
                else:
                    y0 = min(max(0, y0), H_new - crop_size[0] - 1)

                if W_new == crop_size[1]:
                    x0 = 0
                else:
                    x0 = min(max(0, x0), W_new - crop_size[1] - 1)

                rgbs_view[t] = rgbs_view[t][y0: y0 + crop_size[0], x0: x0 + crop_size[1]]
                depths_view[t] = depths_view[t][y0: y0 + crop_size[0], x0: x0 + crop_size[1]]
                intrs[v, t, 0, 2] -= x0
                intrs[v, t, 1, 2] -= y0
                trajs[v, t, :, 0] -= x0
                trajs[v, t, :, 1] -= y0

            H_new = crop_size[0]
            W_new = crop_size[1]

            # # h flip
            # if self.do_flip and np.random.rand() < self.h_flip_prob:
            #     rgbs_view = [rgb[:, ::-1] for rgb in rgbs_view]
            #     depths_view = [depth[:, ::-1] for depth in depths_view]
            #     intrs[v, :, 0, 2] = W_new - intrs[v, :, 0, 2]
            #     trajs[v, :, :, 0] = W_new - trajs[v, :, :, 0]
            #
            # # v flip
            # if np.random.rand() < self.v_flip_prob:
            #     rgbs_view = [rgb[::-1] for rgb in rgbs_view]
            #     depths_view = [depth[::-1] for depth in depths_view]
            #     intrs[v, :, 1, 2] = H_new - intrs[v, :, 1, 2]
            #     trajs[v, :, :, 1] = H_new - trajs[v, :, :, 1]

            rgbs_new[v] = np.stack(rgbs_view)
            depths_new[v] = np.stack(depths_view)[..., None]

        visibles = (visibles &
                    (trajs[..., 0] >= 0) &
                    (trajs[..., 1] >= 0) &
                    (trajs[..., 0] < crop_size[1]) &
                    (trajs[..., 1] < crop_size[0]))

        return rgbs_new, depths_new, intrs, trajs, visibles

    def _add_cropping_augs_with_pp_at_center(self, crop_size, rgbs, depths, intrs, trajs, visibles):
        V, T, H, W, _ = rgbs.shape
        _, _, N, _ = trajs.shape
        assert rgbs.dtype == np.uint8
        assert depths.dtype == np.float32
        assert rgbs.shape == (V, T, H, W, 3)
        assert depths.shape == (V, T, H, W, 1)
        assert intrs.shape == (V, T, 3, 3)
        assert trajs.shape == (V, T, N, 3)
        assert visibles.shape == (V, T, N)

        rgbs = rgbs.copy()
        depths = depths.copy()
        intrs = intrs.copy()
        trajs = trajs.copy()
        visibles = visibles.copy()

        rgbs_new = np.zeros((V, T, crop_size[0], crop_size[1], 3), dtype=np.uint8)
        depths_new = np.zeros((V, T, crop_size[0], crop_size[1], 1), dtype=np.float32)

        for v in range(V):
            pad_x0 = pad_x1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])
            pad_y0 = pad_y1 = np.random.randint(self.pad_bounds[0], self.pad_bounds[1])

            rgbs_view = [np.pad(rgb, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0))) for rgb in rgbs[v]]
            depths_view = [np.pad(depth, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0))) for depth in depths[v]]
            intrs[v, :, 0, 2] += pad_x0
            intrs[v, :, 1, 2] += pad_y0
            trajs[v, :, :, 0] += pad_x0
            trajs[v, :, :, 1] += pad_y0
            H_padded, W_padded = rgbs_view[0].shape[:2]

            scale_x = np.random.uniform(self.resize_lim[0], self.resize_lim[1])
            scale_y = scale_x + np.random.uniform(-0.01, 0.01)
            scale_y = max(self.resize_lim[0], min(self.resize_lim[1], scale_y))
            H_new = max(int(H_padded * scale_y) + int(H_padded * scale_y) % 2, crop_size[0] + 10)
            W_new = max(int(W_padded * scale_x) + int(W_padded * scale_x) % 2, crop_size[1] + 10)
            scale_x = W_new / W_padded
            scale_y = H_new / H_padded

            for t in range(T):
                rgbs_view[t] = cv2.resize(rgbs_view[t], (W_new, H_new), interpolation=cv2.INTER_LINEAR)
                depths_view[t] = cv2.resize(depths_view[t], (W_new, H_new), interpolation=cv2.INTER_NEAREST)
            intrs[v, :, 0, :] *= scale_x
            intrs[v, :, 1, :] *= scale_y
            trajs[v, :, :, 0] *= scale_x
            trajs[v, :, :, 1] *= scale_y

            for t in range(T):
                cx = intrs[v, t, 0, 2]
                cy = intrs[v, t, 1, 2]
                x0 = round(cx - crop_size[1] / 2)
                y0 = round(cy - crop_size[0] / 2)

                H_new, W_new = rgbs_view[t].shape[:2]
                assert x0 >= 0
                assert y0 >= 0
                assert (H_new - crop_size[0]) >= 0
                assert (W_new - crop_size[1]) >= 0
                assert (H_new - crop_size[0]) >= y0
                assert (W_new - crop_size[1]) >= x0

                rgbs_view[t] = rgbs_view[t][y0:y0 + crop_size[0], x0:x0 + crop_size[1]]
                depths_view[t] = depths_view[t][y0:y0 + crop_size[0], x0:x0 + crop_size[1]]
                intrs[v, t, 0, 2] -= x0
                intrs[v, t, 1, 2] -= y0
                trajs[v, t, :, 0] -= x0
                trajs[v, t, :, 1] -= y0

                # Assert principal point is centered
                assert rgbs_view[t].shape[0] == crop_size[0]
                assert rgbs_view[t].shape[1] == crop_size[1]
                assert np.allclose(intrs[v, t, 0, 2], crop_size[1] / 2, atol=0.01)
                assert np.allclose(intrs[v, t, 1, 2], crop_size[0] / 2, atol=0.01)

            rgbs_new[v] = np.stack(rgbs_view)
            depths_new[v] = np.stack(depths_view)[..., None]

        visibles = (visibles &
                    (trajs[..., 0] >= 0) &
                    (trajs[..., 1] >= 0) &
                    (trajs[..., 0] < crop_size[1]) &
                    (trajs[..., 1] < crop_size[0]))

        return rgbs_new, depths_new, intrs, trajs, visibles

    def _rescale_and_erase_depth_patches(self, depths, trajs, visibles, rndstate):
        V, T, H, W, _ = depths.shape
        _, _, N, _ = trajs.shape
        assert depths.dtype == np.float32
        assert depths.shape == (V, T, H, W, 1)
        assert trajs.shape == (V, T, N, 3)
        assert visibles.shape == (V, T, N)

        depths = depths.copy()
        visibles = visibles.copy()

        ############ eraser transform (per image after the first) ############
        for v in range(V):
            for i in range(1, T):
                if rndstate.rand() < self.eraser_aug_prob:
                    n = rndstate.randint(1, self.eraser_max + 1)  # number of times to occlude
                    for _ in range(n):
                        xc = rndstate.randint(0, W)
                        yc = rndstate.randint(0, H)
                        dx = rndstate.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                        dy = rndstate.randint(self.eraser_bounds[0], self.eraser_bounds[1])
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)
                        eraser_depth = {
                            0: depths[v, i, y0:y1, x0:x1].mean(),
                            1: depths[v, i, y0:y1, x0:x1].min(),
                            2: depths[v, i, y0:y1, x0:x1].max(),
                            3: 0,
                        }[rndstate.choice([0, 1, 2, 3], p=[0.2, 0.1, 0.35, 0.35])]
                        depths[v, i, y0:y1, x0:x1] = eraser_depth
                        occ_inds = np.logical_and(
                            np.logical_and(trajs[v, i, :, 0] >= x0, trajs[v, i, :, 0] < x1),
                            np.logical_and(trajs[v, i, :, 1] >= y0, trajs[v, i, :, 1] < y1),
                        )
                        visibles[v, i, occ_inds] = 0

        ############ replace transform (per image after the first) ############
        for v in range(V):
            for i in range(1, T):
                if rndstate.rand() < self.replace_aug_prob:
                    n = rndstate.randint(1, self.replace_max + 1)  # number of times to occlude
                    for _ in range(n):
                        xc = rndstate.randint(0, W)
                        yc = rndstate.randint(0, H)
                        dx = rndstate.randint(self.replace_bounds[0], self.replace_bounds[1])
                        dy = rndstate.randint(self.replace_bounds[0], self.replace_bounds[1])
                        x0 = np.clip(xc - dx / 2, 0, W - 1).round().astype(np.int32)
                        x1 = np.clip(xc + dx / 2, 0, W - 1).round().astype(np.int32)
                        y0 = np.clip(yc - dy / 2, 0, H - 1).round().astype(np.int32)
                        y1 = np.clip(yc + dy / 2, 0, H - 1).round().astype(np.int32)
                        wid = x1 - x0
                        hei = y1 - y0
                        y00 = rndstate.randint(0, H - hei)
                        x00 = rndstate.randint(0, W - wid)
                        v_rnd = rndstate.randint(0, V)
                        i_rnd = rndstate.randint(0, T)
                        depths[v, i, y0:y1, x0:x1] = depths[v_rnd, i_rnd, y00: y00 + hei, x00: x00 + wid]
                        occ_inds = np.logical_and(
                            np.logical_and(trajs[v, i, :, 0] >= x0, trajs[v, i, :, 0] < x1),
                            np.logical_and(trajs[v, i, :, 1] >= y0, trajs[v, i, :, 1] < y1),
                        )
                        visibles[v, i, occ_inds] = 0
        return depths, visibles

    def _crop(self, rgbs, trajs, crop_size):
        T, N, _ = trajs.shape

        S = len(rgbs)
        H, W = rgbs[0].shape[:2]
        assert S == T

        ############ spatial transform ############

        H_new = H
        W_new = W

        # simple random crop
        y0 = 0 if crop_size[0] >= H_new else (H_new - crop_size[0]) // 2
        # np.random.randint(0,
        x0 = 0 if crop_size[1] >= W_new else np.random.randint(0, W_new - crop_size[1])
        rgbs = [rgb[y0: y0 + crop_size[0], x0: x0 + crop_size[1]] for rgb in rgbs]

        trajs[:, :, 0] -= x0
        trajs[:, :, 1] -= y0

        return np.stack(rgbs), trajs
