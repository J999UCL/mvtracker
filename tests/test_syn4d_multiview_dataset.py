import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mvtracker.datasets.syn4d_multiview_dataset import (
    Syn4DMultiViewDataset,
    _SequenceMmapCache,
    _preselect_tracks,
)


def _write_sequence(root: Path, name="temple_group__seq_000000"):
    sequence = root / name
    sequence.mkdir(parents=True)
    frames, points, height, width = 6, 32, 8, 10
    tracks = np.zeros((frames, points, 3), dtype=np.float32)
    tracks[..., 0] = np.linspace(-0.4, 0.4, points)
    tracks[..., 2] = 2.0
    tracks[:, 8:24, 0] += np.arange(frames, dtype=np.float32)[:, None] * 0.1
    tracks[:, 24:, 0] += np.arange(frames, dtype=np.float32)[:, None]
    valid = np.ones((frames, points), dtype=np.bool_)
    valid[1] = False
    np.save(sequence / "tracks_xyz.npy", tracks)
    np.save(sequence / "track_valid.npy", valid)
    np.save(
        sequence / "motion_path_length.npy",
        np.linalg.norm(np.diff(tracks, axis=0), axis=-1).sum(axis=0).astype(np.float32),
    )
    np.save(sequence / "queries_xytv.npy", np.zeros((points, 4), dtype=np.float32))
    intrinsics = np.asarray(
        [[5.0, 0.0, width / 2], [0.0, 5.0, height / 2], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    for view in range(8):
        view_root = sequence / str(view)
        view_root.mkdir()
        np.save(
            view_root / "rgb.npy",
            np.full((frames, 3, height, width), 10 + view, dtype=np.uint8),
        )
        np.save(
            view_root / "depth.npy",
            np.full((frames, height, width), 2.0, dtype=np.float32),
        )
        np.save(
            view_root / "intrinsics.npy",
            np.repeat(intrinsics[None], frames, axis=0),
        )
        np.save(
            view_root / "extrinsics_w2c.npy",
            np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0),
        )
        np.save(
            view_root / "visibility.npy",
            np.ones((frames, points), dtype=np.bool_),
        )
    (sequence / "manifest.json").write_text(
        json.dumps(
            {
                "format": "syn4d-tapvid-mv",
                "version": 1,
                "frames": frames,
                "tracks": points,
                "views": 8,
                "cache_resolution": [height, width],
            }
        )
    )
    return name


def _dataset(root: Path):
    name = _write_sequence(root)
    dataset = Syn4DMultiViewDataset.__new__(Syn4DMultiViewDataset)
    dataset.data_root = str(root)
    dataset.seq_names = [name]
    dataset.real_len = 1
    dataset.seq_len = 3
    dataset.num_views = None
    dataset.view_count_probabilities = (1 / 6,) * 6
    dataset.traj_per_sample = 8
    dataset.seed = 72
    dataset.add_index_to_seed = True
    dataset.crop_size = (8, 10)
    dataset.enable_cropping_augs = False
    dataset.enable_rgb_augs = False
    dataset.enable_depth_augs = False
    dataset.enable_variable_trajpersample_augs = False
    dataset.enable_variable_num_views_augs = True
    dataset.enable_scene_transform_augs = False
    dataset.enable_camera_params_noise_augs = False
    dataset.augmentation_probability = 0.0
    dataset.ratio_dynamic = 0.5
    dataset.ratio_very_dynamic = 0.25
    dataset.max_tracks_to_preload = 24
    dataset.max_depth = 1000.0
    dataset.eraser_aug_prob = 0.5
    dataset.eraser_max = 10
    dataset.eraser_bounds = [2, 100]
    dataset.replace_aug_prob = 0.5
    dataset.replace_max = 10
    dataset.replace_bounds = [2, 100]
    dataset._manifests = {name: dataset._load_manifest(name)}
    dataset._sequence_cache = _SequenceMmapCache(root, maximum=1)
    return dataset


class Syn4DLoaderTests(unittest.TestCase):
    def test_motion_preselection_does_not_substitute_missing_pools(self):
        movement = np.array([0.0] * 8 + [0.5] * 16, dtype=np.float32)
        selected = _preselect_tracks(
            movement,
            np.ones_like(movement, dtype=bool),
            np.random.RandomState(3),
            ratio_dynamic=0.5,
            ratio_very_dynamic=0.25,
            maximum=24,
        )
        self.assertEqual(selected.size, 0)

    def test_factory_maps_the_direct_sequence_cache(self):
        config = SimpleNamespace(
            datasets={
                "syn4d_cache_version": "v1",
                "syn4d_num_views": 6,
                "syn4d_mmap_cache_sequences": 2,
            }
        )
        kwargs = Syn4DMultiViewDataset.from_name(
            "syn4d-multiview-validation",
            "/datasets/syn4d-mvtracker",
            training_args=config,
            just_return_kwargs=True,
        )
        self.assertEqual(
            kwargs["data_root"],
            "/datasets/syn4d-mvtracker/v1/validation",
        )
        self.assertEqual(kwargs["num_views"], 6)
        self.assertEqual(kwargs["mmap_cache_sequences"], 2)
        self.assertFalse(kwargs["enable_variable_depth_type_augs"])

    def test_plans_window_views_tracks_and_materializes_raw_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            request = SimpleNamespace(virtual_index=0, scene_index=0, view_count=6)

            plan = dataset.plan_sample(request)
            self.assertIsNotNone(plan)
            self.assertEqual(plan.image_codec, "raw")
            self.assertEqual(len(plan.views), 6)
            self.assertEqual(plan.frame_indices.shape, (3,))
            self.assertLessEqual(plan.track_count, 8)
            self.assertEqual(plan.track_validity.shape, (3, plan.track_count))
            self.assertTrue((~plan.track_validity).any())

            sample, gotit = dataset.materialize_sample(plan)
            self.assertTrue(gotit)
            self.assertEqual(sample.raw_rgb.shape, (6, 3, 3, 8, 10))
            self.assertEqual(str(sample.raw_rgb.dtype), "torch.uint8")
            self.assertEqual(sample.depth.shape, (6, 3, 1, 8, 10))
            self.assertEqual(sample.valid.shape, (3, plan.track_count))
            self.assertTrue((~sample.valid.bool()).any())
            self.assertEqual(len(dataset._sequence_cache.stores), 1)


if __name__ == "__main__":
    unittest.main()
