import json
import pickle
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
from mvtracker.datasets.tapvid3d_multiview_dataset import (
    _camera_rig_anchor,
    _recenter_world_coordinates,
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
        jpeg_root = sequence / f"view_{view}"
        jpeg_root.mkdir()
        payloads = [bytes([0xFF, 0xD8, view, frame, 0xFF, 0xD9]) for frame in range(frames)]
        offsets = np.zeros(frames + 1, dtype=np.int64)
        with (jpeg_root / "jpeg_bytes.bin").open("wb") as handle:
            for frame, payload in enumerate(payloads):
                handle.write(payload)
                offsets[frame + 1] = offsets[frame] + len(payload)
        np.save(jpeg_root / "jpeg_offsets.npy", offsets)
    (sequence / "manifest.json").write_text(
        json.dumps(
            {
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
    dataset._mmap_cache_sequences = 1
    return dataset


class Syn4DLoaderTests(unittest.TestCase):
    def test_camera_rig_recentering_preserves_projection(self):
        centres = np.asarray(((1000.0, 20.0, -3.0), (1002.0, 22.0, -3.0)))
        extrinsics = np.repeat(
            np.eye(4, dtype=np.float32)[None, None], 2, axis=0
        )
        extrinsics[:, 0, :3, 3] = -centres
        tracks = np.asarray([[[1010.0, 24.0, 7.0]]], dtype=np.float32)

        anchor = _camera_rig_anchor(extrinsics)
        centred_tracks, centred_extrinsics = _recenter_world_coordinates(
            tracks,
            extrinsics[:, :, :3, :4],
            anchor,
        )

        np.testing.assert_allclose(anchor, centres.mean(axis=0))
        for view in range(2):
            original_camera = (
                tracks[0, 0]
                @ extrinsics[view, 0, :3, :3].T
                + extrinsics[view, 0, :3, 3]
            )
            centred_camera = (
                centred_tracks[0, 0]
                @ centred_extrinsics[view, 0, :3, :3].T
                + centred_extrinsics[view, 0, :3, 3]
            )
            np.testing.assert_allclose(centred_camera, original_camera)

    def test_dataset_recreates_mmap_cache_in_spawned_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            restored = pickle.loads(pickle.dumps(dataset))
            self.assertEqual(restored._sequence_cache.maximum, 1)
            self.assertIsNot(restored._sequence_cache, dataset._sequence_cache)

    def test_motion_preselection_fills_missing_buckets(self):
        movement = np.array([0.0] * 8 + [0.5] * 16, dtype=np.float32)
        selected = _preselect_tracks(
            movement,
            np.ones_like(movement, dtype=bool),
            np.random.RandomState(3),
            ratio_dynamic=0.5,
            ratio_very_dynamic=0.25,
            maximum=24,
        )
        self.assertEqual(selected.size, 24)
        self.assertEqual(np.unique(selected).size, 24)

    def test_motion_preselection_uses_every_candidate_without_duplicates(self):
        movement = np.array([0.0] * 100 + [0.5] * 80 + [3.0] * 20, dtype=np.float32)
        selected = _preselect_tracks(
            movement,
            np.ones_like(movement, dtype=bool),
            np.random.RandomState(3),
            ratio_dynamic=0.5,
            ratio_very_dynamic=0.25,
            maximum=200,
        )
        self.assertEqual(selected.size, 200)
        self.assertEqual(np.unique(selected).size, 200)

    def test_factory_maps_the_direct_sequence_cache(self):
        config = SimpleNamespace(
            datasets={
                "syn4d_num_views": 6,
                "syn4d_mmap_cache_sequences": 2,
                "syn4d_max_depth": 321.0,
            }
        )
        kwargs = Syn4DMultiViewDataset.from_name(
            "syn4d-multiview-validation",
            "/datasets/syn4d-mvtracker",
            training_args=config,
            just_return_kwargs=True,
            storage_split="shared",
        )
        self.assertEqual(
            kwargs["data_root"],
            "/datasets/syn4d-mvtracker/shared",
        )
        self.assertEqual(kwargs["num_views"], 6)
        self.assertEqual(kwargs["mmap_cache_sequences"], 2)
        self.assertEqual(kwargs["max_depth"], 321.0)
        self.assertFalse(kwargs["enable_variable_depth_type_augs"])

    def test_plans_window_views_tracks_and_materializes_indexed_jpegs(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            request = SimpleNamespace(virtual_index=0, scene_index=0, view_count=6)

            plan = dataset.plan_sample(request)
            self.assertIsNotNone(plan)
            self.assertEqual(plan.image_codec, "jpeg")
            self.assertEqual(len(plan.views), 6)
            self.assertEqual(plan.frame_indices.shape, (3,))
            self.assertLessEqual(plan.track_count, 8)
            self.assertEqual(plan.track_validity.shape, (3, plan.track_count))
            self.assertTrue((~plan.track_validity).any())

            sample, gotit = dataset.materialize_sample(plan)
            self.assertTrue(gotit)
            self.assertEqual(sample.image_codec, "jpeg")
            self.assertEqual(len(sample.jpeg_bytes), 6 * 3)
            self.assertTrue(all(str(encoded.dtype) == "torch.uint8" for encoded in sample.jpeg_bytes))
            self.assertEqual(sample.depth.shape, (6, 3, 1, 8, 10))
            self.assertEqual(sample.valid.shape, (3, plan.track_count))
            self.assertTrue((~sample.valid.bool()).any())
            self.assertEqual(len(dataset._sequence_cache.stores), 1)


if __name__ == "__main__":
    unittest.main()
