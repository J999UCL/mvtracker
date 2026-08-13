import importlib.util
import json
import sys
import tempfile
import types
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image


def _load_module():
    class StubKubric:
        @staticmethod
        def from_name(*args, **kwargs):
            training_args = kwargs.get("training_args")
            augmentations = getattr(training_args, "augmentations", {})
            get_aug = (
                augmentations.get
                if isinstance(augmentations, dict)
                else lambda name, default: getattr(augmentations, name, default)
            )
            return {
                "enable_rgb_augs": get_aug("rgb", False),
                "enable_depth_augs": get_aug("depth", False),
                "enable_variable_depth_type_augs": get_aug("variable_depth_type", False),
                "enable_variable_num_views_augs": get_aug("variable_num_views", False),
                "enable_variable_trajpersample_augs": get_aug("variable_trajpersample", False),
                "normalize_scene_following_vggt": False,
            }

    def legal_starts(frame_count, sequence_length, invalid_frame_indices=()):
        del invalid_frame_indices
        return np.arange(max(0, frame_count - sequence_length + 1), dtype=np.int64)

    @dataclass
    class StubDatapoint:
        video: object
        segmentation: object
        videodepth: object = None
        trajectory: object = None
        trajectory_3d: object = None
        visibility: object = None
        valid: object = None
        seq_name: object = None
        intrs: object = None
        extrs: object = None
        query_points: object = None
        query_points_3d: object = None
        sample_metadata: object = None
        track_upscaling_factor: object = 1.0

    kubric = types.ModuleType("mvtracker.datasets.kubric_multiview_dataset")
    kubric.KubricMultiViewDataset = StubKubric
    kubric._legal_contiguous_window_starts = legal_starts
    utils = types.ModuleType("mvtracker.datasets.utils")
    utils.Datapoint = StubDatapoint
    utils.aug_depth = lambda depth, **kwargs: depth
    package = types.ModuleType("mvtracker")
    package.__path__ = []
    datasets = types.ModuleType("mvtracker.datasets")
    datasets.__path__ = []
    path = Path(__file__).resolve().parents[1] / "mvtracker/datasets/tapvid3d_multiview_dataset.py"
    name = "_tapvid3d_loader_under_test"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    stubs = {
        "mvtracker": package,
        "mvtracker.datasets": datasets,
        "mvtracker.datasets.kubric_multiview_dataset": kubric,
        "mvtracker.datasets.utils": utils,
        name: module,
    }
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


loader = _load_module()


def _jpeg_bytes(height, width, value):
    image = Image.fromarray(np.full((height, width, 3), value, dtype=np.uint8))
    import io

    output = io.BytesIO()
    image.save(output, format="JPEG", quality=95)
    return np.frombuffer(output.getvalue(), dtype=np.uint8)


def _write_raw(root: Path, *, frames=6, points=6, views=2, height=8, width=10):
    sequence = root / "train/scene-alpha"
    sequence.mkdir(parents=True)
    tracks = np.zeros((frames, points, 3), dtype=np.float32)
    tracks[..., 0] = np.linspace(-0.5, 0.5, points)
    tracks[..., 1] = np.linspace(-0.3, 0.3, points)
    tracks[..., 2] = 2.0
    tracks[:, 0, 0] += np.arange(frames, dtype=np.float32) * 0.1
    np.save(sequence / "tracks_xyz.npy", tracks)
    np.save(sequence / "queries_xytv.npy", np.zeros((points, 4), dtype=np.float32))
    for view in range(views):
        view_root = sequence / str(view)
        view_root.mkdir()
        images = np.empty(frames, dtype=object)
        for frame in range(frames):
            images[frame] = _jpeg_bytes(height, width, 20 + frame + view)
        np.save(view_root / "images_jpeg_bytes.npy", images, allow_pickle=True)
        np.save(view_root / "intrinsics.npy", np.asarray([5, 5, width / 2, height / 2], dtype=np.float32))
        np.save(view_root / "extrinsics_w2c.npy", np.repeat(np.eye(4, dtype=np.float32)[None], frames, axis=0))
        np.save(view_root / "visibility.npy", np.ones((frames, points), dtype=np.bool_))
        np.save(view_root / "depth.npy", np.full((frames, height, width), 2, dtype=np.float32))
        np.save(view_root / "foreground_mask.npy", np.ones((frames, height, width), dtype=np.bool_))
    return sequence


def _dataset(root: Path, *, points=16, views=4):
    _write_raw(root / "raw", points=points, views=views)
    loader.prepare_tapvid3d_cache(root / "raw", root / "cache")
    dataset = loader.TapVid3DMultiViewDataset.__new__(loader.TapVid3DMultiViewDataset)
    dataset.raw_root = root / "raw/train"
    dataset.data_root = str(root / "cache/train")
    dataset.seq_names = ["scene-alpha"]
    dataset.real_len = 1
    dataset.seq_len = 3
    dataset.num_views = views
    dataset.traj_per_sample = 8
    dataset.seed = 12
    dataset.add_index_to_seed = True
    dataset.crop_size = (8, 10)
    dataset.enable_cropping_augs = False
    dataset.enable_rgb_augs = True
    dataset.enable_depth_augs = True
    dataset.enable_variable_trajpersample_augs = True
    dataset.enable_variable_num_views_augs = False
    dataset.enable_scene_transform_augs = False
    dataset.enable_camera_params_noise_augs = False
    dataset.ratio_dynamic = 0.0
    dataset.ratio_very_dynamic = 0.0
    dataset.max_tracks_to_preload = None
    dataset.augmentation_probability = 1.0
    dataset.eraser_aug_prob = 0.5
    dataset.eraser_max = 10
    dataset.eraser_bounds = [2, 100]
    dataset.replace_aug_prob = 0.5
    dataset.replace_max = 10
    dataset.replace_bounds = [2, 100]
    dataset.max_depth = 24
    return dataset


class CacheTests(unittest.TestCase):
    def test_cache_preserves_jpegs_and_reuses_clean_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_raw(root / "raw")
            cache = root / "cache"
            self.assertEqual(
                loader.prepare_tapvid3d_cache(root / "raw", cache, workers=2),
                {"prepared": 1, "reused": 0},
            )
            manifest = json.loads((cache / "train/scene-alpha/manifest.json").read_text())
            self.assertEqual(manifest["frame_count"], 6)
            self.assertEqual(manifest["point_count"], 6)
            offsets = np.load(cache / "train/scene-alpha/view_0/jpeg_offsets.npy")
            packed = (cache / "train/scene-alpha/view_0/jpeg_bytes.bin").read_bytes()
            original = np.load(source / "0/images_jpeg_bytes.npy", allow_pickle=True)
            for frame in range(6):
                self.assertEqual(packed[offsets[frame]:offsets[frame + 1]], original[frame].tobytes())
            self.assertEqual(
                loader.prepare_tapvid3d_cache(root / "raw", cache),
                {"prepared": 0, "reused": 1},
            )

    def test_incomplete_cache_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_raw(root / "raw")
            cache = root / "cache"
            loader.prepare_tapvid3d_cache(root / "raw", cache)
            (cache / "train/scene-alpha/view_0/jpeg_bytes.bin").unlink()
            self.assertEqual(
                loader.prepare_tapvid3d_cache(root / "raw", cache),
                {"prepared": 1, "reused": 0},
            )


class SelectiveLoaderTests(unittest.TestCase):
    def test_reads_only_selected_window_and_returns_encoded_jpegs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_raw(root / "raw")
            loader.prepare_tapvid3d_cache(root / "raw", root / "cache")
            dataset = loader.TapVid3DMultiViewDataset.__new__(loader.TapVid3DMultiViewDataset)
            dataset.raw_root = root / "raw/train"
            dataset.data_root = str(root / "cache/train")
            dataset.seq_names = ["scene-alpha"]
            dataset.real_len = 1
            dataset.seq_len = 3
            dataset.num_views = 2
            dataset.traj_per_sample = 4
            dataset.seed = 12
            dataset.add_index_to_seed = True
            dataset.crop_size = (8, 10)
            dataset.enable_cropping_augs = False
            dataset.enable_depth_augs = False
            dataset.enable_rgb_augs = False
            dataset.enable_variable_trajpersample_augs = False
            dataset.enable_variable_num_views_augs = False
            dataset.enable_scene_transform_augs = False
            dataset.enable_camera_params_noise_augs = False
            dataset.augmentation_probability = 0
            dataset.ratio_dynamic = 0.0
            dataset.ratio_very_dynamic = 0.0
            dataset.max_tracks_to_preload = None

            sample, gotit = dataset[0]
            self.assertTrue(gotit)
            self.assertEqual(len(sample.jpeg_bytes), 2 * 3)
            self.assertEqual(sample.depth.shape, (2, 3, 1, 8, 10))
            self.assertEqual(sample.trajectory_3d.shape, (3, 4, 3))
            self.assertEqual(sample.metadata["window_end_exclusive"] - sample.metadata["window_start"], 3)
            self.assertEqual(sample.metadata["motion_track_count"], 4)
            self.assertIn("motion_window_mean_m", sample.metadata)
            self.assertIn("motion_full_dynamic_window_static_count", sample.metadata)
            with mock.patch.object(loader.np, "load", wraps=loader.np.load) as np_load:
                dataset[1]
            self.assertFalse(any(
                str(call.args[0]).endswith("images_jpeg_bytes.npy")
                for call in np_load.call_args_list
            ))

    def test_stale_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = _write_raw(root / "raw")
            loader.prepare_tapvid3d_cache(root / "raw", root / "cache")
            depth = np.load(source / "0/depth.npy")
            np.save(source / "0/depth.npy", depth)
            dataset = loader.TapVid3DMultiViewDataset.__new__(loader.TapVid3DMultiViewDataset)
            dataset.raw_root = root / "raw/train"
            dataset.data_root = str(root / "cache/train")
            dataset.seq_names = ["scene-alpha"]
            dataset.real_len = 1
            dataset.seq_len = 3
            dataset.num_views = 2
            dataset.traj_per_sample = 4
            dataset.seed = 12
            dataset.add_index_to_seed = True
            dataset.crop_size = (8, 10)
            dataset.enable_cropping_augs = False
            dataset.enable_depth_augs = False
            dataset.enable_rgb_augs = False
            dataset.enable_variable_trajpersample_augs = False
            dataset.enable_variable_num_views_augs = False
            dataset.enable_scene_transform_augs = False
            dataset.enable_camera_params_noise_augs = False
            dataset.augmentation_probability = 0
            dataset.ratio_dynamic = 0.0
            dataset.ratio_very_dynamic = 0.0
            dataset.max_tracks_to_preload = None
            with self.assertRaisesRegex(ValueError, "cache is stale"):
                dataset[0]


class FromNameTests(unittest.TestCase):
    def test_maps_generic_split_and_directories(self):
        config = types.SimpleNamespace(datasets={
            "tapvid3d_raw_dir": "raw-data",
            "tapvid3d_cache_dir": "indexed-data",
            "tapvid3d_num_views": 3,
        })
        kwargs = loader.TapVid3DMultiViewDataset.from_name(
            "tapvid3d-multiview-validation",
            "/datasets",
            training_args=config,
            just_return_kwargs=True,
        )
        self.assertEqual(kwargs["data_root"], "/datasets/indexed-data/validation")
        self.assertEqual(kwargs["raw_root"], "/datasets/raw-data/validation")
        self.assertEqual(kwargs["num_views"], 3)

    def test_training_keeps_rgb_depth_and_variable_view_augmentations(self):
        config = types.SimpleNamespace(
            datasets={
                "tapvid3d_raw_dir": "raw-data",
                "tapvid3d_cache_dir": "indexed-data",
                "tapvid3d_num_views": 4,
            },
            augmentations=types.SimpleNamespace(
                rgb=True,
                depth=True,
                variable_depth_type=False,
                variable_num_views=True,
                variable_trajpersample=True,
            ),
        )
        kwargs = loader.TapVid3DMultiViewDataset.from_name(
            "tapvid3d-multiview-training",
            "/datasets",
            training_args=config,
            fabric=object(),
            just_return_kwargs=True,
        )
        self.assertTrue(kwargs["enable_rgb_augs"])
        self.assertTrue(kwargs["enable_depth_augs"])
        self.assertTrue(kwargs["enable_variable_num_views_augs"])
        self.assertTrue(kwargs["enable_variable_trajpersample_augs"])
        self.assertFalse(kwargs["enable_variable_depth_type_augs"])
        self.assertIsNone(kwargs["num_views"])


class MvTrackerSamplingParityTests(unittest.TestCase):
    def test_visible_path_length_only_counts_consecutively_visible_steps(self):
        tracks = np.zeros((4, 2, 3), dtype=np.float32)
        tracks[:, 0, 0] = [0.0, 0.1, 0.3, 0.6]
        tracks[:, 1, 0] = [0.0, 1.0, 2.0, 3.0]
        visibility = np.ones((1, 4, 2), dtype=np.bool_)
        visibility[:, 2, 1] = False

        np.testing.assert_allclose(
            loader._visible_path_lengths(tracks, visibility),
            [0.6, 1.0],
        )

    def test_variable_views_are_uniformly_sampled_from_one_to_four(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            dataset.num_views = None
            dataset.enable_variable_num_views_augs = True
            dataset.enable_rgb_augs = False
            dataset.enable_depth_augs = False

            counts = np.zeros(4, dtype=np.int64)
            for index in range(400):
                sample, gotit = dataset[index]
                self.assertTrue(gotit)
                selected_views = sample.metadata["selected_views"]
                self.assertEqual(len(selected_views), len(set(selected_views)))
                self.assertTrue(set(selected_views).issubset({0, 1, 2, 3}))
                counts[len(selected_views) - 1] += 1

            np.testing.assert_allclose(counts, np.full(4, 100), atol=30)

    def test_shared_augmentation_gate_controls_rgb_depth_and_track_count(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = _dataset(Path(directory))
            dataset.augmentation_probability = 0.0
            plain, gotit = dataset[20]
            self.assertTrue(gotit)
            self.assertEqual(plain.trajectory_3d.shape[1], 8)
            self.assertFalse(plain.apply_rgb_aug)
            self.assertFalse(plain.apply_depth_aug)

            dataset.augmentation_probability = 1.0
            augmented, gotit = dataset[20]
            self.assertTrue(gotit)
            self.assertEqual(augmented.trajectory_3d.shape[1], 1)
            self.assertTrue(augmented.apply_rgb_aug)
            self.assertTrue(augmented.apply_depth_aug)

    def test_query_policy_uses_first_visibility_for_three_quarters(self):
        tracks = np.zeros((5, 8, 3), dtype=np.float32)
        tracks[..., 0] = np.arange(8, dtype=np.float32)
        xy = np.zeros((1, 5, 8, 2), dtype=np.float32)
        camera_z = np.ones((1, 5, 8), dtype=np.float32)
        visibility = np.zeros((1, 5, 8), dtype=np.bool_)
        visibility[:, 1, :] = True
        visibility[:, 2:, :] = True

        selected, queries = loader._sample_tracks(
            tracks,
            xy,
            camera_z,
            visibility,
            8,
            np.random.RandomState(7),
            augment_this_datapoint=False,
            enable_variable_trajpersample_augs=False,
            sample_index=1,
        )

        self.assertEqual(len(selected), 8)
        query_times = queries[:, 0].astype(np.int64)
        np.testing.assert_array_equal(query_times[2:], np.ones(6, dtype=np.int64))
        self.assertTrue(np.all(np.isin(query_times[:2], [1, 2, 3])))
        self.assertFalse(np.any(query_times == 4))
        self.assertTrue(np.all(visibility.any(axis=0)[query_times, selected]))


class SpatialTransformTests(unittest.TestCase):
    def test_identity_transform_preserves_projection(self):
        xy = np.asarray([[[[1.0, 2.0], [8.0, 6.0]]]], dtype=np.float32)
        visibility = np.ones((1, 1, 2), dtype=np.bool_)
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, None], 1, axis=0)
        transformed, visible, adjusted, theta = loader._spatial_transform(
            xy, visibility, intrinsics, (8, 10), (8, 10), np.random.RandomState(0), False
        )
        np.testing.assert_allclose(transformed, xy)
        np.testing.assert_array_equal(visible, visibility)
        np.testing.assert_allclose(adjusted, intrinsics)
        np.testing.assert_allclose(theta[0, 0], np.asarray([[1, 0, 0], [0, 1, 0]]))

    def test_non_square_resize_uses_integer_pixel_centers(self):
        xy = np.asarray([[[[0.0, 0.0], [9.0, 7.0]]]], dtype=np.float32)
        visibility = np.ones((1, 1, 2), dtype=np.bool_)
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, None], 1, axis=0)
        transformed, _, adjusted, _ = loader._spatial_transform(
            xy, visibility, intrinsics, (8, 10), (4, 6), np.random.RandomState(0), False
        )
        np.testing.assert_allclose(transformed[0, 0], [[0, 0], [5, 3]], atol=1e-6)
        self.assertAlmostEqual(float(adjusted[0, 0, 0, 0]), 5 / 9)
        self.assertAlmostEqual(float(adjusted[0, 0, 1, 1]), 3 / 7)


if __name__ == "__main__":
    unittest.main()
