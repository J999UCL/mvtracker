import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image


def _load_loader_module():
    """Load the adapter with a minimal Kubric stub, avoiding optional dependencies."""
    class StubKubricMultiViewDataset:
        @staticmethod
        def from_name(*args, **kwargs):
            raise AssertionError("Tests must mock KubricMultiViewDataset.from_name")

    def read_json(filename):
        with open(filename, "r", encoding="utf-8") as handle:
            return json.load(handle)

    mvtracker_module = types.ModuleType("mvtracker")
    mvtracker_module.__path__ = []
    datasets_module = types.ModuleType("mvtracker.datasets")
    datasets_module.__path__ = []
    kubric_module = types.ModuleType("mvtracker.datasets.kubric_multiview_dataset")
    kubric_module.KubricMultiViewDataset = StubKubricMultiViewDataset
    utils_module = types.ModuleType("mvtracker.datasets.utils")
    utils_module.read_json = read_json

    source_path = (
        Path(__file__).resolve().parents[1]
        / "mvtracker"
        / "datasets"
        / "pointodyssey_multiview_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("_pointodyssey_loader_under_test", source_path)
    module = importlib.util.module_from_spec(spec)
    stubs = {
        "mvtracker": mvtracker_module,
        "mvtracker.datasets": datasets_module,
        "mvtracker.datasets.kubric_multiview_dataset": kubric_module,
        "mvtracker.datasets.utils": utils_module,
    }
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


loader = _load_loader_module()


class MotionBucketTests(unittest.TestCase):
    @staticmethod
    def _dataset():
        dataset = loader.PointOdysseyMultiViewDataset.__new__(
            loader.PointOdysseyMultiViewDataset
        )
        dataset.ratio_dynamic = 0.5
        dataset.ratio_very_dynamic = 0.25
        dataset.traj_per_sample = 512
        dataset.max_tracks_to_preload = 18000
        return dataset

    def test_preserves_kubric_ratios_when_very_dynamic_quota_is_available(self):
        ratios = self._dataset()._motion_bucket_ratios(
            total_tracks=2600,
            very_dynamic_tracks=128,
        )
        self.assertEqual(ratios, (0.5, 0.25))

    def test_reassigns_undersupplied_very_dynamic_quota_to_dynamic_tracks(self):
        ratios = self._dataset()._motion_bucket_ratios(
            total_tracks=2600,
            very_dynamic_tracks=127,
        )
        self.assertEqual(ratios, (0.75, 0.0))


class PreparedSceneTests(unittest.TestCase):
    FRAME_COUNT = 2
    POINT_COUNT = 2
    HEIGHT = 2
    WIDTH = 3

    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.scene_path = Path(self.temporary_directory.name) / "train" / "000000"
        self.scene_path.mkdir(parents=True)
        self.constants = mock.patch.multiple(
            loader,
            _POINT_COUNT=self.POINT_COUNT,
            _HEIGHT=self.HEIGHT,
            _WIDTH=self.WIDTH,
        )
        self.constants.start()
        self.addCleanup(self.constants.stop)

        self.tracks = np.asarray(
            [
                [[1.0, 2.0, 1.0], [2.0, 1.0, 2.0]],
                [[3.0, 2.0, 1.0], [4.0, 2.0, 2.0]],
            ],
            dtype=np.float32,
        )
        self.intrinsics = np.asarray(
            [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        self.extrinsics = np.zeros((self.FRAME_COUNT, 3, 4), dtype=np.float32)
        self.extrinsics[:, :3, :3] = np.eye(3, dtype=np.float32)
        self.depth = np.asarray(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            dtype=np.float32,
        )
        self.visibility = np.asarray([[True, False], [False, True]], dtype=np.bool_)
        self._write_scene()

    def _metadata(self):
        return {
            "schema_version": 4,
            "format": "pointodyssey_mvtracker_preprocessed",
            "split": "train",
            "scene_id": "000000",
            "output": {
                "frame_count": self.FRAME_COUNT,
                "views": [0, 1, 2, 3],
                "resolution_hw": [self.HEIGHT, self.WIDTH],
                "rgb": {
                    "format": "jpeg",
                    "invalid_frame_indices": [1],
                },
                "depth": {
                    "format": "npy",
                    "dtype": "float32",
                    "semantics": "optical_z_meters",
                    "invalid_value": 0.0,
                    "clipped": False,
                },
                "visibility": {
                    "format": "npy",
                    "dtype": "bool",
                },
            },
        }

    def _write_scene(self):
        (self.scene_path / "scene.json").write_text(
            json.dumps(self._metadata()),
            encoding="utf-8",
        )
        np.save(self.scene_path / "tracks_3d.npy", self.tracks)
        colors = ((220, 40, 10), (20, 180, 70))
        for view in range(4):
            view_path = self.scene_path / f"view_{view}"
            view_path.mkdir()
            np.save(view_path / "depth.npy", self.depth)
            np.save(view_path / "intrinsics.npy", self.intrinsics)
            np.save(view_path / "extrinsics_w2c.npy", self.extrinsics)
            np.save(view_path / "visibility.npy", self.visibility)
            for frame, color in enumerate(colors):
                pixels = np.empty((self.HEIGHT, self.WIDTH, 3), dtype=np.uint8)
                pixels[:] = color
                Image.fromarray(pixels, mode="RGB").save(
                    view_path / f"rgba_{frame:05d}.jpg",
                    format="JPEG",
                    quality=100,
                    subsampling=0,
                )

    def test_reads_exact_prepared_contract(self):
        datapoint = loader.PointOdysseyMultiViewDataset.getitem_raw_datapoint(self.scene_path)

        self.assertEqual(set(datapoint), {"tracks_3d", "views", "invalid_rgb_frame_indices"})
        self.assertEqual(datapoint["invalid_rgb_frame_indices"], [1])
        self.assertEqual(datapoint["tracks_3d"].dtype, torch.float32)
        np.testing.assert_array_equal(datapoint["tracks_3d"].numpy(), self.tracks)
        self.assertEqual(len(datapoint["views"]), 4)

        expected_tracks_2d = np.asarray(
            [
                [[2.0, 6.0], [2.0, 1.5]],
                [[6.0, 6.0], [4.0, 3.0]],
            ],
            dtype=np.float32,
        )
        for view in datapoint["views"]:
            self.assertEqual(view["rgba"].shape, (2, 2, 3, 3))
            self.assertEqual(view["rgba"].dtype, torch.uint8)
            first_pixel = view["rgba"][0, 0, 0].numpy()
            self.assertGreater(first_pixel[0], first_pixel[2])  # RGB, not OpenCV BGR.
            np.testing.assert_allclose(first_pixel, [220, 40, 10], atol=3)
            self.assertEqual(view["depth"].shape, (2, 2, 3, 1))
            self.assertEqual(view["depth"].dtype, torch.float32)
            np.testing.assert_array_equal(view["depth"].numpy()[..., 0], self.depth)
            np.testing.assert_array_equal(view["intrinsics"].numpy(), self.intrinsics)
            np.testing.assert_array_equal(view["extrinsics"].numpy(), self.extrinsics)
            np.testing.assert_allclose(view["tracks_2d"].numpy(), expected_tracks_2d)
            np.testing.assert_array_equal(view["occlusion"].numpy(), ~self.visibility)

    def test_wrong_schema_is_fatal(self):
        metadata = self._metadata()
        metadata["schema_version"] = 2
        (self.scene_path / "scene.json").write_text(json.dumps(metadata), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "schema_version"):
            loader.PointOdysseyMultiViewDataset.getitem_raw_datapoint(self.scene_path)

    def test_missing_rgb_frame_is_fatal(self):
        (self.scene_path / "view_2" / "rgba_00001.jpg").unlink()

        with self.assertRaisesRegex(ValueError, "exact contiguous RGB sequence"):
            loader.PointOdysseyMultiViewDataset.getitem_raw_datapoint(self.scene_path)


class FromNameTests(unittest.TestCase):
    @staticmethod
    def _base_kwargs(**overrides):
        kwargs = {
            "data_root": "/unused",
            "use_duster_depths": False,
            "clean_duster_depths": False,
            "enable_variable_depth_type_augs": False,
            "enable_variable_num_views_augs": False,
        }
        kwargs.update(overrides)
        return kwargs

    def test_split_names_map_to_exact_prepared_roots(self):
        expected = {
            "training": "train",
            "validation": "validation",
            "test": "test",
        }
        for name, directory in expected.items():
            with self.subTest(name=name), mock.patch.object(
                loader.KubricMultiViewDataset,
                "from_name",
                return_value=self._base_kwargs(),
            ):
                kwargs = loader.PointOdysseyMultiViewDataset.from_name(
                    f"pointodyssey-multiview-{name}",
                    "/datasets",
                    training_args=SimpleNamespace(modes=SimpleNamespace(debug=False)),
                    fabric=object(),
                    just_return_kwargs=True,
                )
            self.assertEqual(
                kwargs["data_root"],
                str(Path("/datasets") / "PointOdyssey_MVTracker" / directory),
            )
            self.assertEqual(kwargs["num_views"], 4)
            self.assertFalse(kwargs["enable_variable_depth_type_augs"])
            self.assertFalse(kwargs["enable_variable_num_views_augs"])
            self.assertFalse(kwargs["use_duster_depths"])

    def test_rejects_unsupported_variable_depth_and_view_configuration(self):
        cases = (
            ("enable_variable_depth_type_augs", "variable-depth augmentation"),
            ("enable_variable_num_views_augs", "variable-view augmentation"),
        )
        for option, message in cases:
            with self.subTest(option=option), mock.patch.object(
                loader.KubricMultiViewDataset,
                "from_name",
                return_value=self._base_kwargs(**{option: True}),
            ):
                with self.assertRaisesRegex(ValueError, message):
                    loader.PointOdysseyMultiViewDataset.from_name(
                        "pointodyssey-multiview-training",
                        "/datasets",
                        training_args=SimpleNamespace(modes=SimpleNamespace(debug=False)),
                        fabric=object(),
                        just_return_kwargs=True,
                    )

    def test_training_debug_uses_validation_split(self):
        with mock.patch.object(
            loader.KubricMultiViewDataset,
            "from_name",
            return_value=self._base_kwargs(),
        ):
            kwargs = loader.PointOdysseyMultiViewDataset.from_name(
                "pointodyssey-multiview-training",
                "/datasets",
                training_args=SimpleNamespace(modes=SimpleNamespace(debug=True)),
                fabric=object(),
                just_return_kwargs=True,
            )

        self.assertEqual(
            kwargs["data_root"],
            str(Path("/datasets") / "PointOdyssey_MVTracker" / "validation"),
        )


if __name__ == "__main__":
    unittest.main()
