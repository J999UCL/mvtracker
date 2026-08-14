import ast
import os
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch


def _load_window_start_helper():
    """Load the helper without importing Kubric's optional vision dependencies."""
    source_path = (
        Path(__file__).resolve().parents[1]
        / "mvtracker"
        / "datasets"
        / "kubric_multiview_dataset.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_legal_contiguous_window_starts"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"np": np}
    exec(compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace)
    return namespace[function.name]


_legal_contiguous_window_starts = _load_window_start_helper()


def _load_top_level_function(function_name, namespace):
    source_path = (
        Path(__file__).resolve().parents[1]
        / "mvtracker"
        / "datasets"
        / "kubric_multiview_dataset.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace)
    return namespace[function_name]


_track_cache_file = _load_top_level_function("_track_cache_file", {"os": os})
_training_virtual_dataset_size = _load_top_level_function(
    "_training_virtual_dataset_size",
    {},
)


def _load_kubric_method(method_name):
    """Load one dataset method without importing optional vision dependencies."""
    source_path = (
        Path(__file__).resolve().parents[1]
        / "mvtracker"
        / "datasets"
        / "kubric_multiview_dataset.py"
    )
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    dataset_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "KubricMultiViewDataset"
    )
    method = next(
        node
        for node in dataset_class.body
        if isinstance(node, ast.FunctionDef)
        and node.name == method_name
    )
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {}
    exec(compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace)
    return namespace[method_name]


_kubric_getitem = _load_kubric_method("__getitem__")
_kubric_getitem_helper = _load_kubric_method("_getitem_helper")
_kubric_getitem_helper.__globals__.update({
    "np": np,
    "os": os,
    "time": time,
    "torch": torch,
    "_legal_contiguous_window_starts": _legal_contiguous_window_starts,
})
_kubric_cache_key = _load_kubric_method("_cache_key")


class LegalContiguousWindowStartsTests(unittest.TestCase):
    def test_includes_final_legal_start(self):
        starts = _legal_contiguous_window_starts(n_frames=30, seq_len=24)

        np.testing.assert_array_equal(starts, np.arange(7, dtype=np.int64))
        self.assertEqual(starts[-1], 30 - 24)

    def test_excludes_every_window_containing_one_bad_frame(self):
        starts = _legal_contiguous_window_starts(
            n_frames=50,
            seq_len=24,
            invalid_frame_indices=[25],
        )

        np.testing.assert_array_equal(starts, np.asarray([0, 1, 26]))

    def test_multiple_bad_frames_use_union_of_overlapping_exclusions(self):
        starts = _legal_contiguous_window_starts(
            n_frames=50,
            seq_len=24,
            invalid_frame_indices=[23, 24],
        )

        np.testing.assert_array_equal(starts, np.asarray([25, 26]))

    def test_bad_frames_at_scene_boundaries_only_remove_intersecting_starts(self):
        starts = _legal_contiguous_window_starts(
            n_frames=30,
            seq_len=24,
            invalid_frame_indices=[0, 29],
        )

        np.testing.assert_array_equal(starts, np.arange(1, 6, dtype=np.int64))

    def test_rejects_non_integer_bad_frame_indices(self):
        for invalid_frame_indices in ([True], [1.5], ["1"]):
            with self.subTest(invalid_frame_indices=invalid_frame_indices):
                with self.assertRaisesRegex(ValueError, "must be integers"):
                    _legal_contiguous_window_starts(
                        n_frames=30,
                        seq_len=24,
                        invalid_frame_indices=invalid_frame_indices,
                    )

    def test_rejects_duplicate_bad_frame_indices(self):
        with self.assertRaisesRegex(ValueError, "must be unique"):
            _legal_contiguous_window_starts(
                n_frames=30,
                seq_len=24,
                invalid_frame_indices=[3, 3],
            )

    def test_rejects_out_of_range_bad_frame_indices(self):
        for invalid_frame_indices in ([-1], [30]):
            with self.subTest(invalid_frame_indices=invalid_frame_indices):
                with self.assertRaisesRegex(ValueError, r"must be in \[0, 30\)"):
                    _legal_contiguous_window_starts(
                        n_frames=30,
                        seq_len=24,
                        invalid_frame_indices=invalid_frame_indices,
                    )

    def test_rejects_invalid_frame_index_container_types(self):
        for invalid_frame_indices in (None, 3):
            with self.subTest(invalid_frame_indices=invalid_frame_indices):
                with self.assertRaises(TypeError):
                    _legal_contiguous_window_starts(
                        n_frames=30,
                        seq_len=24,
                        invalid_frame_indices=invalid_frame_indices,
                    )

    def test_returns_empty_when_no_legal_window_remains(self):
        starts = _legal_contiguous_window_starts(
            n_frames=30,
            seq_len=24,
            invalid_frame_indices=[6],
        )

        self.assertEqual(starts.dtype, np.int64)
        self.assertEqual(starts.size, 0)

    def test_dataset_raises_when_no_legal_window_remains(self):
        class DatasetStub:
            data_root = "/prepared"
            seq_names = ["000000"]
            seq_len = 24
            seed = 7
            add_index_to_seed = True

            @staticmethod
            def getitem_raw_datapoint(_scene_path):
                return {
                    "tracks_3d": torch.zeros((30, 1, 3), dtype=torch.float32),
                    "views": [],
                    "invalid_frame_indices": [6],
                }

        with self.assertRaisesRegex(
            ValueError,
            "No valid 24-frame windows remain.*invalid frame indices: \\[6\\]",
        ):
            _kubric_getitem_helper(DatasetStub(), 0)


class CacheKeyTests(unittest.TestCase):
    def test_cache_key_uses_generic_invalid_frame_contract_version(self):
        class DatasetStub:
            seed = 1
            ratio_dynamic = 0.5
            ratio_very_dynamic = 0.25
            views_to_return = None
            traj_per_sample = 256
            num_views = 4
            seq_len = 24
            sample_vis_1st_frame = False
            cache_version = "v3"

        self.assertTrue(_kubric_cache_key(DatasetStub()).endswith("--v3"))

    def test_cache_key_can_select_released_v1_tracks(self):
        class DatasetStub:
            seed = 72
            ratio_dynamic = 0.5
            ratio_very_dynamic = 0.25
            views_to_return = [0, 1, 2, 3]
            traj_per_sample = 512
            num_views = -1
            seq_len = 24
            sample_vis_1st_frame = False
            cache_version = "v1"

        self.assertTrue(_kubric_cache_key(DatasetStub()).endswith("--v1"))


class VirtualDatasetIndexTests(unittest.TestCase):
    def test_training_virtual_size_accounts_for_serial_microbatches(self):
        self.assertEqual(
            _training_virtual_dataset_size(
                world_size=1,
                num_steps=200_000,
                gradient_accumulation_steps=8,
            ),
            1_608_000,
        )

    def test_training_virtual_size_preserves_upstream_default(self):
        self.assertEqual(
            _training_virtual_dataset_size(
                world_size=2,
                num_steps=100,
                gradient_accumulation_steps=1,
            ),
            2_200,
        )

    def test_training_virtual_size_rejects_zero_accumulation(self):
        with self.assertRaisesRegex(ValueError, "must be at least 1"):
            _training_virtual_dataset_size(
                world_size=1,
                num_steps=100,
                gradient_accumulation_steps=0,
            )

    def test_repeated_scene_keeps_distinct_seed_index(self):
        class DatasetStub:
            real_len = 78

            def __init__(self):
                self.calls = []

            def _getitem_helper(self, scene_index, seed_index=None):
                self.calls.append((scene_index, seed_index))
                return scene_index, True

        dataset = DatasetStub()

        first_sample, first_gotit = _kubric_getitem(dataset, 5)
        repeated_sample, repeated_gotit = _kubric_getitem(dataset, 83)

        self.assertTrue(first_gotit)
        self.assertTrue(repeated_gotit)
        self.assertEqual(first_sample, 5)
        self.assertEqual(repeated_sample, 5)
        self.assertEqual(dataset.calls, [(5, 5), (5, 83)])


class TrackCachePathTests(unittest.TestCase):
    def test_disabled_cache_does_not_create_a_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = _track_cache_file(
                temporary_directory,
                "000000",
                "tracks",
                enabled=False,
            )

            self.assertIsNone(result)
            self.assertFalse((Path(temporary_directory) / "000000" / "cache").exists())

    def test_enabled_cache_creates_the_expected_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            scene_root = Path(temporary_directory) / "000000"
            scene_root.mkdir()

            result = _track_cache_file(
                temporary_directory,
                "000000",
                "tracks",
                enabled=True,
            )

            self.assertEqual(result, str(scene_root / "cache" / "tracks.npz"))
            self.assertTrue((scene_root / "cache").is_dir())


if __name__ == "__main__":
    unittest.main()
