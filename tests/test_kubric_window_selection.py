import ast
import unittest
from pathlib import Path

import numpy as np


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


class VirtualDatasetIndexTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
