import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts import pointodyssey_preprocessing as preprocessing


class SplitPlanTests(unittest.TestCase):
    def test_approved_split_ids_chunks_and_counts(self):
        specs = preprocessing.build_scene_specs()

        expected = {
            "train": (78, 8846),
            "validation": (3, 301),
            "test": (3, 301),
        }
        for split, (scene_count, frame_count) in expected.items():
            split_specs = [spec for spec in specs if spec.split == split]
            self.assertEqual([spec.scene_id for spec in split_specs], [f"{i:06d}" for i in range(scene_count)])
            self.assertEqual(sum(spec.frame_count for spec in split_specs), frame_count)

        expected_held_out = {
            "validation": [
                ("000000", "raw", "indoor_01_classroom_manual"),
                ("000001", "raw", "outdoor_02_hidden_alley_manual"),
                ("000002", "short", "indoor_01_classroom"),
            ],
            "test": [
                ("000000", "raw", "indoor_04_modern_loft_manual"),
                ("000001", "raw", "outdoor_06_city_scene_manual"),
                ("000002", "short", "indoor_04_modern_loft"),
            ],
        }
        for split, expected_sources in expected_held_out.items():
            actual_sources = [
                (spec.scene_id, spec.layout, spec.source_sequence)
                for spec in specs
                if spec.split == split
            ]
            self.assertEqual(actual_sources, expected_sources)

        long_groups = {}
        for spec in specs:
            if spec.layout == "long":
                long_groups.setdefault(spec.source_sequence, []).append(spec)
        self.assertEqual(set(long_groups), {
            "candidate_empty_office",
            "candidate_parking",
            "candidate_warehouse",
            "og_parking_lot",
        })
        expected_ranges = [(start, min(start + 120, 2000)) for start in range(0, 2000, 120)]
        for group in long_groups.values():
            self.assertEqual(
                [(spec.source_frame_start, spec.source_frame_end) for spec in group],
                expected_ranges,
            )
            self.assertEqual([spec.frame_count for spec in group], [120] * 16 + [80])
        self.assertEqual(
            {sequence: group[0].scene_id for sequence, group in long_groups.items()},
            {
                "candidate_empty_office": "000010",
                "candidate_parking": "000027",
                "candidate_warehouse": "000044",
                "og_parking_lot": "000061",
            },
        )

    def test_intrinsics_use_approved_anisotropic_scaling(self):
        intrinsics = np.asarray([1920.0, 1080.0, 960.0, 540.0], dtype=np.float32)
        actual = preprocessing.scale_intrinsics(intrinsics)
        expected = np.asarray(
            [[512.0, 0.0, 256.0], [0.0, 384.0, 192.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.float32)


class ValidationPolicyTests(unittest.TestCase):
    @staticmethod
    def _spec() -> preprocessing.SceneSpec:
        return preprocessing.SceneSpec(
            split="train",
            scene_id="000000",
            layout="raw",
            source_sequence="synthetic",
            environment_family="synthetic",
            source_frame_start=0,
            source_frame_end=1,
            source_frame_count=1,
            source_fps=30,
        )

    def test_query_failure_is_source_validated_and_assigned_to_owning_chunk(self):
        first = preprocessing.SceneSpec(
            "train", "000000", "long", "synthetic", "synthetic", 0, 1, 2, 15
        )
        second = preprocessing.SceneSpec(
            "train", "000001", "long", "synthetic", "synthetic", 1, 2, 2, 15
        )
        tracks = np.zeros((2, preprocessing.POINT_COUNT, 3), dtype=np.float32)
        tracks[..., 2] = 1.0
        queries = np.zeros((preprocessing.POINT_COUNT, 4), dtype=np.float32)
        queries[:, :2] = [50.0, 40.0]
        queries[:, 2] = 0.0
        queries[:, 3] = 0.0
        queries[0] = [51.0, 40.0, 1.0, 0.0]
        intrinsics = [np.asarray([100.0, 100.0, 50.0, 40.0], dtype=np.float32) for _ in range(4)]
        extrinsics = [np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0) for _ in range(4)]
        visibility = [np.ones((2, preprocessing.POINT_COUNT), dtype=bool) for _ in range(4)]
        recorder = preprocessing.ValidationRecorder(ignore_failures=True)

        preprocessing._validate_source_queries(
            [first, second], tracks, queries, intrinsics, extrinsics, visibility, recorder
        )

        self.assertEqual(recorder.for_scene(first), [])
        second_checks = {failure["check"] for failure in recorder.for_scene(second)}
        self.assertEqual(second_checks, {"query_reprojection_source", "query_reprojection_resized"})
        np.testing.assert_array_equal(queries[0], [51.0, 40.0, 1.0, 0.0])

    def test_nonfinite_failure_is_strict_json(self):
        recorder = preprocessing.ValidationRecorder(ignore_failures=True)
        recorder.record(self._spec(), "synthetic", float("nan"), "finite")
        report = preprocessing._report([self._spec()], recorder)
        scene_metadata = preprocessing._scene_metadata(
            self._spec(), recorder, {"views": {}}
        )

        encoded = json.dumps(report, allow_nan=False)
        self.assertIn('"nan"', encoded)
        self.assertTrue(report["failures"][0]["ignored"])
        self.assertEqual(scene_metadata["validation"]["failure_count"], 1)
        self.assertEqual(scene_metadata["validation"]["failures"][0]["check"], "synthetic")

    def test_strict_semantic_failure_does_not_publish(self):
        self._run_mock_preprocess(ignore=False, expected_exception=preprocessing.SemanticValidationError)

    def test_ignore_semantic_failure_publishes_report_without_repair(self):
        output_root = self._run_mock_preprocess(ignore=True)
        report = json.loads((output_root / "validation_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "completed_with_ignored_validation_failures")
        self.assertEqual(report["totals"]["semantic_validation_failures"], 1)
        self.assertTrue(report["failures"][0]["ignored"])

    def test_structural_failure_remains_fatal_in_ignore_mode(self):
        self._run_mock_preprocess(ignore=True, structural_failure=True, expected_exception=ValueError)

    def _run_mock_preprocess(
        self,
        *,
        ignore: bool,
        structural_failure: bool = False,
        expected_exception=None,
    ) -> Path:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        output_root = root / "prepared"
        spec = self._spec()

        def convert(_source_root, _build_root, _source_specs, recorder, scene_stats):
            if structural_failure:
                raise ValueError("synthetic structural failure")
            recorder.record(spec, "synthetic_semantic_failure", -1, "non-negative")
            scene_stats[(spec.split, spec.scene_id)] = {"views": {}}

        patches = (
            mock.patch.object(preprocessing, "build_scene_specs", return_value=[spec]),
            mock.patch.object(preprocessing, "preflight"),
            mock.patch.object(preprocessing, "_prepare_scene_roots"),
            mock.patch.object(preprocessing, "_group_by_source", return_value=[[spec]]),
            mock.patch.object(preprocessing, "_convert_source_group", side_effect=convert),
            mock.patch.object(preprocessing, "_write_scene_metadata"),
            mock.patch.object(preprocessing, "_validate_output_tree"),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            if expected_exception is None:
                preprocessing.preprocess(root / "source", output_root, ignore_validation_failures=ignore)
                self.assertTrue(output_root.is_dir())
            else:
                with self.assertRaises(expected_exception):
                    preprocessing.preprocess(root / "source", output_root, ignore_validation_failures=ignore)
                self.assertFalse(output_root.exists())
        return output_root


if __name__ == "__main__":
    unittest.main()
