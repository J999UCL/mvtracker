import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
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


class DepthTrackConsistencyMaskTests(unittest.TestCase):
    @staticmethod
    def _single_point_mask(
        depth_frame: np.ndarray,
        projected_xy: tuple[float, float],
        camera_z: float,
        *,
        candidate: bool = True,
    ) -> np.ndarray:
        return preprocessing._depth_track_consistency_mask(
            np.asarray(depth_frame, dtype=np.float32),
            np.asarray([projected_xy], dtype=np.float32),
            np.asarray([camera_z], dtype=np.float32),
            np.asarray([candidate], dtype=bool),
        )

    def test_tolerance_constant_is_exactly_five_centimetres(self):
        self.assertEqual(preprocessing.DEPTH_TRACK_TOLERANCE_METRES, 0.05)

    def test_inclusive_threshold_and_both_residual_sides(self):
        exact_depth = np.float32(0.1)
        exact_camera_z = np.float32(0.05)
        self.assertEqual(exact_depth - exact_camera_z, np.float32(0.05))
        cases = (
            (exact_depth, exact_camera_z, True),
            (2.049, 2.0, True),
            (3.051, 3.0, False),
            (1.0, 2.0, False),
        )
        for sampled_depth, camera_z, expected in cases:
            with self.subTest(sampled_depth=sampled_depth, camera_z=camera_z):
                depth = np.full(
                    (preprocessing.OUTPUT_HEIGHT, preprocessing.OUTPUT_WIDTH),
                    100.0,
                    dtype=np.float32,
                )
                depth[1, 1] = sampled_depth
                actual = self._single_point_mask(depth, (1.0, 1.0), camera_z)
                np.testing.assert_array_equal(actual, np.asarray([expected], dtype=bool))
                self.assertEqual(actual.dtype, np.dtype(np.bool_))

    def test_matching_alternative_in_three_by_three_neighborhood_is_accepted(self):
        depth = np.full(
            (preprocessing.OUTPUT_HEIGHT, preprocessing.OUTPUT_WIDTH),
            10.0,
            dtype=np.float32,
        )
        depth[2, 2] = 5.0
        depth[2, 3] = 2.04

        actual = self._single_point_mask(depth, (2.0, 2.0), 2.0)

        np.testing.assert_array_equal(actual, np.asarray([True], dtype=bool))

    def test_matching_depth_at_radius_two_is_rejected(self):
        depth = np.full(
            (preprocessing.OUTPUT_HEIGHT, preprocessing.OUTPUT_WIDTH),
            10.0,
            dtype=np.float32,
        )
        depth[10, 12] = 1.0

        actual = self._single_point_mask(depth, (10.0, 10.0), 1.0)

        np.testing.assert_array_equal(actual, np.asarray([False], dtype=bool))

    def test_invalid_depths_are_ignored_and_no_valid_neighbor_is_rejected(self):
        depth = np.full(
            (preprocessing.OUTPUT_HEIGHT, preprocessing.OUTPUT_WIDTH),
            np.nan,
            dtype=np.float32,
        )
        depth[:3, :3] = np.asarray(
            [
                [0.0, -0.01, np.nan],
                [np.inf, -np.inf, 0.0],
                [np.nan, -0.02, np.inf],
            ],
            dtype=np.float32,
        )

        actual = self._single_point_mask(depth, (1.0, 1.0), 0.01)

        np.testing.assert_array_equal(actual, np.asarray([False], dtype=bool))

    def test_invalid_center_is_ignored_when_valid_neighbor_matches(self):
        depth = np.full(
            (preprocessing.OUTPUT_HEIGHT, preprocessing.OUTPUT_WIDTH),
            np.nan,
            dtype=np.float32,
        )
        depth[1, 1] = 0.0
        depth[0, 1] = 1.02

        actual = self._single_point_mask(depth, (1.0, 1.0), 1.0)

        np.testing.assert_array_equal(actual, np.asarray([True], dtype=bool))

    def test_non_candidate_remains_false_even_with_exact_depth(self):
        depth = np.ones(
            (preprocessing.OUTPUT_HEIGHT, preprocessing.OUTPUT_WIDTH),
            dtype=np.float32,
        )

        actual = self._single_point_mask(depth, (1.0, 1.0), 1.0, candidate=False)

        np.testing.assert_array_equal(actual, np.asarray([False], dtype=bool))

    def test_mixed_candidate_mask_preserves_original_point_mapping(self):
        depth = np.full(
            (preprocessing.OUTPUT_HEIGHT, preprocessing.OUTPUT_WIDTH),
            10.0,
            dtype=np.float32,
        )
        projected_xy = np.asarray(
            [[10.0, 10.0], [20.0, 10.0], [30.0, 10.0], [40.0, 10.0]],
            dtype=np.float32,
        )
        camera_z = np.ones((4,), dtype=np.float32)
        candidate_mask = np.asarray([False, True, False, True], dtype=bool)
        depth[10, 10] = 1.0
        depth[10, 20] = 1.0
        depth[10, 30] = 1.0

        actual = preprocessing._depth_track_consistency_mask(
            depth,
            projected_xy,
            camera_z,
            candidate_mask,
        )

        np.testing.assert_array_equal(
            actual,
            np.asarray([False, True, False, False], dtype=bool),
        )
        self.assertEqual(actual.dtype, np.dtype(np.bool_))

    def test_plural_helper_distinguishes_valid_depth_from_consistency(self):
        depth = np.full(
            (preprocessing.OUTPUT_HEIGHT, preprocessing.OUTPUT_WIDTH),
            np.nan,
            dtype=np.float32,
        )
        projected_xy = np.asarray(
            [[10.0, 10.0], [20.0, 10.0], [30.0, 10.0], [40.0, 10.0]],
            dtype=np.float32,
        )
        camera_z = np.ones((4,), dtype=np.float32)
        candidate_mask = np.asarray([True, True, True, False], dtype=bool)
        depth[10, 10] = 10.0
        depth[10, 20] = 0.0
        depth[10, 21] = -1.0
        depth[10, 30] = 1.0
        depth[10, 40] = 1.0

        consistent, any_valid_depth = preprocessing._depth_track_consistency_masks(
            depth,
            projected_xy,
            camera_z,
            candidate_mask,
        )

        np.testing.assert_array_equal(
            consistent,
            np.asarray([False, False, True, False], dtype=bool),
        )
        np.testing.assert_array_equal(
            any_valid_depth,
            np.asarray([True, False, True, False], dtype=bool),
        )
        self.assertEqual(consistent.dtype, np.dtype(np.bool_))
        self.assertEqual(any_valid_depth.dtype, np.dtype(np.bool_))

    def test_border_neighborhood_skips_out_of_bounds_without_index_wrapping(self):
        depth = np.full(
            (preprocessing.OUTPUT_HEIGHT, preprocessing.OUTPUT_WIDTH),
            10.0,
            dtype=np.float32,
        )
        depth[1, 1] = 1.0
        accepted = self._single_point_mask(depth, (0.0, 0.0), 1.0)
        np.testing.assert_array_equal(accepted, np.asarray([True], dtype=bool))

        depth[1, 1] = 10.0
        depth[-1, -1] = 1.0
        rejected = self._single_point_mask(depth, (0.0, 0.0), 1.0)
        np.testing.assert_array_equal(rejected, np.asarray([False], dtype=bool))

    def test_nearest_pixel_uses_floor_coordinate_plus_half(self):
        depth = np.full(
            (preprocessing.OUTPUT_HEIGHT, preprocessing.OUTPUT_WIDTH),
            10.0,
            dtype=np.float32,
        )
        depth[2, 4] = 1.0
        projected_xy = np.asarray([[2.49, 2.0], [2.5, 2.0]], dtype=np.float32)
        camera_z = np.ones((2,), dtype=np.float32)
        candidate_mask = np.ones((2,), dtype=bool)

        actual = preprocessing._depth_track_consistency_mask(
            depth,
            projected_xy,
            camera_z,
            candidate_mask,
        )

        np.testing.assert_array_equal(actual, np.asarray([False, True], dtype=bool))


class ConverterDepthConsistencyIntegrationTests(unittest.TestCase):
    def test_converter_uses_persisted_resized_depth_and_reconciles_visibility(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        source_root = root / "source"
        build_root = root / "build"
        build_root.mkdir()
        spec = preprocessing.SceneSpec(
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
        source_scene = (
            source_root
            / preprocessing.SOURCE_SUBROOTS[spec.layout]
            / spec.source_sequence
        )
        source_view = source_scene / "0"
        source_view.mkdir(parents=True)

        tracks = np.asarray(
            [[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [10.0, 0.0, 1.0]]],
            dtype=np.float32,
        )
        np.save(source_scene / "tracks_xyz.npy", tracks)
        np.save(source_scene / "queries_xytv.npy", np.zeros((3, 4), dtype=np.float32))
        np.save(
            source_view / "intrinsics.npy",
            np.asarray([2.0, 2.0, 2.0, 2.0], dtype=np.float32),
        )
        np.save(
            source_view / "extrinsics_w2c.npy",
            np.eye(4, dtype=np.float32)[None],
        )
        np.save(
            source_view / "visibility.npy",
            np.ones((1, 3), dtype=bool),
        )
        source_depth = np.asarray(
            [
                [
                    [1.00, 9.0, 1.01, 9.0],
                    [9.0, 9.0, 9.0, 9.0],
                    [1.02, 9.0, 1.03, 9.0],
                    [9.0, 9.0, 9.0, 9.0],
                ]
            ],
            dtype=np.float32,
        )
        np.save(source_view / "depth.npy", source_depth)
        source_image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
        encoded_ok, encoded_image = cv2.imencode(".jpg", source_image)
        self.assertTrue(encoded_ok)
        jpeg_objects = np.empty((1,), dtype=object)
        jpeg_objects[0] = encoded_image.reshape(-1)
        np.save(source_view / "images_jpeg_bytes.npy", jpeg_objects)

        original_masks = preprocessing._depth_track_consistency_masks
        captured_calls = []

        def spy_masks(depth_frame, projected_xy, camera_z, candidate_mask):
            captured_calls.append(
                tuple(
                    np.array(value, copy=True)
                    for value in (depth_frame, projected_xy, camera_z, candidate_mask)
                )
            )
            return original_masks(depth_frame, projected_xy, camera_z, candidate_mask)

        recorder = preprocessing.ValidationRecorder(ignore_failures=False)
        scene_stats = {}
        with mock.patch.multiple(
            preprocessing,
            POINT_COUNT=3,
            VIEW_IDS=(0,),
            SOURCE_HEIGHT=4,
            SOURCE_WIDTH=4,
            OUTPUT_HEIGHT=2,
            OUTPUT_WIDTH=2,
            SCALE_X=0.5,
            SCALE_Y=0.5,
        ):
            with mock.patch.object(
                preprocessing,
                "_validate_source_queries",
            ), mock.patch.object(
                preprocessing,
                "_depth_track_consistency_masks",
                side_effect=spy_masks,
            ) as masks_spy:
                preprocessing._prepare_scene_roots(build_root, [spec])
                preprocessing._convert_source_group(
                    source_root,
                    build_root,
                    [spec],
                    recorder,
                    scene_stats,
                )

        output_view = build_root / "train" / "000000" / "view_0"
        persisted_depth = np.load(output_view / "depth.npy")
        expected_resized_depth = cv2.resize(
            source_depth[0],
            (2, 2),
            interpolation=cv2.INTER_NEAREST,
        )
        np.testing.assert_array_equal(persisted_depth[0], expected_resized_depth)
        self.assertEqual(masks_spy.call_count, 1)
        self.assertEqual(len(captured_calls), 1)
        captured_depth, captured_xy, captured_camera_z, captured_candidates = (
            captured_calls[0]
        )
        np.testing.assert_array_equal(captured_depth, persisted_depth[0])
        np.testing.assert_array_equal(captured_camera_z, [1.0, 2.0, 1.0])
        np.testing.assert_array_equal(
            captured_candidates,
            np.asarray([True, True, False], dtype=bool),
        )
        self.assertEqual(captured_xy.shape, (3, 2))

        persisted_visibility = np.load(output_view / "visibility.npy")
        np.testing.assert_array_equal(
            persisted_visibility,
            np.asarray([[True, False, False]], dtype=bool),
        )
        expected_stats = {
            "visibility_true_before_gating": 3,
            "visibility_true_after_geometric_gating": 2,
            "visibility_removed_by_geometric_gating": 1,
            "depth_consistency_candidate_count": 2,
            "depth_consistency_tolerance_metres": 0.05,
            "visibility_rejected_no_valid_depth": 0,
            "visibility_rejected_residual_over_tolerance": 1,
            "visibility_true_after_depth_consistency_gating": 1,
            "visibility_removed_by_depth_consistency_gating": 1,
            "visibility_true_after_gating": 1,
            "visibility_removed_by_gating": 2,
        }
        self.assertEqual(scene_stats[("train", "000000")]["views"]["0"], expected_stats)
        self.assertEqual(recorder.failures, [])

        output_jpeg = output_view / "rgba_00000.jpg"
        self.assertTrue(output_jpeg.is_file())
        decoded_output = cv2.imread(str(output_jpeg), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded_output)
        self.assertEqual(decoded_output.shape, (2, 2, 3))
        np.testing.assert_array_equal(
            np.load(output_view.parent / "tracks_3d.npy"),
            tracks,
        )


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
