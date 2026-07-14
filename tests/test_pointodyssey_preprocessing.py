import io
import json
import multiprocessing
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from scripts import pointodyssey_preprocessing as preprocessing


def _minimal_v5_scene_stats(
    frame_count,
    views,
    *,
    depth_candidate_counts=None,
    depth_failure_counts=None,
):
    candidates = list(depth_candidate_counts or [0] * frame_count)
    failures = list(depth_failure_counts or [0] * frame_count)
    for view_stats in views.values():
        view_stats.setdefault(
            "depth_consistency_candidate_count_by_frame",
            list(candidates),
        )
        view_stats.setdefault(
            "depth_consistency_failure_count_by_frame",
            list(failures),
        )
        view_stats.setdefault("depth_consistency_candidate_count", sum(candidates))
        view_stats.setdefault("depth_consistency_failure_count", sum(failures))
    depth_invalid = preprocessing._depth_invalid_frame_indices(candidates, failures)
    rgb_invalid = sorted(
        {
            frame
            for view_stats in views.values()
            for frame in view_stats["rgb"]["invalid_frame_indices"]
        }
    )
    invalid = sorted(set(rgb_invalid) | set(depth_invalid))
    total, excluded, legal = preprocessing._window_start_counts(frame_count, invalid)
    per_frame = [
        {
            "frame": frame,
            "candidate_count": candidate_count,
            "failure_count": failure_count,
            "failure_fraction": (
                failure_count / candidate_count if candidate_count else None
            ),
        }
        for frame, (candidate_count, failure_count) in enumerate(
            zip(candidates, failures)
        )
    ]
    return {
        "views": views,
        "depth_track_consistency": {
            "candidate_count": sum(candidates),
            "failure_count": sum(failures),
            "invalid_frame_count": len(depth_invalid),
            "invalid_frame_indices": depth_invalid,
            "per_frame": per_frame,
        },
        "window_exclusion": {
            "window_length": preprocessing.WINDOW_LENGTH,
            "invalid_frame_indices": invalid,
            "reasons": {
                "rgb_decode": rgb_invalid,
                "depth_track_majority_mismatch": depth_invalid,
            },
            "total_start_count": total,
            "excluded_start_count": excluded,
            "legal_start_count": legal,
        },
    }


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


class DepthFrameExclusionTests(unittest.TestCase):
    def test_depth_frame_majority_rule_is_strict_and_ignores_zero_candidates(self):
        self.assertEqual(
            preprocessing._depth_invalid_frame_indices(
                [10, 10, 10, 0, 100],
                [6, 5, 1, 0, 51],
            ),
            [0, 4],
        )

    def test_window_counts_remove_every_start_intersecting_invalid_frames(self):
        self.assertEqual(
            preprocessing._window_start_counts(30, [0, 23, 29]),
            (7, 7, 0),
        )
        self.assertEqual(
            preprocessing._window_start_counts(48, [24]),
            (25, 24, 1),
        )

    def test_frame_majority_is_aggregated_across_views(self):
        spec = preprocessing.SceneSpec(
            "train", "000000", "raw", "synthetic", "synthetic", 0, 48, 48, 30
        )
        view_0_candidates = [0] * 48
        view_0_failures = [0] * 48
        view_1_candidates = [0] * 48
        view_1_failures = [0] * 48
        view_0_candidates[47], view_0_failures[47] = 4, 2
        view_1_candidates[47], view_1_failures[47] = 4, 3
        scene_candidates = [a + b for a, b in zip(view_0_candidates, view_1_candidates)]
        scene_failures = [a + b for a, b in zip(view_0_failures, view_1_failures)]
        views = {
            "0": {
                "rgb": {"invalid_frame_indices": []},
                "depth_consistency_candidate_count_by_frame": view_0_candidates,
                "depth_consistency_failure_count_by_frame": view_0_failures,
                "depth_consistency_candidate_count": 4,
                "depth_consistency_failure_count": 2,
            },
            "1": {
                "rgb": {"invalid_frame_indices": []},
                "depth_consistency_candidate_count_by_frame": view_1_candidates,
                "depth_consistency_failure_count_by_frame": view_1_failures,
                "depth_consistency_candidate_count": 4,
                "depth_consistency_failure_count": 3,
            },
        }
        stats = _minimal_v5_scene_stats(
            48,
            views,
            depth_candidate_counts=scene_candidates,
            depth_failure_counts=scene_failures,
        )
        metadata = preprocessing._scene_metadata(
            spec,
            preprocessing.ValidationRecorder(ignore_failures=False),
            stats,
        )

        with mock.patch.object(preprocessing, "VIEW_IDS", (0, 1)):
            preprocessing._validate_scene_metadata_contract(metadata, spec)
        self.assertEqual(
            metadata["output"]["depth_track_consistency"]["invalid_frame_indices"],
            [47],
        )

    def test_scene_contract_rejects_zero_legal_windows(self):
        spec = preprocessing.SceneSpec(
            "train", "000000", "raw", "synthetic", "synthetic", 0, 24, 24, 30
        )
        views = {"0": {"rgb": {"invalid_frame_indices": [0]}}}
        stats = _minimal_v5_scene_stats(24, views)
        metadata = preprocessing._scene_metadata(
            spec,
            preprocessing.ValidationRecorder(ignore_failures=False),
            stats,
        )
        with self.assertRaisesRegex(ValueError, "no legal 24-frame windows"):
            preprocessing._validate_scene_metadata_contract(metadata, spec)

    def test_scene_contract_enforces_exact_sorted_reason_union(self):
        spec = preprocessing.SceneSpec(
            "train", "000000", "raw", "synthetic", "synthetic", 0, 48, 48, 30
        )
        candidates = [0] * 48
        failures = [0] * 48
        candidates[47] = 4
        failures[47] = 3
        views = {"0": {"rgb": {"invalid_frame_indices": [0]}}}
        stats = _minimal_v5_scene_stats(
            48,
            views,
            depth_candidate_counts=candidates,
            depth_failure_counts=failures,
        )
        metadata = preprocessing._scene_metadata(
            spec,
            preprocessing.ValidationRecorder(ignore_failures=False),
            stats,
        )
        with mock.patch.object(preprocessing, "VIEW_IDS", (0,)):
            preprocessing._validate_scene_metadata_contract(metadata, spec)
            malformed = json.loads(json.dumps(metadata))
            malformed["output"]["window_exclusion"]["invalid_frame_indices"] = [
                0,
                1,
                47,
            ]
            with self.assertRaisesRegex(ValueError, "exact reason union"):
                preprocessing._validate_scene_metadata_contract(malformed, spec)

    def test_intrinsics_use_approved_anisotropic_scaling(self):
        intrinsics = np.asarray([1920.0, 1080.0, 960.0, 540.0], dtype=np.float32)
        actual = preprocessing.scale_intrinsics(intrinsics)
        expected = np.asarray(
            [[512.0, 0.0, 256.0], [0.0, 384.0, 192.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.float32)


class CommandLineTests(unittest.TestCase):
    @staticmethod
    def _base_args() -> list[str]:
        return [
            "--source-root",
            "/tmp/source root",
            "--output-root",
            "/tmp/output root",
        ]

    def test_workers_default_is_four(self):
        args = preprocessing.parse_args(self._base_args())

        self.assertEqual(args.workers, 4)
        self.assertEqual(args.source_root, Path("/tmp/source root"))
        self.assertEqual(args.output_root, Path("/tmp/output root"))

    def test_positive_worker_overrides(self):
        for workers in (1, 2, 7):
            with self.subTest(workers=workers):
                args = preprocessing.parse_args(
                    [*self._base_args(), "--workers", str(workers)]
                )
                self.assertEqual(args.workers, workers)

    def test_nonpositive_and_noninteger_workers_are_rejected(self):
        for workers in ("0", "-1", "not-an-integer"):
            with self.subTest(workers=workers), mock.patch("sys.stderr"):
                with self.assertRaisesRegex(SystemExit, "2"):
                    preprocessing.parse_args(
                        [*self._base_args(), "--workers", workers]
                    )


class ProgressReporterTests(unittest.TestCase):
    @staticmethod
    def _spec() -> preprocessing.SceneSpec:
        return preprocessing.SceneSpec(
            split="train",
            scene_id="000000",
            layout="raw",
            source_sequence="synthetic",
            environment_family="synthetic",
            source_frame_start=0,
            source_frame_end=2,
            source_frame_count=2,
            source_fps=30,
        )

    def test_atomic_progress_sidecar_and_terminal_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "prepared"
            spec = self._spec()
            recorder = preprocessing.ValidationRecorder(ignore_failures=True)
            reporter = preprocessing.ProgressReporter(
                root / "source",
                output_root,
                [spec],
                workers=3,
                update_interval_seconds=0.0,
            )
            reporter.bind_recorder(recorder)
            terminal = io.StringIO()
            with redirect_stdout(terminal):
                reporter.set_stage("converting", "source_conversion")
                reporter.set_active(
                    "rgb",
                    layout="raw",
                    source_sequence="synthetic",
                    split="train",
                    scene_id="000000",
                    view=2,
                )
                recorder.record(spec, "synthetic", -1, "non-negative")
                reporter.record_invalid_rgb_frame(spec, 1)
                reporter.record_invalid_rgb_frame(spec, 1)
                reporter.advance_many(
                    sources=1,
                    scenes=1,
                    frames=2,
                    camera_frames=8,
                    jpegs=8,
                    validated_jpegs=8,
                    output_bytes=4096,
                )
                reporter.completed()

            progress_path = Path(f"{output_root}.progress.json")
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(payload["format"], "pointodyssey_preprocessing_progress")
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["workers"], 3)
            self.assertEqual(payload["active"]["stage"], "completed")
            self.assertEqual(payload["progress"]["frames"], {
                "completed": 2,
                "total": 2,
                "percent": 100.0,
            })
            self.assertEqual(payload["progress"]["output_bytes"]["completed"], 4096)
            self.assertIsNone(payload["progress"]["output_bytes"]["total"])
            self.assertIsNone(payload["progress"]["output_bytes"]["percent"])
            self.assertEqual(payload["diagnostics"], {
                "semantic_validation_failures": 1,
                "invalid_rgb_frames": 1,
            })
            self.assertIn("source_conversion", payload["timing"]["stages_seconds"])
            self.assertEqual(payload["timing"]["current_stage_elapsed_seconds"], 0.0)
            self.assertIsNone(payload["error"])
            self.assertIn("POINTODYSSEY_PROGRESS status=completed", terminal.getvalue())
            self.assertFalse(list(root.glob(".*progress.json.tmp-*")))

    def test_failure_state_keeps_exception_details(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_root = root / "prepared"
            reporter = preprocessing.ProgressReporter(
                root / "source",
                output_root,
                [self._spec()],
                workers=1,
            )
            with redirect_stderr(io.StringIO()):
                reporter.failed(ValueError("synthetic failure"))

            payload = json.loads(
                Path(f"{output_root}.progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["active"]["stage"], "failed")
            self.assertEqual(payload["error"], {
                "type": "ValueError",
                "message": "synthetic failure",
            })

    def test_active_phase_does_not_finish_top_level_stage_timing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reporter = preprocessing.ProgressReporter(
                root / "source",
                root / "prepared",
                [self._spec()],
                workers=1,
                update_interval_seconds=0.0,
            )
            with redirect_stdout(io.StringIO()):
                reporter.set_stage("converting", "source_conversion")
                reporter.set_active("tracks")
                tracks_snapshot = reporter.snapshot()
                reporter.set_active("depth")
                depth_snapshot = reporter.snapshot()

            self.assertEqual(tracks_snapshot["active"]["stage"], "source_conversion")
            self.assertEqual(tracks_snapshot["active"]["phase"], "tracks")
            self.assertEqual(depth_snapshot["active"]["stage"], "source_conversion")
            self.assertEqual(depth_snapshot["active"]["phase"], "depth")
            self.assertNotIn(
                "source_conversion",
                depth_snapshot["timing"]["stages_seconds"],
            )

            with redirect_stdout(io.StringIO()):
                reporter.set_stage("validating_output", "output_validation")
            transitioned = reporter.snapshot()
            self.assertIn("source_conversion", transitioned["timing"]["stages_seconds"])
            self.assertEqual(transitioned["active"]["stage"], "output_validation")
            self.assertIsNone(transitioned["active"]["phase"])


class DepthTrackConsistencyMaskTests(unittest.TestCase):
    @staticmethod
    def _single_point_mask(
        depth_frame: np.ndarray,
        projected_xy: tuple[float, float],
        camera_z: float,
        *,
        candidate: bool = True,
    ) -> np.ndarray:
        consistent, _any_valid = preprocessing._depth_track_consistency_masks(
            np.asarray(depth_frame, dtype=np.float32),
            np.asarray([projected_xy], dtype=np.float32),
            np.asarray([camera_z], dtype=np.float32),
            np.asarray([candidate], dtype=bool),
        )
        return consistent

    @staticmethod
    def _scalar_masks(
        depth_frame: np.ndarray,
        projected_xy: np.ndarray,
        camera_z: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        consistent = np.zeros(candidate_mask.shape, dtype=bool)
        any_valid = np.zeros(candidate_mask.shape, dtype=bool)
        height, width = depth_frame.shape
        for index in np.ndindex(candidate_mask.shape):
            xy = projected_xy[index]
            z = camera_z[index]
            if not (
                candidate_mask[index]
                and np.isfinite(xy).all()
                and np.isfinite(z)
                and z > 0.0
                and -0.5 <= xy[0] < width - 0.5
                and -0.5 <= xy[1] < height - 0.5
            ):
                continue
            center_x = int(np.floor(xy[0] + 0.5))
            center_y = int(np.floor(xy[1] + 0.5))
            for offset_y in (-1, 0, 1):
                sample_y = center_y + offset_y
                for offset_x in (-1, 0, 1):
                    sample_x = center_x + offset_x
                    if not (0 <= sample_x < width and 0 <= sample_y < height):
                        continue
                    sampled_depth = depth_frame[sample_y, sample_x]
                    if not np.isfinite(sampled_depth) or sampled_depth <= 0.0:
                        continue
                    any_valid[index] = True
                    if abs(float(sampled_depth) - float(z)) <= 0.05:
                        consistent[index] = True
        return consistent, any_valid

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

        actual, _any_valid = preprocessing._depth_track_consistency_masks(
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

        actual, _any_valid = preprocessing._depth_track_consistency_masks(
            depth,
            projected_xy,
            camera_z,
            candidate_mask,
        )

        np.testing.assert_array_equal(actual, np.asarray([False, True], dtype=bool))

    def test_vectorized_masks_match_independent_scalar_oracle(self):
        depth = np.asarray(
            [
                [1.0, np.nan, 10.0, 10.0, 10.0, 10.0],
                [0.0, -1.0, 10.0, 10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0, 10.0, 2.05, 10.0],
                [10.0, 10.0, np.inf, 0.0, -1.0, 7.0],
                [10.0, 10.0, np.nan, 0.0, -1.0, 7.0],
            ],
            dtype=np.float32,
        )
        projected_xy = np.asarray(
            [
                [-0.49, -0.49],
                [5.49, 4.49],
                [2.5, 2.0],
                [2.49, 2.0],
                [0.0, 0.0],
                [np.nan, 1.0],
                [1.0, np.inf],
                [1.0, 1.0],
                [1.0, 1.0],
                [6.0, 2.0],
            ],
            dtype=np.float32,
        )
        camera_z = np.asarray(
            [1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 0.0, np.nan, 1.0],
            dtype=np.float32,
        )
        candidates = np.asarray(
            [True, True, True, True, False, True, True, True, True, True],
            dtype=bool,
        )
        original_inputs = tuple(
            np.array(value, copy=True)
            for value in (depth, projected_xy, camera_z, candidates)
        )

        with mock.patch.multiple(preprocessing, OUTPUT_HEIGHT=5, OUTPUT_WIDTH=6):
            actual = preprocessing._depth_track_consistency_masks(
                depth,
                projected_xy,
                camera_z,
                candidates,
            )
        expected = self._scalar_masks(depth, projected_xy, camera_z, candidates)

        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])
        self.assertEqual(actual[0].dtype, np.dtype(np.bool_))
        self.assertEqual(actual[1].dtype, np.dtype(np.bool_))
        self.assertTrue(np.all(actual[0] <= actual[1]))
        self.assertTrue(np.all(actual[1] <= candidates))
        for value, original in zip(
            (depth, projected_xy, camera_z, candidates),
            original_inputs,
        ):
            np.testing.assert_array_equal(value, original)


class RGBWorkerTests(unittest.TestCase):
    @staticmethod
    def _job(root: Path, index: int) -> preprocessing.RGBFrameJob:
        image = np.zeros((6, 8, 3), dtype=np.uint8)
        image[..., 0] = index * 31
        image[..., 1] = np.arange(8, dtype=np.uint8)[None, :] * 17
        image[..., 2] = np.arange(6, dtype=np.uint8)[:, None] * 23
        encoded_ok, encoded = cv2.imencode(".jpg", image)
        if not encoded_ok:
            raise AssertionError("failed to create synthetic source JPEG")
        return preprocessing.RGBFrameJob(
            layout="raw",
            sequence="synthetic",
            view=0,
            source_frame=index,
            split="train",
            scene_id="000000",
            local_frame=index,
            output_path=str(root / f"rgba_{index:05d}.jpg"),
            encoded_jpeg=encoded.tobytes(),
            source_height=6,
            source_width=8,
            output_height=3,
            output_width=4,
            jpeg_quality=95,
        )

    def test_serial_rgb_worker_and_persisted_validator(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        job = self._job(root, 0)

        (result,) = preprocessing._transcode_rgb_batch((job,))

        self.assertEqual(result.output_path, job.output_path)
        self.assertEqual(result.source_frame, job.source_frame)
        self.assertEqual(result.local_frame, job.local_frame)
        self.assertIsNone(result.source_decode_error)
        decoded = cv2.imdecode(
            np.frombuffer(result.encoded_jpeg, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertEqual(decoded.shape, (3, 4, 3))
        Path(result.output_path).write_bytes(result.encoded_jpeg)
        validation_job = preprocessing.JPEGValidationJob(
            split="train",
            scene_id="000000",
            view=0,
            frame=0,
            path=result.output_path,
            expected_height=3,
            expected_width=4,
        )
        self.assertEqual(preprocessing._validate_jpeg_batch((validation_job,)), 1)

        Path(result.output_path).write_bytes(b"not-a-jpeg")
        with self.assertRaisesRegex(RuntimeError, "frame=0"):
            preprocessing._validate_jpeg_batch((validation_job,))

    def test_invalid_jpeg_bytes_produce_decodable_black_placeholder(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        job = replace(self._job(root, 2), encoded_jpeg=b"not-a-jpeg")

        (result,) = preprocessing._transcode_rgb_batch((job,))

        self.assertEqual(result.source_frame, 2)
        self.assertEqual(result.local_frame, 2)
        self.assertEqual(
            result.source_decode_error,
            "cv2.imdecode returned no image",
        )
        decoded = cv2.imdecode(
            np.frombuffer(result.encoded_jpeg, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertIsNotNone(decoded)
        self.assertEqual(decoded.shape, (3, 4, 3))
        self.assertEqual(int(np.count_nonzero(decoded)), 0)

    def test_opencv_decode_exception_produces_placeholder(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        job = self._job(root, 1)

        with mock.patch.object(
            preprocessing.cv2,
            "imdecode",
            side_effect=cv2.error("synthetic decode failure"),
        ):
            (result,) = preprocessing._transcode_rgb_batch((job,))

        self.assertIsNotNone(result.source_decode_error)
        self.assertTrue(
            result.source_decode_error.startswith("cv2.imdecode failed:"),
            result.source_decode_error,
        )
        decoded = cv2.imdecode(
            np.frombuffer(result.encoded_jpeg, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertEqual(decoded.shape, (3, 4, 3))
        self.assertEqual(int(np.count_nonzero(decoded)), 0)

    def test_decoded_wrong_resolution_remains_fatal(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        wrong_size = np.zeros((5, 8, 3), dtype=np.uint8)
        encoded_ok, encoded = cv2.imencode(".jpg", wrong_size)
        self.assertTrue(encoded_ok)
        job = replace(
            self._job(root, 0),
            encoded_jpeg=encoded.tobytes(),
        )

        with self.assertRaisesRegex(RuntimeError, "RGB transcode failed") as raised:
            preprocessing._transcode_rgb_batch((job,))

        self.assertIsInstance(raised.exception.__cause__, ValueError)
        self.assertIn("source JPEG decodes to", str(raised.exception.__cause__))

    def test_spawn_processpool_matches_serial_jpeg_bytes_and_validation(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        jobs = tuple(self._job(root, index) for index in range(4))
        batches = tuple((job,) for job in jobs)
        serial = tuple(
            result
            for batch in batches
            for result in preprocessing._transcode_rgb_batch(batch)
        )

        with ProcessPoolExecutor(
            max_workers=2,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=preprocessing._initialize_process_worker,
        ) as process_pool:
            parallel = tuple(
                result
                for results in preprocessing._run_batches(
                    batches,
                    preprocessing._transcode_rgb_batch,
                    process_pool=process_pool,
                    process_workers=2,
                    stage="synthetic RGB transcode",
                )
                for result in results
            )
            self.assertEqual(
                [result.output_path for result in parallel],
                [result.output_path for result in serial],
            )
            self.assertEqual(
                [result.encoded_jpeg for result in parallel],
                [result.encoded_jpeg for result in serial],
            )
            validation_batches = []
            for index, result in enumerate(parallel):
                Path(result.output_path).write_bytes(result.encoded_jpeg)
                validation_batches.append(
                    (
                        preprocessing.JPEGValidationJob(
                            split="train",
                            scene_id="000000",
                            view=0,
                            frame=index,
                            path=result.output_path,
                            expected_height=3,
                            expected_width=4,
                        ),
                    )
                )
            validated_counts = list(
                preprocessing._run_batches(
                    validation_batches,
                    preprocessing._validate_jpeg_batch,
                    process_pool=process_pool,
                    process_workers=2,
                    stage="synthetic JPEG validation",
                )
            )

        self.assertEqual(validated_counts, [1, 1, 1, 1])

    def test_ordered_scheduler_bounds_submissions_and_preserves_order(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        batches = tuple((self._job(root, index),) for index in range(5))

        class ImmediateFuture:
            def __init__(self, executor, result):
                self.executor = executor
                self.value = result

            def result(self):
                self.executor.in_flight -= 1
                return self.value

            def cancel(self):
                self.executor.in_flight -= 1
                return True

        class RecordingExecutor:
            def __init__(self):
                self.in_flight = 0
                self.maximum_in_flight = 0

            def submit(self, worker, batch):
                self.in_flight += 1
                self.maximum_in_flight = max(
                    self.maximum_in_flight,
                    self.in_flight,
                )
                return ImmediateFuture(self, worker(batch))

        executor = RecordingExecutor()
        results = list(
            preprocessing._ordered_bounded_batches(
                executor,
                preprocessing._transcode_rgb_batch,
                batches,
                max_in_flight=2,
                stage="synthetic bounded scheduling",
            )
        )

        self.assertEqual(executor.maximum_in_flight, 2)
        self.assertEqual(executor.in_flight, 0)
        self.assertEqual(
            [batch[0].output_path for batch in results],
            [batch[0].output_path for batch in batches],
        )


class ConverterDepthConsistencyIntegrationTests(unittest.TestCase):
    def test_scene_metadata_unions_invalid_rgb_frames_across_views(self):
        spec = preprocessing.SceneSpec(
            split="train",
            scene_id="000000",
            layout="raw",
            source_sequence="synthetic",
            environment_family="synthetic",
            source_frame_start=0,
            source_frame_end=24,
            source_frame_count=24,
            source_fps=30,
        )
        stats = _minimal_v5_scene_stats(
            24,
            {
                "0": {"rgb": {"invalid_frame_indices": [1, 3]}},
                "1": {"rgb": {"invalid_frame_indices": [0, 1]}},
            },
        )

        metadata = preprocessing._scene_metadata(
            spec,
            preprocessing.ValidationRecorder(ignore_failures=False),
            stats,
        )

        self.assertEqual(
            metadata["output"]["rgb"]["invalid_frame_indices"],
            [0, 1, 3],
        )
        self.assertEqual(
            metadata["output"]["window_exclusion"]["invalid_frame_indices"],
            [0, 1, 3],
        )

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
            [
                [
                    [0.0, 0.0, 1.0],
                    [3.0, 3.0, 1.0],
                    [0.0, 0.0, 2.0],
                    [10.0, 0.0, 1.0],
                ]
            ],
            dtype=np.float32,
        )
        np.save(source_scene / "tracks_xyz.npy", tracks)
        np.save(source_scene / "queries_xytv.npy", np.zeros((4, 4), dtype=np.float32))
        np.save(
            source_view / "intrinsics.npy",
            np.asarray([2.0, 2.0, 0.0, 0.0], dtype=np.float32),
        )
        np.save(
            source_view / "extrinsics_w2c.npy",
            np.eye(4, dtype=np.float32)[None],
        )
        np.save(
            source_view / "visibility.npy",
            np.ones((1, 4), dtype=bool),
        )
        source_depth = np.zeros((1, 8, 8), dtype=np.float32)
        source_depth[0, 0, 0] = 1.0
        source_depth[0, 0, 2] = 1.0
        source_depth[0, 2, 0] = 1.0
        source_depth[0, 2, 2] = 1.0
        np.save(source_view / "depth.npy", source_depth)
        source_image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
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
            POINT_COUNT=4,
            VIEW_IDS=(0,),
            SOURCE_HEIGHT=8,
            SOURCE_WIDTH=8,
            OUTPUT_HEIGHT=4,
            OUTPUT_WIDTH=4,
            SCALE_X=0.5,
            SCALE_Y=0.5,
            WINDOW_LENGTH=1,
            DEPTH_FRAME_FAILURE_FRACTION=1.0,
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
                    process_workers=1,
                )

        output_view = build_root / "train" / "000000" / "view_0"
        persisted_depth = np.load(output_view / "depth.npy")
        expected_resized_depth = cv2.resize(
            source_depth[0],
            (4, 4),
            interpolation=cv2.INTER_NEAREST,
        )
        np.testing.assert_array_equal(persisted_depth[0], expected_resized_depth)
        self.assertEqual(masks_spy.call_count, 1)
        self.assertEqual(len(captured_calls), 1)
        captured_depth, captured_xy, captured_camera_z, captured_candidates = (
            captured_calls[0]
        )
        np.testing.assert_array_equal(captured_depth, persisted_depth[0])
        np.testing.assert_array_equal(captured_camera_z, [1.0, 1.0, 2.0, 1.0])
        np.testing.assert_array_equal(
            captured_candidates,
            np.asarray([True, True, True, False], dtype=bool),
        )
        self.assertEqual(captured_xy.shape, (4, 2))

        persisted_visibility = np.load(output_view / "visibility.npy")
        np.testing.assert_array_equal(
            persisted_visibility,
            np.asarray([[True, True, True, False]], dtype=bool),
        )
        view_stats = scene_stats[("train", "000000")]["views"]["0"]
        expected_visibility_stats = {
            "visibility_true_before_gating": 4,
            "visibility_true_after_geometric_gating": 3,
            "visibility_removed_by_geometric_gating": 1,
            "depth_consistency_candidate_count": 3,
            "depth_consistency_failure_count": 2,
            "depth_consistency_candidate_count_by_frame": [3],
            "depth_consistency_failure_count_by_frame": [2],
            "depth_consistency_tolerance_metres": 0.05,
            "depth_consistency_no_valid_depth_count": 1,
            "depth_consistency_residual_over_tolerance_count": 1,
            "visibility_true_saved": 3,
        }
        for key, expected in expected_visibility_stats.items():
            self.assertEqual(view_stats[key], expected, key)
        self.assertEqual(
            view_stats["depth_consistency_no_valid_depth_count"]
            + view_stats["depth_consistency_residual_over_tolerance_count"],
            view_stats["depth_consistency_failure_count"],
        )
        self.assertEqual(
            view_stats["prepared_depth"],
            {
                "value_count": 16,
                "finite_count": 16,
                "positive_count": 4,
                "zero_count": 12,
                "negative_count": 0,
                "nonfinite_count": 0,
                "finite_min": 0.0,
                "finite_max": 1.0,
            },
        )
        self.assertEqual(
            view_stats["source_depth_diagnostics"],
            {"negative_count": 0, "nonfinite_count": 0},
        )
        self.assertEqual(view_stats["rgb"]["source_frame_count"], 1)
        self.assertEqual(view_stats["rgb"]["output_frame_count"], 1)
        self.assertEqual(view_stats["rgb"]["written_file_count"], 1)
        self.assertEqual(view_stats["rgb"]["source_decode_failure_count"], 0)
        self.assertEqual(view_stats["rgb"]["placeholder_file_count"], 0)
        self.assertEqual(view_stats["rgb"]["invalid_frame_indices"], [])
        self.assertEqual(view_stats["rgb"]["source_decode_failures"], [])
        self.assertGreater(view_stats["rgb"]["output_bytes"], 0)
        self.assertEqual(view_stats["planned_view_regular_file_count"], 5)
        self.assertEqual(view_stats["written_view_regular_file_count"], 5)
        for seconds in view_stats["exclusive_stage_seconds"].values():
            self.assertTrue(np.isfinite(seconds))
            self.assertGreaterEqual(seconds, 0.0)

        scene = scene_stats[("train", "000000")]
        self.assertEqual(
            scene["tracks"],
            {
                "track_id_count": 4,
                "frame_count": 1,
                "track_sample_count": 4,
                "coordinate_value_count": 12,
                "finite_coordinate_value_count": 12,
                "nonfinite_coordinate_value_count": 0,
            },
        )
        self.assertEqual(scene["prepared_depth_all_views"]["value_count"], 16)
        self.assertEqual(scene["prepared_depth_all_views"]["finite_count"], 16)
        self.assertEqual(scene["prepared_depth_all_views"]["positive_count"], 4)
        self.assertEqual(scene["prepared_depth_all_views"]["zero_count"], 12)
        self.assertEqual(scene["rgb_all_views"]["source_frame_count"], 1)
        self.assertEqual(scene["rgb_all_views"]["output_frame_count"], 1)
        self.assertEqual(scene["rgb_all_views"]["written_file_count"], 1)
        self.assertEqual(scene["rgb_all_views"]["source_decode_failure_count"], 0)
        self.assertEqual(scene["rgb_all_views"]["placeholder_file_count"], 0)
        self.assertEqual(
            scene["depth_track_consistency"],
            {
                "candidate_count": 3,
                "failure_count": 2,
                "invalid_frame_count": 0,
                "invalid_frame_indices": [],
                "per_frame": [
                    {
                        "frame": 0,
                        "candidate_count": 3,
                        "failure_count": 2,
                        "failure_fraction": 2 / 3,
                    }
                ],
            },
        )
        self.assertEqual(
            scene["io_counts"]["source"],
            {
                "source_sequence_frame_count": 1,
                "source_chunk_frame_count": 1,
                "source_camera_chunk_frame_count": 1,
                "logical_asset_reference_count": 7,
                "logical_asset_semantics": (
                    "tracks and queries plus images, depth, intrinsics, extrinsics, and "
                    "visibility per view; each Zarr store counts as one logical asset"
                ),
            },
        )
        self.assertEqual(
            scene["io_counts"]["output"],
            {
                "scene_frame_count": 1,
                "camera_frame_count": 1,
                "jpeg_file_count": 1,
                "npy_file_count": 5,
                "json_file_count": 1,
                "planned_scene_regular_file_count": 7,
                "written_before_scene_metadata_file_count": 6,
            },
        )
        self.assertEqual(
            scene["accounted_scene_stage_total"],
            sum(scene["exclusive_stage_seconds"].values()),
        )
        for seconds in scene["exclusive_stage_seconds"].values():
            self.assertTrue(np.isfinite(seconds))
            self.assertGreaterEqual(seconds, 0.0)
        self.assertEqual(recorder.failures, [])

        output_jpeg = output_view / "rgba_00000.jpg"
        self.assertTrue(output_jpeg.is_file())
        decoded_output = cv2.imread(str(output_jpeg), cv2.IMREAD_COLOR)
        self.assertIsNotNone(decoded_output)
        self.assertEqual(decoded_output.shape, (4, 4, 3))
        self.assertEqual(view_stats["rgb"]["output_bytes"], output_jpeg.stat().st_size)
        np.testing.assert_array_equal(
            np.load(output_view.parent / "tracks_3d.npy"),
            tracks,
        )

    def test_converter_records_invalid_rgb_frame_and_failure_provenance(self):
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
            source_sequence="synthetic_corrupt_rgb",
            environment_family="synthetic",
            source_frame_start=0,
            source_frame_end=2,
            source_frame_count=2,
            source_fps=30,
        )
        source_scene = (
            source_root
            / preprocessing.SOURCE_SUBROOTS[spec.layout]
            / spec.source_sequence
        )
        source_view = source_scene / "0"
        source_view.mkdir(parents=True)
        np.save(
            source_scene / "tracks_xyz.npy",
            np.asarray(
                [[[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]],
                dtype=np.float32,
            ),
        )
        np.save(
            source_scene / "queries_xytv.npy",
            np.zeros((1, 4), dtype=np.float32),
        )
        np.save(
            source_view / "intrinsics.npy",
            np.asarray([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
        )
        np.save(
            source_view / "extrinsics_w2c.npy",
            np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0),
        )
        np.save(source_view / "visibility.npy", np.ones((2, 1), dtype=bool))
        np.save(source_view / "depth.npy", np.ones((2, 2, 2), dtype=np.float32))
        jpeg_objects = np.empty((2,), dtype=object)
        jpeg_objects[0] = np.frombuffer(b"not-a-jpeg", dtype=np.uint8).copy()
        encoded_ok, encoded_image = cv2.imencode(
            ".jpg", np.full((2, 2, 3), 127, dtype=np.uint8)
        )
        self.assertTrue(encoded_ok)
        jpeg_objects[1] = encoded_image.reshape(-1)
        np.save(source_view / "images_jpeg_bytes.npy", jpeg_objects)

        recorder = preprocessing.ValidationRecorder(ignore_failures=False)
        scene_stats = {}
        with mock.patch.multiple(
            preprocessing,
            POINT_COUNT=1,
            VIEW_IDS=(0,),
            SOURCE_HEIGHT=2,
            SOURCE_WIDTH=2,
            OUTPUT_HEIGHT=2,
            OUTPUT_WIDTH=2,
            SCALE_X=1.0,
            SCALE_Y=1.0,
            WINDOW_LENGTH=1,
        ), mock.patch.object(preprocessing, "_validate_source_queries"):
            preprocessing._prepare_scene_roots(build_root, [spec])
            progress = preprocessing.ProgressReporter(
                source_root,
                root / "prepared",
                [spec],
                workers=1,
                update_interval_seconds=0.0,
            )
            with redirect_stdout(io.StringIO()):
                preprocessing._convert_source_group(
                    source_root,
                    build_root,
                    [spec],
                    recorder,
                    scene_stats,
                    process_workers=1,
                    progress=progress,
                )
            metadata = preprocessing._scene_metadata(
                spec,
                recorder,
                scene_stats[(spec.split, spec.scene_id)],
            )
            with redirect_stdout(io.StringIO()):
                preprocessing._write_scene_metadata(
                    build_root,
                    [spec],
                    recorder,
                    scene_stats,
                    progress=progress,
                )
                preprocessing._validate_output_tree(
                    build_root,
                    [spec],
                    process_workers=1,
                    progress=progress,
                )
            progress_snapshot = progress.snapshot()

        rgb_stats = scene_stats[(spec.split, spec.scene_id)]["views"]["0"]["rgb"]
        self.assertEqual(rgb_stats["source_decode_failure_count"], 1)
        self.assertEqual(rgb_stats["placeholder_file_count"], 1)
        self.assertEqual(rgb_stats["invalid_frame_indices"], [0])
        self.assertEqual(len(rgb_stats["source_decode_failures"]), 1)
        self.assertEqual(
            rgb_stats["source_decode_failures"][0],
            {
                "layout": "raw",
                "source_sequence": "synthetic_corrupt_rgb",
                "split": "train",
                "scene_id": "000000",
                "view": 0,
                "local_frame": 0,
                "source_frame": 0,
                "error": "cv2.imdecode returned no image",
                "output_file": "view_0/rgba_00000.jpg",
                "recovery": "black_output_resolution_jpeg_quality_95",
            },
        )
        self.assertEqual(metadata["output"]["rgb"]["invalid_frame_indices"], [0])
        self.assertEqual(
            progress_snapshot["statistics"]["tracks"],
            {
                "track_id_slots": 1,
                "track_samples": 2,
                "coordinate_values": 6,
                "finite_coordinate_values": 6,
                "nonfinite_coordinate_values": 0,
            },
        )
        self.assertEqual(
            progress_snapshot["statistics"]["depth"],
            {
                "values": 8,
                "finite": 8,
                "positive": 8,
                "zero": 0,
                "negative": 0,
                "nonfinite": 0,
                "finite_min": 1.0,
                "finite_max": 1.0,
            },
        )
        self.assertEqual(
            progress_snapshot["statistics"]["visibility"],
            {
                "before_gating": 2,
                "after_geometric_gating": 2,
                "rejected_no_valid_depth": 0,
                "rejected_residual_over_tolerance": 0,
                "accepted": 2,
            },
        )
        output_jpeg_bytes = sum(
            path.stat().st_size
            for path in (build_root / "train" / "000000" / "view_0").glob(
                "rgba_*.jpg"
            )
        )
        self.assertEqual(
            progress_snapshot["statistics"]["rgb"],
            {
                "source_frames": 2,
                "output_frames": 2,
                "output_bytes": output_jpeg_bytes,
                "invalid_frames": 1,
            },
        )
        self.assertEqual(
            progress_snapshot["statistics"]["io"],
            {
                "source_scene_frames": 2,
                "source_camera_frames": 2,
                "source_files": 7,
                "output_scene_frames": 2,
                "output_camera_frames": 2,
                "output_files": 8,
            },
        )
        self.assertEqual(progress_snapshot["progress"]["camera_frames"]["completed"], 2)
        self.assertEqual(progress_snapshot["progress"]["jpegs"]["completed"], 2)
        self.assertEqual(
            progress_snapshot["progress"]["validated_jpegs"]["completed"],
            2,
        )
        self.assertEqual(
            metadata["output"]["rgb"]["decode_failure_placeholder"],
            {
                "image": "constant_black",
                "resolution_hw": [2, 2],
                "quality": 95,
                "training_use": "forbidden_by_invalid_frame_indices",
            },
        )
        self.assertEqual(recorder.failures, [])

        output_jpeg = build_root / "train" / "000000" / "view_0" / "rgba_00000.jpg"
        decoded = cv2.imread(str(output_jpeg), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, (2, 2, 3))
        self.assertEqual(int(np.count_nonzero(decoded)), 0)


class StatisticsAggregationTests(unittest.TestCase):
    def test_successful_nested_statistics_aggregation_reconciles_all_counts(self):
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
        visibility = {
            "visibility_true_before_gating": 4,
            "visibility_true_after_geometric_gating": 3,
            "visibility_removed_by_geometric_gating": 1,
            "depth_consistency_candidate_count": 3,
            "depth_consistency_failure_count": 2,
            "depth_consistency_no_valid_depth_count": 1,
            "depth_consistency_residual_over_tolerance_count": 1,
            "visibility_true_saved": 3,
        }
        scene_stats = {
            ("train", "000000"): {
                "tracks": {
                    "track_id_count": 4,
                    "frame_count": 1,
                    "track_sample_count": 4,
                    "coordinate_value_count": 12,
                    "finite_coordinate_value_count": 9,
                    "nonfinite_coordinate_value_count": 3,
                },
                "prepared_depth_all_views": {
                    "value_count": 16,
                    "finite_count": 15,
                    "positive_count": 4,
                    "zero_count": 10,
                    "negative_count": 1,
                    "nonfinite_count": 1,
                    "finite_min": -0.5,
                    "finite_max": 2.0,
                },
                "rgb_all_views": {
                    "source_frame_count": 1,
                    "output_frame_count": 1,
                    "written_file_count": 1,
                    "output_bytes": 123,
                    "source_decode_failure_count": 0,
                    "placeholder_file_count": 0,
                },
                "depth_track_consistency": {
                    "candidate_count": 3,
                    "failure_count": 2,
                    "invalid_frame_count": 1,
                    "invalid_frame_indices": [0],
                    "per_frame": [
                        {
                            "frame": 0,
                            "candidate_count": 3,
                            "failure_count": 2,
                            "failure_fraction": 2 / 3,
                        }
                    ],
                },
                "window_exclusion": {
                    "window_length": 24,
                    "invalid_frame_indices": [0],
                    "reasons": {
                        "rgb_decode": [],
                        "depth_track_majority_mismatch": [0],
                    },
                    "total_start_count": 0,
                    "excluded_start_count": 0,
                    "legal_start_count": 0,
                },
                "views": {
                    "0": {
                        **visibility,
                        "rgb": {
                            "source_decode_failure_count": 0,
                            "placeholder_file_count": 0,
                            "invalid_frame_indices": [],
                            "source_decode_failures": [],
                        },
                    }
                },
                "io_counts": {
                    "source": {
                        "source_chunk_frame_count": 1,
                        "source_camera_chunk_frame_count": 1,
                        "logical_asset_reference_count": 7,
                    },
                    "output": {
                        "scene_frame_count": 1,
                        "camera_frame_count": 1,
                        "npy_file_count": 5,
                        "json_file_count": 1,
                        "planned_scene_regular_file_count": 7,
                        "written_before_scene_metadata_file_count": 6,
                    },
                },
                "accounted_scene_stage_total": 0.25,
            }
        }
        output_validation = {
            "planned_temporary_tree_file_count_before_root_report": 7,
            "written_temporary_tree_file_count_before_root_report": 7,
            "planned_rgb_file_count": 1,
            "written_rgb_file_count": 1,
            "validated_rgb_file_count": 1,
        }
        timings = {
            "path_and_scene_spec_setup": 1.0,
            "preflight": 2.0,
            "temporary_tree_setup": 3.0,
            "source_conversion": 4.0,
            "scene_metadata_write": 5.0,
            "output_validation": 6.0,
            "process_pool_shutdown": 7.0,
            "measured_total": 30.0,
            "accounted_stage_total": 28.0,
            "unattributed_orchestration": 2.0,
        }

        with mock.patch.object(preprocessing, "VIEW_IDS", (0,)):
            statistics = preprocessing._aggregate_scene_statistics(
                [spec],
                scene_stats,
                output_validation,
                timings,
            )

        self.assertEqual(statistics["scene_count"], 1)
        self.assertEqual(statistics["source_sequence_count"], 1)
        self.assertEqual(
            statistics["tracks"],
            {
                "prepared_scene_track_id_slot_count": 4,
                "prepared_scene_frame_count": 1,
                "track_sample_count": 4,
                "coordinate_value_count": 12,
                "finite_coordinate_value_count": 9,
                "nonfinite_coordinate_value_count": 3,
            },
        )
        self.assertEqual(
            statistics["prepared_depth"],
            {
                "value_count": 16,
                "finite_count": 15,
                "positive_count": 4,
                "zero_count": 10,
                "negative_count": 1,
                "nonfinite_count": 1,
                "finite_min": -0.5,
                "finite_max": 2.0,
            },
        )
        self.assertEqual(
            statistics["rgb"],
            {
                "source_frame_count": 1,
                "output_frame_count": 1,
                "written_file_count": 1,
                "output_bytes": 123,
                "source_decode_failure_count": 0,
                "placeholder_file_count": 0,
                "invalid_scene_frame_count": 0,
                "scenes_with_invalid_rgb_count": 0,
                "invalid_scene_frames": [],
                "source_decode_failures": [],
            },
        )
        self.assertEqual(
            statistics["visibility"],
            {
                **visibility,
                "depth_consistency_tolerance_metres": 0.05,
            },
        )
        self.assertEqual(
            statistics["depth_track_consistency"],
            {
                "candidate_count": 3,
                "failure_count": 2,
                "invalid_frame_count": 1,
                "scenes_with_invalid_depth_count": 1,
                "invalid_scene_frames": [
                    {
                        "split": "train",
                        "scene_id": "000000",
                        "layout": "raw",
                        "source_sequence": "synthetic",
                        "invalid_frame_indices": [0],
                    }
                ],
                "tolerance_metres": 0.05,
                "frame_failure_fraction_threshold": 0.5,
                "scenes": [
                    {
                        "split": "train",
                        "scene_id": "000000",
                        "layout": "raw",
                        "source_sequence": "synthetic",
                        "candidate_count": 3,
                        "failure_count": 2,
                        "invalid_frame_indices": [0],
                        "per_frame": [
                            {
                                "frame": 0,
                                "candidate_count": 3,
                                "failure_count": 2,
                                "failure_fraction": 2 / 3,
                            }
                        ],
                    }
                ],
            },
        )
        self.assertEqual(
            statistics["window_exclusion"],
            {
                "window_length": 24,
                "invalid_scene_frame_count": 1,
                "scenes_with_invalid_frames_count": 1,
                "total_start_count": 0,
                "excluded_start_count": 0,
                "legal_start_count": 0,
                "scenes": [
                    {
                        "split": "train",
                        "scene_id": "000000",
                        "invalid_frame_indices": [0],
                        "reasons": {
                            "rgb_decode": [],
                            "depth_track_majority_mismatch": [0],
                        },
                        "total_start_count": 0,
                        "excluded_start_count": 0,
                        "legal_start_count": 0,
                    }
                ],
            },
        )
        self.assertEqual(
            statistics["io_counts"],
            {
                "source_sequence_count": 1,
                "source_unique_frame_count": 1,
                "source_chunk_frame_count": 1,
                "source_camera_chunk_frame_count": 1,
                "source_distinct_logical_asset_count": 7,
                "source_logical_asset_reference_count": 7,
                "source_logical_asset_semantics": (
                    "tracks and queries plus images, depth, intrinsics, extrinsics, and "
                    "visibility per view; each Zarr store counts as one logical asset"
                ),
                "prepared_scene_count": 1,
                "output_scene_frame_count": 1,
                "output_camera_frame_count": 1,
                "output_jpeg_file_count": 1,
                "output_npy_file_count": 5,
                "output_scene_json_file_count": 1,
                "output_root_json_file_count": 1,
                "converted_file_count_before_scene_metadata": 6,
                "temporary_tree_file_count_before_root_report": 7,
                "published_regular_file_count": 8,
            },
        )
        self.assertEqual(statistics["output_validation"], output_validation)
        self.assertEqual(statistics["timing"]["scene_exclusive_stage_seconds"], 0.25)
        self.assertEqual(statistics["timing"]["wall_seconds"], timings)
        self.assertNotIn("scene_totals", statistics)
        self.assertNotIn("view_totals", statistics)
        self.assertNotIn("timings_seconds", statistics)


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
            self._spec(), recorder, _minimal_v5_scene_stats(1, {})
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

    def test_completed_sidecar_precedes_publication_and_is_not_rewritten_after(self):
        original_replace = preprocessing.os.replace
        observed_at_publication = []

        def observe_replace(source, destination):
            if Path(source).is_dir():
                progress_path = Path(f"{destination}.progress.json")
                observed_at_publication.append(
                    json.loads(progress_path.read_text(encoding="utf-8"))
                )
            return original_replace(source, destination)

        with mock.patch.object(preprocessing.os, "replace", side_effect=observe_replace):
            output_root = self._run_mock_preprocess(ignore=True)

        self.assertEqual(len(observed_at_publication), 1)
        self.assertEqual(observed_at_publication[0]["status"], "completed")
        final_payload = json.loads(
            Path(f"{output_root}.progress.json").read_text(encoding="utf-8")
        )
        self.assertEqual(final_payload, observed_at_publication[0])

    def test_publication_failure_overwrites_completed_sidecar_without_double_timing(self):
        original_replace = preprocessing.os.replace
        completed_at_publication = []

        def fail_dataset_replace(source, destination):
            if Path(source).is_dir():
                progress_path = Path(f"{destination}.progress.json")
                completed_at_publication.append(
                    json.loads(progress_path.read_text(encoding="utf-8"))
                )
                raise OSError("synthetic publication failure")
            return original_replace(source, destination)

        with mock.patch.object(
            preprocessing.os,
            "replace",
            side_effect=fail_dataset_replace,
        ):
            output_root = self._run_mock_preprocess(
                ignore=True,
                expected_exception=OSError,
            )

        self.assertEqual(len(completed_at_publication), 1)
        self.assertEqual(completed_at_publication[0]["status"], "completed")
        failed_payload = json.loads(
            Path(f"{output_root}.progress.json").read_text(encoding="utf-8")
        )
        self.assertEqual(failed_payload["status"], "failed")
        self.assertEqual(
            failed_payload["timing"]["stages_seconds"],
            completed_at_publication[0]["timing"]["stages_seconds"],
        )
        self.assertEqual(failed_payload["error"]["type"], "OSError")

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

        def convert(
            _source_root,
            _build_root,
            _source_specs,
            recorder,
            scene_stats,
            *,
            process_pool,
            process_workers,
            progress,
        ):
            self.assertIsNone(process_pool)
            self.assertEqual(process_workers, 1)
            self.assertIsInstance(progress, preprocessing.ProgressReporter)
            if structural_failure:
                raise ValueError("synthetic structural failure")
            recorder.record(spec, "synthetic_semantic_failure", -1, "non-negative")
            scene_stats[(spec.split, spec.scene_id)] = {"views": {}}

        original_report = preprocessing._report

        def report(
            report_specs,
            recorder,
            _scene_stats,
            _output_validation,
            _timings_seconds,
            workers,
        ):
            return original_report(report_specs, recorder, workers=workers)

        patches = (
            mock.patch.object(preprocessing, "build_scene_specs", return_value=[spec]),
            mock.patch.object(preprocessing, "preflight"),
            mock.patch.object(preprocessing, "_prepare_scene_roots"),
            mock.patch.object(preprocessing, "_group_by_source", return_value=[[spec]]),
            mock.patch.object(preprocessing, "_convert_source_group", side_effect=convert),
            mock.patch.object(preprocessing, "_write_scene_metadata"),
            mock.patch.object(preprocessing, "_validate_output_tree"),
            mock.patch.object(preprocessing, "_report", side_effect=report),
        )
        with patches[0], patches[1], patches[2], patches[3]:
            with patches[4], patches[5], patches[6], patches[7]:
                if expected_exception is None:
                    preprocessing.preprocess(
                        root / "source",
                        output_root,
                        ignore_validation_failures=ignore,
                        workers=1,
                    )
                    self.assertTrue(output_root.is_dir())
                else:
                    with self.assertRaises(expected_exception):
                        preprocessing.preprocess(
                            root / "source",
                            output_root,
                            ignore_validation_failures=ignore,
                            workers=1,
                        )
                    self.assertFalse(output_root.exists())
        return output_root


if __name__ == "__main__":
    unittest.main()
