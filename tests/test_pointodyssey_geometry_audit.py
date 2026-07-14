import unittest
import tempfile
from pathlib import Path

import numpy as np

from scripts import pointodyssey_contract as contract
from scripts import pointodyssey_geometry_audit as audit


class SamplingPlanTests(unittest.TestCase):
    def test_anchor_frames_are_deterministic_unique_and_offset_safe(self):
        for layout, count in audit.SAMPLE_COUNTS.items():
            frame_count = contract.SOURCE_FRAME_COUNTS[layout]
            first = audit.sample_anchor_frames(frame_count, count)
            second = audit.sample_anchor_frames(frame_count, count)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(len(first), count)
            self.assertEqual(len(np.unique(first)), count)
            self.assertGreaterEqual(int(first.min()), 2)
            self.assertLessEqual(int(first.max()), frame_count - 3)

    def test_contract_has_all_twenty_sources(self):
        sources = list(contract.unique_source_keys())
        self.assertEqual(len(sources), 20)
        self.assertEqual(sum(layout == "raw" for layout, _ in sources), 8)
        self.assertEqual(sum(layout == "short" for layout, _ in sources), 8)
        self.assertEqual(sum(layout == "long" for layout, _ in sources), 4)


class GeometryTests(unittest.TestCase):
    def test_projection_uses_w2c_optical_z_and_source_visibility(self):
        tracks = np.zeros((contract.POINT_COUNT, 3), dtype=np.float32)
        tracks[:, 2] = 2.0
        visibility = np.zeros((contract.POINT_COUNT,), dtype=bool)
        visibility[0] = True
        extrinsics = np.eye(4, dtype=np.float32)
        intrinsics = np.asarray([100.0, 100.0, 960.0, 540.0], dtype=np.float32)

        x, y, camera_z = audit.project_visible_tracks(
            tracks, extrinsics, intrinsics, visibility
        )

        np.testing.assert_array_equal(x, [960.0])
        np.testing.assert_array_equal(y, [540.0])
        np.testing.assert_array_equal(camera_z, [2.0])

    def test_nearest_bilinear_and_neighborhood_are_distinct(self):
        depth = np.zeros(
            (contract.SOURCE_HEIGHT, contract.SOURCE_WIDTH), dtype=np.float32
        )
        depth[20, 10] = 1.0
        depth[20, 11] = 2.0
        depth[21, 10] = 3.0
        depth[21, 11] = 4.0

        residuals = audit.sample_depth_residuals(
            depth,
            np.asarray([10.25]),
            np.asarray([20.5]),
            np.asarray([1.0]),
        )

        np.testing.assert_allclose(residuals["nearest_pixel"], [2.0])
        np.testing.assert_allclose(residuals["bilinear"], [1.25])
        np.testing.assert_allclose(residuals["neighborhood_3x3"], [0.0])

    def test_nonpositive_depth_is_invalid_not_repaired(self):
        depth = np.zeros(
            (contract.SOURCE_HEIGHT, contract.SOURCE_WIDTH), dtype=np.float32
        )
        residuals = audit.sample_depth_residuals(
            depth,
            np.asarray([10.0]),
            np.asarray([20.0]),
            np.asarray([1.0]),
        )
        for values in residuals.values():
            self.assertEqual(values.size, 0)

    def test_bilinear_ignores_zero_weight_invalid_neighbors_and_clamps_edge(self):
        depth = np.zeros(
            (contract.SOURCE_HEIGHT, contract.SOURCE_WIDTH), dtype=np.float32
        )
        depth[0, 0] = 1.5
        depth[20, 10] = 2.0
        depth[-1, -1] = 3.0
        residuals = audit.sample_depth_residuals(
            depth,
            np.asarray([-0.25, 10.0, contract.SOURCE_WIDTH - 0.75]),
            np.asarray([-0.25, 20.0, contract.SOURCE_HEIGHT - 0.75]),
            np.asarray([1.0, 1.0, 1.0]),
        )
        np.testing.assert_allclose(residuals["bilinear"], [0.5, 1.0, 2.0])

    def test_temporal_pairing_keeps_the_same_candidate_identities(self):
        by_offset = {}
        for offset in audit.TEMPORAL_OFFSETS:
            values = np.asarray([0.0, 1.0], dtype=np.float32)
            if offset == 1:
                values[1] = np.nan
            by_offset[offset] = {mode: values.copy() for mode in audit.SAMPLING_MODES}

        paired = audit.paired_temporal_residuals(by_offset)

        for mode in audit.SAMPLING_MODES:
            np.testing.assert_array_equal(paired[mode][0], [0.0])
            np.testing.assert_array_equal(paired[mode][1], [0.0])


class ReportingTests(unittest.TestCase):
    def test_report_cannot_be_written_inside_source_tree(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            with self.assertRaisesRegex(ValueError, "outside the read-only"):
                audit.audit(source, source / "audit.json")

    def test_summary_reports_coverage_and_reference_fraction(self):
        summary = audit.summarize_residuals(
            np.asarray([-0.01, 0.02, 0.10], dtype=np.float32), candidate_count=4
        )
        self.assertEqual(summary["candidate_count"], 4)
        self.assertEqual(summary["valid_positive_depth_count"], 3)
        self.assertEqual(summary["invalid_or_missing_depth_count"], 1)
        self.assertAlmostEqual(summary["valid_positive_depth_fraction"], 0.75)
        self.assertAlmostEqual(
            summary["absolute_residual_metres"][
                "fraction_at_or_below_raycast_visibility_reference_0_05m"
            ],
            2.0 / 3.0,
        )


if __name__ == "__main__":
    unittest.main()
