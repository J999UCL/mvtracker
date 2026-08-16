import unittest

import numpy as np

from scripts.smoke_vggt_omega_chunk import selected_chunk

from mvtracker.preprocessing.vggt_omega_quality import (
    depth_quality_metrics,
    representative_frame_indices,
)


class VGGTQualityTest(unittest.TestCase):
    def test_representative_frames_are_unique_for_short_sequences(self):
        self.assertEqual(representative_frame_indices(1), (0,))
        self.assertEqual(representative_frame_indices(2), (0, 1))
        self.assertEqual(representative_frame_indices(5), (0, 2, 4))

    def test_smoke_chunk_is_one_bounded_contiguous_range(self):
        self.assertEqual(selected_chunk(20, 5, 4), (5, 6, 7, 8))
        with self.assertRaises(ValueError):
            selected_chunk(20, 18, 4)

    def test_depth_metrics_report_metric_error_and_camera_residual(self):
        gt = np.full((2, 2, 2, 2), 2.0, dtype=np.float32)
        estimated = gt.copy()
        estimated[0, 0, 0, 0] = 3.0
        cleaned = np.ones_like(gt, dtype=bool)
        cleaned[1, 1, 1, 1] = False
        known = np.repeat(np.eye(4, dtype=np.float32)[None, None], 4, axis=0).reshape(2, 2, 4, 4)
        known[1, :, 0, 3] = -2.0
        predicted = known.copy()
        predicted[1, :, 0, 3] = -1.0
        metrics = depth_quality_metrics(
            estimated,
            cleaned,
            gt,
            np.array([2.0, 2.0], dtype=np.float32),
            predicted,
            known,
            (0, 1),
        )
        self.assertEqual(metrics["view_count"], 2)
        self.assertAlmostEqual(metrics["mean_absolute_error_m"], 1 / 16)
        self.assertAlmostEqual(metrics["mean_camera_center_rmse_m"], np.sqrt(0.5))
        self.assertAlmostEqual(metrics["cleaned_mean_absolute_relative_error"], 1 / 30)
        self.assertEqual(metrics["sampled_metric_scales"], [2.0, 2.0])
        self.assertAlmostEqual(metrics["cleaned_mask_fraction"], 15 / 16)

    def test_rejects_non_overlapping_depth(self):
        shape = (2, 1, 2, 2)
        cameras = np.repeat(np.eye(4)[None, None], 2, axis=0)
        cameras[1, :, 0, 3] = -1
        with self.assertRaisesRegex(ValueError, "no overlapping"):
            depth_quality_metrics(
                np.zeros(shape),
                np.zeros(shape, dtype=bool),
                np.ones(shape),
                np.ones(1),
                cameras,
                cameras,
                (0,),
            )


if __name__ == "__main__":
    unittest.main()
