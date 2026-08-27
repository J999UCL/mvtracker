import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from mvtracker.datasets.estimated_depth import RuntimeDepthLoad, RuntimeRecipeDepthStore


class RuntimeRecipeDepthStoreTest(unittest.TestCase):
    def test_load_reports_phase_timings_and_deletes_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_root = root / "step-000012" / "sample-03"
            sample_root.mkdir(parents=True)
            expected_depth = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)
            expected_mask = expected_depth > 3
            np.save(sample_root / "depth.npy", expected_depth, allow_pickle=False)
            np.save(sample_root / "cleaned_mask.npy", expected_mask, allow_pickle=False)
            (sample_root / "ready").touch()

            with (
                mock.patch(
                    "mvtracker.datasets.estimated_depth.time.perf_counter",
                    side_effect=(10.0, 10.25, 10.75, 10.8),
                ),
                mock.patch.object(os, "fsync", wraps=os.fsync) as fsync,
            ):
                result = RuntimeRecipeDepthStore(root).load(12, 3)

            self.assertIsInstance(result, RuntimeDepthLoad)
            np.testing.assert_array_equal(result.depth, expected_depth)
            np.testing.assert_array_equal(result.cleaned_mask, expected_mask)
            self.assertEqual(result.depth.dtype, np.float32)
            self.assertEqual(result.cleaned_mask.dtype, np.bool_)
            self.assertEqual(result.ready_wait_seconds, 0.25)
            self.assertEqual(result.read_seconds, 0.5)
            self.assertAlmostEqual(result.delete_seconds, 0.05)
            self.assertAlmostEqual(result.total_seconds, 0.8)
            self.assertEqual(result.byte_count, expected_depth.nbytes + expected_mask.nbytes)
            self.assertFalse(sample_root.exists())
            store = RuntimeRecipeDepthStore(root)
            self.assertEqual(
                store.inventory(),
                {"ready_samples": 0, "resident_samples": 1, "effective_samples": 1},
            )
            store.release(12, 3)
            self.assertEqual(store.inventory()["effective_samples"], 0)
            fsync.assert_not_called()


if __name__ == "__main__":
    unittest.main()
