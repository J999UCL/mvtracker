import unittest

import numpy as np

from mvtracker.datasets.estimated_depth import sample_depth_source
from mvtracker.datasets.mixed_source_schedule import ScheduledSampleRequest


class RecipeDatasetReplayTests(unittest.TestCase):
    def test_request_carries_recorded_depth_and_expected_dali_scene(self):
        request = ScheduledSampleRequest(
            virtual_index=12,
            scene_index=3,
            depth_source="estimated_cleaned",
            expected_scene="scene-003",
        )

        self.assertEqual(request.depth_source, "estimated_cleaned")
        self.assertEqual(request.expected_scene, "scene-003")

    def test_replay_override_preserves_the_native_rng_draw(self):
        replay_rng = np.random.RandomState(72)
        live_rng = np.random.RandomState(72)

        replayed = sample_depth_source(
            replay_rng,
            variable=True,
            replay_depth_source="estimated_cleaned",
        )
        sample_depth_source(live_rng, variable=True)

        self.assertEqual(replayed, "estimated_cleaned")
        self.assertEqual(replay_rng.rand(), live_rng.rand())

    def test_native_depth_draw_uses_the_70_20_10_distribution(self):
        rng = np.random.RandomState(1234)
        draws = [sample_depth_source(rng, variable=True) for _ in range(20_000)]
        frequencies = {
            source: draws.count(source) / len(draws)
            for source in ("gt", "estimated", "estimated_cleaned")
        }

        self.assertAlmostEqual(frequencies["gt"], 0.70, delta=0.02)
        self.assertAlmostEqual(frequencies["estimated"], 0.20, delta=0.02)
        self.assertAlmostEqual(frequencies["estimated_cleaned"], 0.10, delta=0.02)


if __name__ == "__main__":
    unittest.main()
