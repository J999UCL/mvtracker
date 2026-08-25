"""Focused CPU contracts for the pure physical-batch scheduler."""

from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
from pathlib import Path
import sys
import unittest


_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "mvtracker"
    / "datasets"
    / "physical_batch_scheduler.py"
)
_SPEC = spec_from_file_location("physical_batch_scheduler", _MODULE_PATH)
_MODULE = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
H100_BATCH_CAPACITY = _MODULE.H100_BATCH_CAPACITY
SceneSummary = _MODULE.SceneSummary
schedule_physical_batch = _MODULE.schedule_physical_batch
schedule_rank_local_batch = _MODULE.schedule_rank_local_batch


def _scene(index: int, *, views: int = 1, tracks: int = 512, source: str = "d"):
    return SceneSummary(
        source=source,
        scene=f"scene-{index}",
        cursor=index,
        view_count=views,
        frame_count=24,
        resolution=(384, 512),
        track_count=tracks,
    )


def _identities(wave):
    return [
        (scene.source, scene.scene, scene.cursor)
        for rank in wave.ranks
        for group in rank.groups
        for scene in group.scenes
    ]


class PhysicalBatchSchedulerTests(unittest.TestCase):
    def assert_valid_wave(self, wave, summaries):
        self.assertEqual(len(wave.ranks), 2)
        self.assertEqual(sum(rank.logical_scene_count for rank in wave.ranks), 8)
        self.assertEqual(len(wave.ranks[0].groups), len(wave.ranks[1].groups))
        self.assertEqual(sorted(_identities(wave)), sorted(_identities_from(summaries)))
        for rank in wave.ranks:
            for group in rank.groups:
                self.assertIn(len(group.scenes), (1, 2))
                self.assertEqual(len({scene.shape_key for scene in group.scenes}), 1)
                if len(group.scenes) == 2:
                    self.assertNotIn(group.view_count, {5, 6})

    def test_h100_capacity_table_is_explicit(self):
        self.assertEqual(
            dict(H100_BATCH_CAPACITY.pair_track_capacity_by_views),
            {1: 1024, 2: 1024, 3: 1280, 4: 1024},
        )
        self.assertEqual(H100_BATCH_CAPACITY.singleton_only_views, frozenset({5, 6}))

    def test_all_compatible_scenes_are_paired_before_other_objectives(self):
        summaries = tuple(_scene(i, tracks=100 + i * 50) for i in range(8))
        wave = schedule_physical_batch(summaries)
        self.assert_valid_wave(wave, summaries)
        self.assertEqual(wave.pair_count, 4)
        self.assertEqual(wave.physical_group_count, 2)

    def test_cross_source_pairing_is_allowed(self):
        summaries = tuple(
            _scene(i, source="d" if i % 2 == 0 else "k") for i in range(8)
        )
        wave = schedule_physical_batch(summaries)
        self.assert_valid_wave(wave, summaries)
        self.assertEqual(wave.pair_count, 4)
        self.assertTrue(
            any({scene.source for scene in group.scenes} == {"d", "k"}
                for rank in wave.ranks for group in rank.groups)
        )

    def test_track_mismatch_is_padded_without_rejecting_pair(self):
        summaries = tuple(
            SceneSummary(
                source="d",
                scene=f"scene-{i}",
                cursor=i,
                view_count=1,
                frame_count=24,
                resolution=(384, 512),
                track_count=(100 + i if i % 2 == 0 else 900 + i),
            )
            for i in range(8)
        )
        wave = schedule_physical_batch(summaries)
        self.assert_valid_wave(wave, summaries)
        self.assertEqual(wave.pair_count, 4)
        self.assertTrue(
            any(
                len(group.scenes) == 2
                and len({scene.track_count for scene in group.scenes}) == 2
                for rank in wave.ranks
                for group in rank.groups
            )
        )

    def test_track_counts_are_padded_not_pair_compatibility(self):
        summaries = tuple(
            _scene(i, tracks=(1025 if i < 2 else 512)) for i in range(8)
        )
        wave = schedule_physical_batch(summaries)
        self.assert_valid_wave(wave, summaries)
        self.assertEqual(wave.pair_count, 4)

    def test_query_schedule_start_does_not_prevent_scene_batching(self):
        summaries = tuple(
            _MODULE.SceneSummary(
                source="d",
                scene=f"scene-{index}",
                cursor=index,
                view_count=2,
                frame_count=24,
                resolution=(384, 512),
                track_count=512,
                schedule_start=index % 3,
            )
            for index in range(8)
        )
        wave = schedule_physical_batch(summaries)
        self.assertEqual(wave.pair_count, 4)

    def test_views_five_and_six_are_always_singletons(self):
        for views in (5, 6):
            summaries = tuple(_scene(i, views=views) for i in range(8))
            wave = schedule_physical_batch(summaries)
            self.assert_valid_wave(wave, summaries)
            self.assertEqual(wave.pair_count, 0)
            self.assertEqual(wave.physical_group_count, 4)

    def test_shape_mismatch_cannot_be_paired(self):
        summaries = tuple(
            _scene(i, views=1 if i < 4 else 2) for i in range(8)
        )
        wave = schedule_physical_batch(summaries)
        self.assert_valid_wave(wave, summaries)
        self.assertEqual(wave.pair_count, 4)
        for rank in wave.ranks:
            self.assertEqual(
                {group.shape_key[0] for group in rank.groups}, {1, 2}
            )

    def test_determinism_is_exact(self):
        summaries = tuple(
            _scene(i, views=(i % 4) + 1, tracks=256 + i * 31)
            for i in range(8)
        )
        expected = schedule_physical_batch(summaries)
        for _ in range(100):
            self.assertEqual(schedule_physical_batch(summaries), expected)

    def test_one_safe_pair_is_not_discarded_for_equal_group_counts(self):
        summaries = (
            _scene(0, views=1, tracks=256),
            _scene(1, views=1, tracks=512),
            *tuple(_scene(i, views=5, tracks=256) for i in range(2, 8)),
        )
        wave = schedule_physical_batch(summaries)
        self.assert_valid_wave(wave, summaries)
        self.assertEqual(wave.pair_count, 0)
        self.assertEqual([len(rank.groups) for rank in wave.ranks], [4, 4])

    def test_exhaustive_two_shape_assignments_preserve_invariants(self):
        # Exhaust all 2^8 assignments of two compatible tensor shapes.  This
        # covers every possible placement of pairable/non-pairable boundaries
        # while keeping the test strictly CPU-only and small.
        for assignment in product((1, 2), repeat=8):
            summaries = tuple(
                _scene(i, views=views, tracks=128 + i * 17)
                for i, views in enumerate(assignment)
            )
            wave = schedule_physical_batch(summaries)
            self.assert_valid_wave(wave, summaries)
            self.assertEqual(
                wave,
                schedule_physical_batch(summaries),
                msg=f"non-deterministic assignment {assignment}",
            )

    def test_rejects_wrong_scene_count_and_duplicate_identity(self):
        with self.assertRaisesRegex(ValueError, "exactly 8"):
            schedule_physical_batch(tuple(_scene(i) for i in range(7)))
        duplicate = list(_scene(i) for i in range(8))
        duplicate[7] = duplicate[0]
        with self.assertRaisesRegex(ValueError, "unique"):
            schedule_physical_batch(tuple(duplicate))

    def test_rejects_invalid_scene_metadata(self):
        invalid = list(_scene(i) for i in range(8))
        invalid[0] = _scene(0, views=7)
        with self.assertRaisesRegex(ValueError, "view_count"):
            schedule_physical_batch(tuple(invalid))
        invalid[0] = _scene(0, tracks=0)
        with self.assertRaisesRegex(ValueError, "track_count"):
            schedule_physical_batch(tuple(invalid))

    def test_rank_local_scheduler_does_not_require_eight_scenes(self):
        summaries = tuple(_scene(i, views=1, tracks=256 + i) for i in range(4))
        groups = schedule_rank_local_batch(summaries)
        self.assertEqual(
            sorted(
                (scene.source, scene.scene, scene.cursor)
                for group in groups
                for scene in group.scenes
            ),
            sorted(_identities_from(summaries)),
        )
        self.assertEqual(len(groups), 2)


def _identities_from(summaries):
    return [
        (scene.source, scene.scene, scene.cursor)
        for scene in summaries
    ]


if __name__ == "__main__":
    unittest.main()
