import importlib.util
import sys
import unittest
from pathlib import Path


def _load_module(name):
    path = Path(__file__).resolve().parents[1] / "mvtracker/datasets" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BalancedMixedSourceSchedule = _load_module(
    "mixed_source_schedule"
).BalancedMixedSourceSchedule
select_scene_names = _load_module("scene_selection").select_scene_names


class BalancedMixedSourceScheduleTests(unittest.TestCase):
    def setUp(self):
        self.schedule = BalancedMixedSourceSchedule(
            {"diegesis": 17, "mvkubric": 98},
            ("diegesis", "mvkubric", "diegesis", "mvkubric"),
            world_size=2,
            master_seed=1234,
        )

    def test_each_rank_uses_exact_dkdk_pattern(self):
        for rank in range(2):
            self.assertEqual(
                [self.schedule.sample(8, microbatch, rank).source for microbatch in range(4)],
                ["diegesis", "mvkubric", "diegesis", "mvkubric"],
            )

    def test_one_thousand_updates_are_exactly_balanced(self):
        counts = {"diegesis": 0, "mvkubric": 0}
        for step in range(1000):
            for rank in range(2):
                for microbatch in range(4):
                    counts[self.schedule.sample(step, microbatch, rank).source] += 1
        self.assertEqual(counts, {"diegesis": 4000, "mvkubric": 4000})

    def test_global_scene_cycles_are_balanced_and_shuffled(self):
        samples = [
            self.schedule.sample(step, microbatch, rank)
            for step in range(30)
            for microbatch in range(4)
            for rank in range(2)
            if self.schedule.sample(step, microbatch, rank).source == "diegesis"
        ]
        scene_indices = [sample.request.scene_index for sample in samples]
        for start in range(0, 102, 17):
            self.assertEqual(sorted(scene_indices[start : start + 17]), list(range(17)))
        self.assertNotEqual(scene_indices[:17], scene_indices[17:34])

    def test_retry_and_reconstruction_are_identical(self):
        expected = self.schedule.sample(91, 2, 1)
        self.assertEqual(expected, self.schedule.sample(91, 2, 1))
        restored = BalancedMixedSourceSchedule(
            {"diegesis": 17, "mvkubric": 98},
            ("diegesis", "mvkubric", "diegesis", "mvkubric"),
            world_size=2,
            master_seed=1234,
        )
        restored.load_state_dict(self.schedule.state_dict())
        self.assertEqual(expected, restored.sample(91, 2, 1))

    def test_failed_slot_advances_within_the_same_source(self):
        first = self.schedule.sample(3, 0, 0, attempt=0)
        retry = self.schedule.sample(3, 0, 0, attempt=1)
        self.assertEqual(first.source, retry.source)
        self.assertNotEqual(first.request.virtual_index, retry.request.virtual_index)
        self.assertNotEqual(first.request.scene_index, retry.request.scene_index)

    def test_source_cursor_matches_no_failure_step_schedule(self):
        for step in range(10):
            for rank in range(2):
                self.assertEqual(
                    self.schedule.sample(step, 2, rank),
                    self.schedule.sample_source("diegesis", step * 2 + 1, rank),
                )

    def test_schedule_leaves_view_selection_to_each_dataset(self):
        for microbatch in range(4):
            self.assertIsNone(self.schedule.sample(0, microbatch, 0).request.view_count)

    def test_sample_step_selects_the_same_eight_requests_together(self):
        selected = self.schedule.sample_step({"diegesis": 6, "mvkubric": 10})
        self.assertEqual(len(selected), 8)
        self.assertEqual(
            [(item.microbatch, item.rank, item.source) for item in selected],
            [
                (0, 0, "diegesis"), (0, 1, "diegesis"),
                (1, 0, "mvkubric"), (1, 1, "mvkubric"),
                (2, 0, "diegesis"), (2, 1, "diegesis"),
                (3, 0, "mvkubric"), (3, 1, "mvkubric"),
            ],
        )
        self.assertEqual(
            [item.request.virtual_index for item in selected],
            [12, 13, 20, 21, 14, 15, 22, 23],
        )


class SceneSelectionTests(unittest.TestCase):
    def test_mvkubric_training_and_validation_allowlists_are_disjoint(self):
        available = ["101", "102", *[str(scene) for scene in range(900, 1000)]]
        training = select_scene_names(
            available, include=[str(scene) for scene in range(900, 998)]
        )
        validation = select_scene_names(available, include=["101", "102"])
        self.assertEqual(training, [str(scene) for scene in range(900, 998)])
        self.assertEqual(validation, ["101", "102"])
        self.assertFalse(set(training) & set(validation))

    def test_selection_preserves_canonical_order_and_supports_exclusion(self):
        self.assertEqual(
            select_scene_names(
                ["2", "10", "alpha"], include=["alpha", "2"]
            ),
            ["2", "alpha"],
        )

    def test_selection_rejects_unknown_or_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "unknown scene IDs"):
            select_scene_names(["900"], include=["901"])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            select_scene_names(["900"], include=["900", "900"])


if __name__ == "__main__":
    unittest.main()
