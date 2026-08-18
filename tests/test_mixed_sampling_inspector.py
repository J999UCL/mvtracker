import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PATH = Path(__file__).resolve().parents[1] / "tools/inspect_mixed_sampling.py"
SPEC = importlib.util.spec_from_file_location("inspect_mixed_sampling", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeSchedule:
    world_size = 2

    def sample_source(self, source, cursor, rank):
        return SimpleNamespace(
            source=source,
            request=SimpleNamespace(
                virtual_index=cursor * 2 + rank,
                scene_index=cursor * 2 + rank,
            ),
        )

    def sample_step(self, cursors):
        selected = []
        occurrences = {source: 0 for source in cursors}
        for microbatch, source in enumerate(MODULE.SOURCE_PATTERN):
            cursor = cursors[source] + occurrences[source]
            occurrences[source] += 1
            for rank in range(self.world_size):
                selected.append(
                    SimpleNamespace(
                        microbatch=microbatch,
                        rank=rank,
                        source=source,
                        request=self.sample_source(source, cursor, rank).request,
                    )
                )
        return selected


class FakeDataset:
    def __init__(self, source, fail_virtual=None):
        self.source = source
        self.fail_virtual = fail_virtual

    def __getitem__(self, request):
        if request.virtual_index == self.fail_virtual:
            return None, False
        sample = SimpleNamespace(
            metadata={
                "scene_name": f"{self.source}-{request.scene_index}",
                "seed": 72 + request.virtual_index,
                "window_start": 0,
                "window_end_exclusive": 24,
                "selected_views": [0, 1],
            },
            trajectory=SimpleNamespace(shape=(2, 24, 16, 3)),
            apply_rgb_aug=True,
            apply_depth_aug=False,
        )
        return sample, True


class MixedSamplingInspectorTests(unittest.TestCase):
    def test_collects_current_dkdk_rank_order(self):
        rows = MODULE.collect_sequential_samples(
            FakeSchedule(),
            {
                "diegesis": FakeDataset("diegesis"),
                "mvkubric": FakeDataset("mvkubric"),
            },
            steps=1,
        )
        self.assertEqual(len(rows), 8)
        self.assertEqual(
            [(row["microbatch"], row["rank"], row["source"]) for row in rows],
            [
                (0, 0, "diegesis"), (0, 1, "diegesis"),
                (1, 0, "mvkubric"), (1, 1, "mvkubric"),
                (2, 0, "diegesis"), (2, 1, "diegesis"),
                (3, 0, "mvkubric"), (3, 1, "mvkubric"),
            ],
        )

    def test_failed_rank_pair_advances_together(self):
        rows = MODULE.collect_sequential_samples(
            FakeSchedule(),
            {
                "diegesis": FakeDataset("diegesis", fail_virtual=1),
                "mvkubric": FakeDataset("mvkubric"),
            },
            steps=1,
        )
        first_pair = rows[:2]
        self.assertEqual([row["virtual_index"] for row in first_pair], [2, 3])
        self.assertEqual([row["attempt"] for row in first_pair], [1, 1])

    def test_whole_step_selection_is_identical_including_retry(self):
        datasets = {
            "diegesis": FakeDataset("diegesis", fail_virtual=1),
            "mvkubric": FakeDataset("mvkubric", fail_virtual=5),
        }
        sequential = MODULE.collect_sequential_samples(
            FakeSchedule(), datasets, steps=2
        )
        whole_step = MODULE.collect_whole_step_samples(
            FakeSchedule(), datasets, steps=2
        )
        self.assertEqual(whole_step, sequential)


if __name__ == "__main__":
    unittest.main()
