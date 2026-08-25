"""Focused CPU tests for metadata-only training recipe planning."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mvtracker.datasets.mixed_source_schedule import BalancedMixedSourceSchedule
from mvtracker.datasets.physical_batch_scheduler import (
    BatchCapacity,
    schedule_physical_batch,
)
from mvtracker.datasets.training_recipe import (
    RecipeReader,
    plan_training_recipe,
    plan_training_recipe_parallel,
)


class _MetadataDataset:
    def __init__(self, source: str, *, reject_virtual_index: int | None = None):
        self.source = source
        self.reject_virtual_index = reject_virtual_index
        self.materialize_calls = 0

    def plan_sample(self, request):
        if request.virtual_index == self.reject_virtual_index:
            return None
        seed = 1000 + request.virtual_index
        depth_source = "estimated_cleaned" if request.virtual_index % 3 == 0 else "gt"
        return SimpleNamespace(
            virtual_index=request.virtual_index,
            scene_index=request.scene_index,
            sequence=f"{self.source}-scene-{request.scene_index}",
            seed=seed,
            frame_indices=np.array([2, 3, 4], dtype=np.int64),
            views=(0, 1),
            selected_global_track_indices=np.array([7, 9, 12], dtype=np.int64),
            track_count=3,
            query_points_3d=np.array([[1, 0, 0, 0], [1, 0, 0, 0]]),
            output_size=(384, 512),
            depth_source=depth_source,
            apply_rgb_aug=True,
            rgb_augmentation={"brightness": np.float32(0.2)},
            apply_depth_aug=False,
            depth_patch_operations=(),
            augmentation_seed=seed + 1,
        )

    def materialize_sample(self, _plan):
        self.materialize_calls += 1
        raise AssertionError("recipe planning must not materialize media")


class TrainingRecipeTests(unittest.TestCase):
    def _plan(
        self,
        root: Path,
        *,
        steps: int = 2,
        heartbeat_seconds: float = 10.0,
        worker_count: int = 1,
    ):
        datasets = {
            "diegesis": _MetadataDataset("diegesis", reject_virtual_index=0),
            "kubric": _MetadataDataset("kubric"),
        }
        schedule = BalancedMixedSourceSchedule(
            {"diegesis": 5, "kubric": 4},
            ("diegesis", "kubric", "diegesis", "kubric"),
            world_size=2,
            master_seed=31,
        )
        logs = []
        summary = plan_training_recipe(
            root / "recipe",
            datasets=datasets,
            schedule=schedule,
            step_count=steps,
            manifest={
                "recipe_name": "unit-test",
                "source_commit": "abc123",
                "scene_lists": {"diegesis": ["a"], "kubric": ["b"]},
            },
            heartbeat_seconds=heartbeat_seconds,
            log=logs.append,
            worker_count=worker_count,
        )
        return datasets, logs, summary

    def test_planner_writes_replayable_rank_streams_without_media(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            datasets, logs, summary = self._plan(root)
            reader = RecipeReader(root / "recipe")
            reader.validate()
            steps = list(reader.steps())
            self.assertEqual(len(steps), 2)
            self.assertEqual(
                [len(step["logical_samples"]) for step in steps], [8, 8]
            )
            self.assertEqual(
                [
                    sample["logical_index"]
                    for sample in steps[0]["logical_samples"]
                ],
                list(range(8)),
            )
            self.assertEqual(
                {
                    index
                    for group in steps[0]["physical_groups"]
                    for index in group["logical_indices"]
                },
                set(range(8)),
            )

            records = [
                record
                for rank in range(2)
                for record in reader.records(rank)
            ]
            self.assertEqual(len(records), 16)
            self.assertEqual([len(list(reader.records(rank))) for rank in range(2)], [8, 8])
            self.assertEqual(
                len({(item.step, item.microbatch, item.scheduled_rank) for item in records}),
                16,
            )
            self.assertTrue(all(item.frames == (2, 3, 4) for item in records))
            self.assertTrue(all(item.views == (0, 1) for item in records))
            self.assertTrue(all(item.tracks == (7, 9, 12) for item in records))
            self.assertTrue(all(item.physical.rank == item.rank for item in records))
            self.assertTrue(any(item.retry_count == 1 for item in records))
            self.assertEqual(sum(dataset.materialize_calls for dataset in datasets.values()), 0)
            self.assertEqual(summary["records"], 16)
            self.assertGreater(summary["retries"], 0)
            self.assertTrue(logs[0].startswith("recipe planner start"))
            self.assertTrue(logs[-1].startswith("recipe planner complete"))

            request = records[0].replay_request(SimpleNamespace)
            self.assertEqual(request.virtual_index, records[0].request["virtual_index"])

    def test_manifest_completion_summary_and_estimated_requests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._plan(root, steps=1)
            recipe = root / "recipe"
            manifest = json.loads((recipe / "manifest.json").read_text())
            summary = json.loads((recipe / "summary.json").read_text())
            estimated = [
                json.loads(line)
                for line in (recipe / "estimated-depth-requests.jsonl").read_text().splitlines()
            ]

            self.assertTrue(manifest["complete"])
            self.assertEqual(manifest["expected_records"], 8)
            self.assertEqual(manifest["actual_records"], 8)
            self.assertEqual(manifest["rank_record_counts"], [4, 4])
            self.assertEqual(summary["records"], 8)
            self.assertEqual(
                estimated,
                sorted(estimated, key=lambda item: (item["source"], item["scene"])),
            )
            self.assertEqual(
                len({(item["source"], item["scene"]) for item in estimated}),
                len(estimated),
            )
            self.assertTrue(
                all(item["planned_depth_sources"] for item in estimated)
            )

    def test_heartbeat_reports_while_plan_sample_is_slow(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = _MetadataDataset("only")
            original = dataset.plan_sample

            def slow_plan(request):
                time.sleep(0.03)
                return original(request)

            dataset.plan_sample = slow_plan
            schedule = BalancedMixedSourceSchedule(
                {"only": 2}, ("only",) * 4, world_size=2, master_seed=1
            )
            logs = []
            plan_training_recipe(
                Path(temporary) / "recipe",
                datasets={"only": dataset},
                schedule=schedule,
                step_count=1,
                manifest={"recipe_name": "heartbeat"},
                heartbeat_seconds=0.01,
                log=logs.append,
            )
            self.assertTrue(any(line.startswith("recipe heartbeat") for line in logs))

    def test_parallel_planning_matches_serial_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._plan(root / "serial", steps=4, worker_count=1)
            datasets = {
                "diegesis": _MetadataDataset(
                    "diegesis", reject_virtual_index=0
                ),
                "kubric": _MetadataDataset("kubric"),
            }
            schedule = BalancedMixedSourceSchedule(
                {"diegesis": 5, "kubric": 4},
                ("diegesis", "kubric", "diegesis", "kubric"),
                world_size=2,
                master_seed=31,
            )
            summary = plan_training_recipe_parallel(
                root / "parallel" / "recipe",
                datasets=datasets,
                schedule=schedule,
                step_count=4,
                manifest={
                    "recipe_name": "unit-test",
                    "source_commit": "abc123",
                    "scene_lists": {"diegesis": ["a"], "kubric": ["b"]},
                },
                worker_count=4,
                block_steps=2,
                log=lambda _: None,
            )
            self.assertEqual(sum(summary["source_plan_calls"].values()), 34)
            for name in (
                "rank-0.jsonl",
                "rank-1.jsonl",
                "steps.jsonl",
                "estimated-depth-requests.jsonl",
            ):
                self.assertEqual(
                    (root / "serial" / "recipe" / name).read_text(),
                    (root / "parallel" / "recipe" / name).read_text(),
                )

    def test_parallel_planner_stores_synchronized_global_physical_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "recipe"
            datasets = {
                "diegesis": _MetadataDataset("diegesis"),
                "kubric": _MetadataDataset("kubric"),
            }
            schedule = BalancedMixedSourceSchedule(
                {"diegesis": 5, "kubric": 4},
                ("diegesis", "kubric", "diegesis", "kubric"),
                world_size=2,
                master_seed=31,
            )
            capacity = BatchCapacity(
                name="test",
                rank_count=2,
                logical_scenes_per_rank=4,
                max_group_size=2,
                pair_track_capacity_by_views=((2, 2048),),
                singleton_only_views=frozenset(),
            )
            plan_training_recipe_parallel(
                root,
                datasets=datasets,
                schedule=schedule,
                step_count=2,
                manifest={},
                worker_count=2,
                block_steps=1,
                log=lambda _: None,
                physical_scheduler=lambda summaries: schedule_physical_batch(
                    summaries, capacity=capacity
                ),
            )
            reader = RecipeReader(root)
            reader.validate()
            for step in reader.steps():
                counts = Counter(
                    group["rank"] for group in step["physical_groups"]
                )
                self.assertEqual(counts[0], counts[1])
                self.assertTrue(
                    all(
                        len(group["logical_indices"]) == 2
                        for group in step["physical_groups"]
                    )
                )


if __name__ == "__main__":
    unittest.main()
