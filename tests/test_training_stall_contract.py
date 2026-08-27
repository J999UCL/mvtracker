import ast
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_functions(*names):
    tree = ast.parse((ROOT / "mvtracker/cli/train.py").read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "train.py", "exec"), namespace)
    return tuple(namespace[name] for name in names)


class _Fabric:
    device = torch.device("cpu")
    world_size = 2

    def __init__(self):
        self.reduce_calls = 0
        self.gathered = None

    def all_reduce(self, tensor, reduce_op="mean"):
        self.reduce_calls += 1
        return tensor * 2

    def all_gather(self, tensor):
        return self.gathered if self.gathered is not None else torch.stack((tensor, tensor))


class TrainingStallContractTests(unittest.TestCase):
    def test_scalar_mapping_uses_one_collective(self):
        (reduce_dict,) = _load_functions("_reduce_scalar_dict")
        fabric = _Fabric()

        reduced = reduce_dict(fabric, {"a": 1.0, "b": 3.0, "c": 5.0}, "sum")

        self.assertEqual(reduced, {"a": 2.0, "b": 6.0, "c": 10.0})
        self.assertEqual(fabric.reduce_calls, 1)

    def test_recipe_step_and_success_share_one_gather(self):
        (check,) = _load_functions("_check_recipe_step_materialization")
        fabric = _Fabric()
        fabric.gathered = torch.tensor(((7, 1), (7, 1)))

        self.assertTrue(check(fabric, 7, True))

        fabric.gathered = torch.tensor(((7, 1), (7, 0)))
        self.assertFalse(check(fabric, 7, True))

    def test_recipe_scene_losses_use_a_fixed_tensor_gather(self):
        (gather_losses,) = _load_functions("_gather_recipe_scene_losses")
        fabric = _Fabric()
        fabric.world_size = 1
        record = {
            "source": "diegesis",
            "scene": "scene-a",
            "scene_index": 0,
            "rank": 0,
            "physical_group_index": 1,
            "views": 2,
            "tracks": 8,
            "window_start": 3,
            "window_end_exclusive": 27,
            "virtual_index": 11,
            "seed": 72,
            "depth_source": "estimated",
            "rgb_augmented": True,
            "depth_augmented": False,
            "cropping_enabled": True,
            "scene_transform_enabled": True,
            "camera_noise_enabled": True,
            "selected_views": [1, 3],
            "trajectory_loss": 0.2,
            "visibility_loss": 0.1,
            "raw_visibility_loss": 1.0,
            "total_loss": 0.3,
            "optimizer_step": 4,
        }

        gathered = gather_losses(
            fabric,
            [record],
            ("diegesis",),
            {"diegesis": ["scene-a"]},
        )

        self.assertEqual(gathered[0]["scene"], "scene-a")
        self.assertEqual(gathered[0]["selected_views"], [1, 3])
        self.assertAlmostEqual(gathered[0]["total_loss"], 0.3)

    def test_final_recipe_uses_two_decoded_groups(self):
        config = yaml.safe_load(
            (
                ROOT
                / "configs/experiment/diegesis_syn4d_mvkubric_recipe_da3_ddp_5000.yaml"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(
            config["datasets"]["train"]["physical_batching"]["decoded_queue_depth"],
            2,
        )

    def test_decode_prefetch_has_no_per_item_join(self):
        source = (
            ROOT / "mvtracker/datasets/mixed_physical_loader.py"
        ).read_text(encoding="utf-8")
        start = source.index("class PhysicalStepPrefetchIterator")
        end = source.index("class _PrefetchedStepGroups", start)
        producer = source[start:end]
        self.assertNotIn("self.ready.join()", producer)
        self.assertIn("queue_depth: int = 2", producer)

    def test_execution_recipe_path_contains_no_runtime_planner(self):
        train_source = (ROOT / "mvtracker/cli/train.py").read_text(encoding="utf-8")
        loader_source = (
            ROOT / "mvtracker/datasets/mixed_physical_loader.py"
        ).read_text(encoding="utf-8")
        self.assertIn("ExecutionRecipeReader", train_source)
        self.assertIn("execution recipes must never call plan_sample", train_source)
        self.assertNotIn("def _plan_recipe_step", loader_source)
        self.assertIn("materialize_recipe_record", loader_source)

    def test_recipe_lookahead_materializes_only_rank_local_records(self):
        from mvtracker.datasets import mixed_physical_loader as loader

        records = tuple(
            SimpleNamespace(
                source="diegesis",
                logical_index=index,
                physical=SimpleNamespace(rank=1, group=index, position=0),
                recipe=SimpleNamespace(
                    scene=f"scene-{index}",
                    request={"virtual_index": index},
                    retry_count=0,
                ),
            )
            for index in range(2)
        )

        class Dataset:
            def __init__(self):
                self.seen = []

            def plan_sample(self, _):
                raise AssertionError("recipe execution must not plan")

            def materialize_recipe_record(self, record):
                self.seen.append(record.logical_index)
                return SimpleNamespace(metadata={}), True

        dataset = Dataset()
        lookahead = loader.MixedStepLookahead.__new__(loader.MixedStepLookahead)
        lookahead.schedule = SimpleNamespace(recipe_step=lambda step: records)
        lookahead.datasets = {"diegesis": dataset}
        lookahead.rank = 1
        lookahead.recipe_position = 0
        lookahead._next_cursors = {}
        lookahead.gradient_diagnostics_interval = 0
        with (
            ThreadPoolExecutor(max_workers=2) as executor,
            patch.object(loader, "_pin_sample", side_effect=lambda sample: sample),
            patch.object(loader, "_sample_nbytes", return_value=1),
        ):
            prepared = lookahead._prepare_recipe_step(executor)

        self.assertEqual(dataset.seen, [0, 1])
        self.assertEqual(prepared.recipe_step, 0)
        self.assertEqual(prepared.planning_seconds, 0.0)
        self.assertEqual(len(prepared.groups), 2)

    def test_recipe_lookahead_keeps_two_steps_in_flight(self):
        from mvtracker.datasets import mixed_physical_loader as loader

        second_step_started = threading.Event()

        def record(step):
            return SimpleNamespace(
                source="diegesis",
                logical_index=step,
                physical=SimpleNamespace(rank=0, group=0, position=0),
                recipe=SimpleNamespace(
                    scene=f"scene-{step}",
                    request={"virtual_index": step},
                    retry_count=0,
                ),
            )

        class Dataset:
            def materialize_recipe_record(self, execution_record):
                if execution_record.logical_index == 0:
                    if not second_step_started.wait(timeout=2):
                        raise RuntimeError("second recipe step never started")
                else:
                    second_step_started.set()
                return SimpleNamespace(metadata={}), True

        schedule = SimpleNamespace(
            recipe_step=lambda step: (record(step),),
        )
        with (
            patch.object(loader, "_pin_sample", side_effect=lambda sample: sample),
            patch.object(loader, "_sample_nbytes", return_value=1),
        ):
            lookahead = loader.MixedStepLookahead(
                datasets={"diegesis": Dataset()},
                schedule=schedule,
                source_cursors={},
                rank=0,
                remaining_steps=2,
                worker_count=2,
                lookahead_steps=2,
                max_cache_bytes=1024,
            )
            prepared = tuple(lookahead)

        self.assertEqual([step.recipe_step for step in prepared], [0, 1])

    def test_failed_materialization_reaches_step_boundary(self):
        from mvtracker.datasets import mixed_physical_loader as loader

        failed = loader.PreparedMixedStep(
            start_cursors={},
            end_cursors={},
            groups=(),
            fingerprint="",
            retry_count=0,
            encoded_bytes=0,
            planning_seconds=0.0,
            materialization_seconds=0.0,
            pair_count=0,
            padding_tracks=0,
            recipe_step=3,
            materialization_error="failed locally",
        )
        decoder = SimpleNamespace(device=torch.device("cpu"))
        with patch.object(torch.cuda, "set_device"):
            pipeline = loader.PhysicalStepPrefetchIterator([failed], decoder)
            observed, groups = pipeline.next_step()

        self.assertIs(observed, failed)
        self.assertEqual(tuple(groups), ())


if __name__ == "__main__":
    unittest.main()
