import ast
import tempfile
import unittest
from pathlib import Path

from mvtracker.datasets.mixed_source_schedule import MixedSourceSample, ScheduledSampleRequest
from mvtracker.datasets.training_recipe import PhysicalAssignment, RecipeRecord, RecipeWriter


ROOT = Path(__file__).resolve().parents[1]


def _recipe_schedule_class():
    tree = ast.parse((ROOT / "mvtracker/cli/train.py").read_text(encoding="utf-8"))
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef) and item.name == "_RecipeMixedSourceSchedule"
    )
    namespace = {
        "MixedSourceSample": MixedSourceSample,
        "ScheduledSampleRequest": ScheduledSampleRequest,
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), "train.py", "exec"), namespace)
    return namespace["_RecipeMixedSourceSchedule"]


class TrainingRecipeConsumptionContractTests(unittest.TestCase):
    def test_default_config_requires_explicit_recipe_and_gt_override(self):
        source = (ROOT / "configs/train.yaml").read_text(encoding="utf-8")
        self.assertIn("recipe_path: null", source)
        self.assertIn("force_gt_depth: false", source)

    def test_smoke_is_twenty_steps_gt_only_and_has_no_validation(self):
        source = (
            ROOT
            / "configs/experiment/diegesis_syn4d_mvkubric_recipe_gt_ddp_smoke20.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("num_steps: 20", source)
        self.assertIn("force_gt_depth: true", source)
        self.assertIn("names: []", source)
        self.assertIn("hardware_metrics_interval: 1", source)

    def test_recipe_replay_is_checkpointed_and_depth_is_visible(self):
        source = (ROOT / "mvtracker/cli/train.py").read_text(encoding="utf-8")
        self.assertIn("RecipeReader(recipe_path)", source)
        self.assertIn("record.replay_request(ScheduledSampleRequest)", source)
        self.assertIn('state.recipe_position = int(recipe_position)', source)
        self.assertIn('metadata["planned_depth_source"]', source)
        self.assertIn('metadata["effective_depth_source"]', source)
        self.assertIn('depth_source="gt" if self.force_gt_depth', source)

    def test_replay_uses_logical_ordinals_when_actual_cursors_have_gaps(self):
        with tempfile.TemporaryDirectory() as temporary:
            recipe_dir = Path(temporary) / "recipe"
            writer = RecipeWriter(
                recipe_dir,
                manifest={},
                world_size=1,
                step_count=1,
                records_per_step=2,
            )
            for microbatch, actual_cursor in enumerate((3, 8)):
                writer.write(
                    RecipeRecord(
                        step=0,
                        microbatch=microbatch,
                        rank=0,
                        scheduled_rank=0,
                        source="diegesis",
                        source_cursor=actual_cursor,
                        retry_count=actual_cursor,
                        request={
                            "virtual_index": actual_cursor,
                            "scene_index": 0,
                            "view_count": 1,
                        },
                        seed=actual_cursor,
                        scene_index=0,
                        scene="scene",
                        frames=(0,),
                        views=(0,),
                        track_count=1,
                        tracks=(0,),
                        augmentation={},
                        depth_source="estimated",
                        physical=PhysicalAssignment(rank=0, group=microbatch, position=0),
                    )
                )
            writer.finalize(summary={}, estimated_depth_requests=[])

            schedule = _recipe_schedule_class()(recipe_dir, rank=0, world_size=1)
            first = schedule.sample_source("diegesis", 0, 0).request
            second = schedule.sample_source("diegesis", 1, 0).request
            self.assertEqual((first.virtual_index, second.virtual_index), (3, 8))
            self.assertEqual(first.depth_source, "estimated")
            self.assertEqual(first.expected_scene, "scene")

    def test_modal_planner_and_smoke_resource_contract(self):
        source = (ROOT / "tools/modal_continual_training.py").read_text(
            encoding="utf-8"
        )
        planner_start = source.index("def plan_recipe_remote(")
        planner_decorator = source[source.rfind("@app.function(", 0, planner_start):planner_start]
        smoke_start = source.index("def recipe_smoke20_remote(")
        smoke_decorator = source[source.rfind("@app.function(", 0, smoke_start):smoke_start]
        self.assertIn("cpu=16", planner_decorator)
        self.assertIn("data_volume.with_mount_options(read_only=True)", planner_decorator)
        self.assertIn("str(RUN_ROOT): run_volume", planner_decorator)
        self.assertIn('RECIPE_SMOKE_GPU_REQUEST = "H100:2"', source)
        self.assertIn("gpu=RECIPE_SMOKE_GPU_REQUEST", smoke_decorator)
        self.assertIn('heartbeat_seconds=10', source)
        self.assertIn('"purpose": "training"', (
            ROOT / "mvtracker/profiling/modal_continual_training.py"
        ).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
