"""RED contracts for the planned live mixed-training integration.

These tests deliberately describe the seam that still needs production wiring.
They do not change the existing dataset APIs or exercise CUDA.  The intended
interfaces are small helpers in ``mvtracker.cli.train``:

* ``_prepare_mixed_step_cpu`` selects/materializes four rank-local encoded
  samples without waiting on a CUDA event;
* ``_schedule_live_physical_waves`` adapts the pure physical scheduler to the
  live step;
* ``_scale_physical_batch_loss`` applies the B/4 contribution for a four-slot
  accumulation window;
* ``_run_physical_wave`` performs one forward/backward call per physical group;
* ``_aggregate_source_metrics`` preserves DIEGESIS/MV-Kubric labels; and
* ``_restore_live_step`` reconstructs the exact retry/resume cursor state.

The tests are intentionally RED until that integration is implemented.  The
existing pure scheduler, dataset planning, and sequential mixed-loader tests
remain separate and continue to test their current contracts.
"""

import ast
import unittest
from pathlib import Path

import torch


TRAIN_PATH = Path(__file__).resolve().parents[1] / "mvtracker" / "cli" / "train.py"
TRAIN_TREE = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"), filename=str(TRAIN_PATH))


def _function_source(name: str) -> str:
    for node in TRAIN_TREE.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.unparse(node)
    raise AssertionError(
        f"missing planned production interface mvtracker.cli.train.{name}"
    )


class LiveTrainingIntegrationRedTests(unittest.TestCase):
    def test_four_step_cpu_lookahead_has_no_eager_cuda_barrier(self):
        source = _function_source("_prepare_mixed_step_cpu")
        self.assertIn("lookahead", source)
        self.assertIn("encoded", source)
        self.assertNotIn("torch.cuda.synchronize", source)
        self.assertNotIn("wait_event", source)
        self.assertNotIn("events[2].synchronize", source)

    def test_scheduler_waves_are_consumed_by_live_training(self):
        source = _function_source("_schedule_live_physical_waves")
        self.assertIn("schedule_physical_batch", source)
        self.assertIn("wave.ranks", source)
        self.assertIn("fabric.global_rank", source)

    def test_physical_batch_loss_uses_b_over_four(self):
        source = _function_source("_scale_physical_batch_loss")
        self.assertIn("physical_batch_size", source)
        self.assertIn("accumulation_steps", source)

        namespace = {"torch": torch}
        node = next(
            node
            for node in TRAIN_TREE.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_scale_physical_batch_loss"
        )
        module = ast.Module(body=[node], type_ignores=[])
        exec(compile(ast.fix_missing_locations(module), str(TRAIN_PATH), "exec"), namespace)
        scale = namespace["_scale_physical_batch_loss"]
        self.assertEqual(scale(torch.tensor(8.0), 1, 4).item(), 2.0)
        self.assertEqual(scale(torch.tensor(8.0), 2, 4).item(), 4.0)

    def test_physical_wave_keeps_one_ddp_backward_per_physical_group(self):
        source = _function_source("_run_physical_wave")
        self.assertIn("physical_groups", source)
        self.assertIn("fabric.backward", source)
        self.assertEqual(source.count("fabric.backward"), 1)
        self.assertNotIn("for scene in group.scenes", source)

    def test_source_labels_survive_physical_batch_aggregation(self):
        source = _function_source("_aggregate_source_metrics")
        self.assertIn("source", source)
        self.assertIn("diegesis", source)
        self.assertIn("mvkubric", source)
        self.assertIn("sample_count", source)
        self.assertIn("loss", source)

    def test_resume_retry_reconstructs_exact_live_step(self):
        source = _function_source("_restore_live_step")
        self.assertIn("source_cursors", source)
        self.assertIn("retry", source)
        self.assertIn("mixed_schedule_state", source)
        self.assertIn("state", source)


if __name__ == "__main__":
    unittest.main()
