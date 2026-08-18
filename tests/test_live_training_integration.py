"""Focused contracts for live planned physical batching."""

import ast
import unittest
from pathlib import Path

import torch

TRAIN_PATH = Path(__file__).resolve().parents[1] / "mvtracker" / "cli" / "train.py"
TRAIN_TREE = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"))
LOADER_PATH = (
    Path(__file__).resolve().parents[1]
    / "mvtracker"
    / "datasets"
    / "mixed_physical_loader.py"
)
LOADER_TREE = ast.parse(LOADER_PATH.read_text(encoding="utf-8"))


def _function(name):
    return next(
        node
        for node in TRAIN_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


class LivePhysicalTrainingTests(unittest.TestCase):
    def test_physical_loss_weights_scene_count(self):
        node = _function("_scale_physical_batch_loss")
        namespace = {}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                str(TRAIN_PATH),
                "exec",
            ),
            namespace,
        )
        scale = namespace["_scale_physical_batch_loss"]
        self.assertEqual(scale(torch.tensor(8.0), 1, 4).item(), 2.0)
        self.assertEqual(scale(torch.tensor(8.0), 2, 4).item(), 4.0)

    def test_live_loop_consumes_cpu_lookahead_lazily(self):
        source = ast.unparse(_function("main"))
        self.assertIn("MixedStepLookahead", source)
        self.assertIn("PhysicalGroupPrefetchIterator", source)
        self.assertIn("source_cursors = dict(physical_step.end_cursors)", source)
        self.assertNotIn("torch.cuda.synchronize", source)

    def test_cpu_preparation_has_no_cuda_wait(self):
        lookahead = next(
            node
            for node in LOADER_TREE.body
            if isinstance(node, ast.ClassDef) and node.name == "MixedStepLookahead"
        )
        prepare = next(
            node
            for node in lookahead.body
            if isinstance(node, ast.FunctionDef) and node.name == "_prepare_step"
        )
        source = ast.unparse(prepare)
        self.assertNotIn("cuda", source)
        self.assertNotIn("wait_event", source)

    def test_track_padding_marks_added_slots_unusable(self):
        pad_node = next(
            node
            for node in LOADER_TREE.body
            if isinstance(node, ast.FunctionDef) and node.name == "_pad_tensor"
        )
        namespace = {"torch": torch}
        exec(
            compile(
                ast.fix_missing_locations(
                    ast.Module(body=[pad_node], type_ignores=[])
                ),
                str(LOADER_PATH),
                "exec",
            ),
            namespace,
        )
        padded_mask = namespace["_pad_tensor"](
            torch.tensor([[False, False]]), -1, 4, fill_value=True
        )
        torch.testing.assert_close(
            padded_mask, torch.tensor([[False, False, True, True]])
        )
        merge_node = next(
            node
            for node in LOADER_TREE.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "merge_decoded_datapoints"
        )
        self.assertIn(
            "fill_value=field.name == 'track_padding_mask'",
            ast.unparse(merge_node),
        )


if __name__ == "__main__":
    unittest.main()
