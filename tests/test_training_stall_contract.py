import ast
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
