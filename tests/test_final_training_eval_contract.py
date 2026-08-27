import ast
import logging
import unittest
from pathlib import Path
from typing import Optional

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _evaluate_sequence():
    tree = ast.parse(
        (ROOT / "mvtracker/evaluation/evaluator_3dpt.py").read_text(encoding="utf-8")
    )
    evaluator = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Evaluator"
    )
    method = next(
        node
        for node in evaluator.body
        if isinstance(node, ast.FunctionDef) and node.name == "evaluate_sequence"
    )
    method.decorator_list = []
    namespace = {
        "logging": logging,
        "Optional": Optional,
        "SummaryWriter": object,
    }
    exec(compile(ast.Module(body=[method], type_ignores=[]), "<ast>", "exec"), namespace)
    return namespace["evaluate_sequence"]


class FinalTrainingEvalContractTests(unittest.TestCase):
    def test_five_thousand_step_recipe_starts_validation_after_training(self):
        config = yaml.safe_load(
            (
                ROOT
                / "configs/experiment/diegesis_syn4d_mvkubric_recipe_da3_ddp_5000.yaml"
            ).read_text(encoding="utf-8")
        )

        self.assertFalse(config["modes"]["validate_at_start"])
        schedule = config["datasets"]["eval"]["schedule"]
        self.assertTrue(all(0 not in entry["steps"] for entry in schedule))
        self.assertEqual(schedule[0]["steps"][0], 500)
        self.assertEqual(schedule[0]["steps"][-1], 5000)
        self.assertEqual(schedule[1]["steps"], [2500, 5000])

    def test_empty_rank_local_evaluation_returns_empty_metrics(self):
        class EmptyLoader:
            def __len__(self):
                return 0

            def __iter__(self):
                raise AssertionError("an empty rank-local loader must not be iterated")

        metrics = _evaluate_sequence()(
            object(),
            model=None,
            test_dataloader=EmptyLoader(),
            dataset_name="tapvid3d-multiview-validation",
            log_dir="unused",
        )

        self.assertEqual(metrics, {})


if __name__ == "__main__":
    unittest.main()
