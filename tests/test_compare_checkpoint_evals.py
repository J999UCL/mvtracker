import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare_checkpoint_evals.py"
SPEC = importlib.util.spec_from_file_location("compare_checkpoint_evals", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompareCheckpointEvalsTest(unittest.TestCase):
    def test_bootstrap_is_reproducible(self):
        values = np.array([-1.0, 0.0, 1.0, 2.0])
        first = MODULE.paired_bootstrap_ci(values, resamples=500, seed=42)
        second = MODULE.paired_bootstrap_ci(values, resamples=500, seed=42)
        self.assertEqual(first, second)

    def test_non_regression_decisions(self):
        self.assertEqual(MODULE.classify_non_regression(-0.2, 0.8, 1.0), "non-regressed")
        self.assertEqual(MODULE.classify_non_regression(0.8, 1.2, 1.0), "inconclusive")
        self.assertEqual(MODULE.classify_non_regression(1.1, 1.4, 1.0), "regressed")

    def test_checkpoint_comparison_uses_paired_metric_directions(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            original_root = root / "original"
            candidate_root = root / "candidate"
            for dataset_name, dataset in MODULE.DATASETS.items():
                columns = {
                    f"eval_{dataset_name}/model__{metric}__{dataset['point_type']}": [10.0, 12.0]
                    for metric in MODULE.METRICS
                }
                for run_root, offset in ((original_root, 0.0), (candidate_root, 0.5)):
                    dataset_dir = run_root / f"eval_{dataset_name}"
                    dataset_dir.mkdir(parents=True)
                    frame = pd.DataFrame(
                        {column: np.asarray(values) + offset for column, values in columns.items()}
                    )
                    frame.to_csv(dataset_dir / "step-0_metrics.csv")

            summary, per_sequence = MODULE.compare_regression(
                original_root,
                candidate_root,
                resamples=100,
                seed=42,
            )

        self.assertEqual(len(summary), 12)
        self.assertEqual(len(per_sequence), 24)
        by_metric = {row["metric"]: row for row in summary[:4]}
        self.assertEqual(by_metric["AJ"]["median_degradation"], -0.5)
        self.assertEqual(by_metric["MTE (cm)"]["median_degradation"], 0.5)


if __name__ == "__main__":
    unittest.main()
