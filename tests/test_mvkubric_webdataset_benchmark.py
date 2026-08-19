import inspect
import unittest

from mvtracker.profiling import mvkubric_webdataset_benchmark as benchmark


class MvKubricWebDatasetBenchmarkTests(unittest.TestCase):
    def test_dali_benchmark_uses_production_indexed_loader(self):
        source = inspect.getsource(benchmark.benchmark_dali_case)
        self.assertIn("DaliKubricMultiViewDataset", source)
        self.assertIn("dataset.plan_sample", source)
        self.assertIn("dataset.materialize_sample", source)
        self.assertIn("DaliEncodedImageDecoder", source)
        self.assertIn("media_record_count", source)
        self.assertNotIn("DaliKubricSceneStream", source)

    def test_matrix_keeps_requested_view_counts(self):
        source = inspect.getsource(benchmark.benchmark_matrix)
        self.assertIn("(1, 2, 4, 6)", source)
        self.assertIn("event=case_start", source)

    def test_modal_benchmark_runs_real_dali_parity_preflight(self):
        source = (
            __import__("pathlib").Path(__file__).parents[1]
            / "tools/modal_mvkubric_webdataset.py"
        ).read_text(encoding="utf-8")
        self.assertIn("from tools.check_dali_decode import check_pair", source)
        self.assertIn("parity_root = DATA_ROOT /", source)
        self.assertIn("run.summary.update({\"dali_parity\": dali_parity})", source)


if __name__ == "__main__":
    unittest.main()
