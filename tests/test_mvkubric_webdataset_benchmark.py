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


if __name__ == "__main__":
    unittest.main()
