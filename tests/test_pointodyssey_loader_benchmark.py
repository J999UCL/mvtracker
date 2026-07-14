import ast
import importlib.util
import io
import json
import tempfile
import unittest
from collections import Counter
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace


def _load_benchmark():
    path = Path(__file__).resolve().parents[1] / "scripts" / "pointodyssey_loader_benchmark.py"
    spec = importlib.util.spec_from_file_location("_pointodyssey_benchmark", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_collate_fn():
    path = Path(__file__).resolve().parents[1] / "mvtracker" / "datasets" / "utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "collate_fn"
    )

    class FakeTorch:
        @staticmethod
        def stack(values, dim):
            return ("stack", tuple(values), dim)

    namespace = {
        "torch": FakeTorch,
        "Datapoint": lambda **kwargs: SimpleNamespace(**kwargs),
    }
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["collate_fn"]


benchmark = _load_benchmark()
collate_fn = _load_collate_fn()


class BenchmarkTests(unittest.TestCase):
    def test_default_matrix(self):
        args = benchmark.parse_args(
            ["--dataset-root", "/tmp/data", "--output-dir", "/tmp/output"]
        )

        self.assertEqual(args.worker_counts, [0, 2, 4, 8])
        self.assertEqual(args.warmup_samples, 32)
        self.assertEqual(args.samples_per_worker, 156)
        self.assertFalse(args.skip_coverage)

    def test_coverage_can_be_skipped(self):
        args = benchmark.parse_args(
            [
                "--dataset-root",
                "/tmp/data",
                "--output-dir",
                "/tmp/output",
                "--worker-counts",
                "8",
                "--skip-coverage",
            ]
        )

        self.assertEqual(args.worker_counts, [8])
        self.assertTrue(args.skip_coverage)

    def test_duplicate_workers_are_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            benchmark.parse_args(
                [
                    "--dataset-root",
                    "/tmp/data",
                    "--output-dir",
                    "/tmp/output",
                    "--worker-counts",
                    "2",
                    "2",
                ]
            )

    def test_balanced_schedule_visits_every_scene_twice(self):
        schedule = benchmark.build_balanced_schedule(
            156,
            repeat_offset=200,
            seed=123,
        )

        self.assertEqual(len(schedule), 156)
        self.assertEqual(len(set(schedule)), 156)
        self.assertEqual(
            Counter(index % 78 for index in schedule),
            Counter({index: 2 for index in range(78)}),
        )

    def test_diversity_counts_scene_start_pairs(self):
        result = benchmark.summarize_diversity(
            [
                {"scene_name": "000000", "window_start": 3},
                {"scene_name": "000000", "window_start": 3},
                {"scene_name": "000001", "window_start": 8},
            ]
        )

        self.assertEqual(result["unique_scenes"], 2)
        self.assertEqual(result["unique_scene_start_pairs"], 2)

    def test_provenance_checks_invalid_rgb_windows(self):
        contracts = {
            "000003": {
                "frame_count": 120,
                "invalid_rgb_frame_indices": [50],
            }
        }
        metadata = {
            "virtual_index": 81,
            "scene_index": 3,
            "scene_name": "000003",
            "gotit": True,
            "window_start": 40,
            "window_end_exclusive": 64,
        }

        failures = benchmark.validate_provenance(
            metadata,
            gotit=True,
            allowed_indices={81},
            contracts=contracts,
        )

        self.assertIn("sample window intersects an invalid RGB frame", failures)

    def test_clean_lane_requires_every_requested_sample(self):
        clean = {
            "requested_samples": 10,
            "returned_samples": 10,
            "successes": 10,
            "gotit_false": 0,
            "exceptions": 0,
            "invariant_failures": 0,
            "warmup_failures": 0,
        }

        self.assertTrue(benchmark._lane_is_clean(clean))
        self.assertFalse(benchmark._lane_is_clean({**clean, "gotit_false": 1}))

    def test_atomic_json_replaces_running_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            benchmark._atomic_json(path, {"status": "running"})
            benchmark._atomic_json(path, {"status": "completed"})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"status": "completed"},
            )

    def test_sample_metadata_survives_collation(self):
        def sample(index):
            return SimpleNamespace(
                video=index,
                videodepth=index,
                segmentation=index,
                seq_name=index,
                intrs=index,
                videodepthconf=None,
                feats=None,
                trajectory=None,
                valid=None,
                visibility=None,
                trajectory_3d=None,
                extrs=None,
                query_points=None,
                query_points_3d=None,
                sample_metadata={"virtual_index": index},
                track_upscaling_factor=1.0,
                novel_video=None,
            )

        batch, gotit = collate_fn([(sample(10), True), (sample(11), False)])

        self.assertEqual(
            batch.sample_metadata,
            [{"virtual_index": 10}, {"virtual_index": 11}],
        )
        self.assertEqual(gotit, [True, False])


if __name__ == "__main__":
    unittest.main()
