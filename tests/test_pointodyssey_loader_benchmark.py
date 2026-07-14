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


def _load_benchmark_module():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "pointodyssey_loader_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_pointodyssey_loader_benchmark_under_test",
        source_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark_module()


def _load_collate_fn():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "mvtracker"
        / "datasets"
        / "utils.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
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
    exec(compile(ast.fix_missing_locations(module), str(source_path), "exec"), namespace)
    return namespace["collate_fn"]


collate_fn = _load_collate_fn()


class CommandLineTests(unittest.TestCase):
    def test_defaults_use_requested_worker_matrix(self):
        args = benchmark.parse_args(
            [
                "--dataset-root",
                "/tmp/data",
                "--output-dir",
                "/tmp/output",
            ]
        )

        self.assertEqual(args.worker_counts, (0, 2, 4, 8))
        self.assertEqual(args.warmup_samples, 32)
        self.assertEqual(args.measured_scene_repeats, 2)
        self.assertEqual(args.confirmation_warmup, 32)
        self.assertEqual(args.confirmation_samples, 256)
        self.assertEqual(args.child_timeout_seconds, 7200.0)

    def test_duplicate_worker_counts_are_rejected(self):
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
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


class CollationTests(unittest.TestCase):
    def test_sample_metadata_survives_collation_as_one_item_per_sample(self):
        def sample(value):
            return SimpleNamespace(
                video=f"video-{value}",
                videodepth=f"depth-{value}",
                segmentation=f"segmentation-{value}",
                seq_name=f"scene-{value}",
                intrs=f"intrinsics-{value}",
                videodepthconf=None,
                feats=None,
                trajectory=None,
                valid=None,
                visibility=None,
                trajectory_3d=None,
                extrs=None,
                query_points=None,
                query_points_3d=None,
                sample_metadata={"virtual_index": value},
                track_upscaling_factor=1.0,
                novel_video=None,
            )

        batch, gotit = collate_fn([(sample(10), True), (sample(11), False)])

        self.assertEqual(
            batch.sample_metadata,
            [{"virtual_index": 10}, {"virtual_index": 11}],
        )
        self.assertEqual(gotit, [True, False])


class ScheduleTests(unittest.TestCase):
    def test_balanced_schedule_uses_each_scene_twice(self):
        schedule = benchmark.build_balanced_schedule(
            78,
            156,
            repeat_offset=200,
            seed=123,
        )

        self.assertEqual(len(schedule), 156)
        self.assertEqual(len(set(schedule)), 156)
        self.assertEqual(Counter(index % 78 for index in schedule), Counter({index: 2 for index in range(78)}))

    def test_schedule_is_reproducible(self):
        first = benchmark.build_balanced_schedule(5, 13, repeat_offset=7, seed=99)
        second = benchmark.build_balanced_schedule(5, 13, repeat_offset=7, seed=99)

        self.assertEqual(first, second)


class MetricTests(unittest.TestCase):
    def test_percentile_uses_linear_interpolation(self):
        self.assertEqual(benchmark.percentile([], 0.5), None)
        self.assertEqual(benchmark.percentile([1.0, 3.0], 0.5), 2.0)
        self.assertEqual(benchmark.percentile([1.0, 2.0, 3.0, 4.0], 0.95), 3.8499999999999996)

    def test_temporal_diversity_counts_collisions_per_scene(self):
        result = benchmark.summarize_temporal_diversity(
            [
                {"scene_name": "000000", "window_start": 3},
                {"scene_name": "000000", "window_start": 3},
                {"scene_name": "000000", "window_start": 8},
                {"scene_name": "000001", "window_start": 1},
            ]
        )

        self.assertEqual(result["observations"], 4)
        self.assertEqual(result["unique_scenes"], 2)
        self.assertEqual(result["unique_scene_start_pairs"], 3)
        self.assertEqual(result["repeated_scene_start_observations"], 1)
        self.assertEqual(result["starts_by_scene"]["000000"], [3, 3, 8])

    def test_best_trial_requires_a_complete_clean_lane(self):
        trials = [
            {
                "status": "completed",
                "workers": 2,
                "warmup_gotit_false": 0,
                "warmup_invariant_failures": 0,
                "gotit_false": 0,
                "exception_failures": 0,
                "invariant_failures": 0,
                "attempted_measured_samples": 10,
                "returned_measured_samples": 10,
                "requested_measured_samples": 10,
                "successes": 10,
                "successful_samples_per_second": 3.0,
            },
            {
                "status": "completed",
                "workers": 4,
                "warmup_gotit_false": 0,
                "warmup_invariant_failures": 0,
                "gotit_false": 1,
                "exception_failures": 0,
                "invariant_failures": 0,
                "attempted_measured_samples": 10,
                "returned_measured_samples": 10,
                "requested_measured_samples": 10,
                "successes": 9,
                "successful_samples_per_second": 5.0,
            },
        ]

        self.assertEqual(benchmark._best_trial(trials)["workers"], 2)

    def test_all_trials_must_be_clean_and_in_requested_order(self):
        clean = {
            "status": "completed",
            "warmup_gotit_false": 0,
            "warmup_invariant_failures": 0,
            "gotit_false": 0,
            "exception_failures": 0,
            "invariant_failures": 0,
            "attempted_measured_samples": 10,
            "returned_measured_samples": 10,
            "requested_measured_samples": 10,
            "successes": 10,
            "in_order": False,
        }
        trials = [{**clean, "workers": workers} for workers in (0, 2, 4, 8)]

        self.assertTrue(
            benchmark._all_trials_clean(
                trials,
                (0, 2, 4, 8),
                expected_in_order=False,
            )
        )
        self.assertFalse(
            benchmark._all_trials_clean(
                trials[:-1],
                (0, 2, 4, 8),
                expected_in_order=False,
            )
        )
        self.assertFalse(
            benchmark._all_trials_clean(
                trials,
                (2, 0, 4, 8),
                expected_in_order=False,
            )
        )
        self.assertFalse(
            benchmark._all_trials_clean(
                trials,
                (0, 2, 4, 8),
                expected_in_order=True,
            )
        )


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.contracts = {
            "000003": {
                "frame_count": 120,
                "invalid_rgb_frame_indices": [50],
                "decoded_camera_frames_per_sample": 480,
            }
        }
        self.metadata = {
            "virtual_index": 81,
            "scene_index": 3,
            "scene_name": "000003",
            "seed": 153,
            "window_start": 10,
            "window_end_exclusive": 34,
            "selected_views": [3, 1, 0, 2],
            "gotit": True,
        }

    def test_valid_metadata_matches_virtual_scene_contract(self):
        failures = benchmark._validate_provenance(
            self.metadata,
            expected_gotit=True,
            expected_virtual_index=81,
            allowed_virtual_indices={81},
            scene_contracts=self.contracts,
            real_len=78,
            seed_base=72,
        )

        self.assertEqual(failures, [])

    def test_invalid_rgb_window_is_rejected(self):
        metadata = {
            **self.metadata,
            "window_start": 40,
            "window_end_exclusive": 64,
        }

        failures = benchmark._validate_provenance(
            metadata,
            expected_gotit=True,
            expected_virtual_index=81,
            allowed_virtual_indices={81},
            scene_contracts=self.contracts,
            real_len=78,
            seed_base=72,
        )

        self.assertTrue(any("invalid RGB frames [50]" in failure for failure in failures))

    def test_controlled_lane_rejects_out_of_order_return(self):
        failures = benchmark._validate_provenance(
            self.metadata,
            expected_gotit=True,
            expected_virtual_index=82,
            allowed_virtual_indices={81, 82},
            scene_contracts=self.contracts,
            real_len=78,
            seed_base=72,
        )

        self.assertTrue(any("!= requested 82" in failure for failure in failures))

    def test_loader_and_metadata_gotit_must_match_in_both_directions(self):
        metadata_false = {"virtual_index": 81, "scene_index": 3, "scene_name": "000003", "gotit": False}
        false_metadata_failures = benchmark._validate_provenance(
            metadata_false,
            expected_gotit=True,
            expected_virtual_index=81,
            allowed_virtual_indices={81},
            scene_contracts=self.contracts,
            real_len=78,
            seed_base=72,
        )
        true_metadata_failures = benchmark._validate_provenance(
            self.metadata,
            expected_gotit=False,
            expected_virtual_index=81,
            allowed_virtual_indices={81},
            scene_contracts=self.contracts,
            real_len=78,
            seed_base=72,
        )

        self.assertTrue(any("metadata gotit=False" in failure for failure in false_metadata_failures))
        self.assertTrue(any("metadata gotit=True" in failure for failure in true_metadata_failures))


class ReportTests(unittest.TestCase):
    def test_atomic_json_writes_finite_machine_readable_report(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "summary.json"

            benchmark._atomic_json(path, {"status": "completed", "value": 1.25})

            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"status": "completed", "value": 1.25},
            )
            self.assertEqual(list(Path(temporary_directory).iterdir()), [path])

    def test_tree_fingerprint_detects_stat_or_name_changes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_file = root / "a.bin"
            first_file.write_bytes(b"abc")
            before = benchmark.dataset_tree_fingerprint(root)

            (root / "b.bin").write_bytes(b"def")
            after = benchmark.dataset_tree_fingerprint(root)

            self.assertNotEqual(before["sha256"], after["sha256"])
            self.assertEqual(after["regular_file_count"], 2)
            self.assertEqual(after["regular_file_bytes"], 6)


if __name__ == "__main__":
    unittest.main()
