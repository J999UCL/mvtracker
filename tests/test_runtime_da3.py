import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mvtracker.preprocessing import runtime_da3


ROOT = Path(__file__).resolve().parents[1]


def _record(ordinal: int):
    return SimpleNamespace(
        depth_source="estimated",
        source="diegesis",
        scene=f"scene-{ordinal}",
        step=ordinal // 8,
        logical_index=ordinal % 8,
    )


class _Model:
    def to(self, device):
        return self

    def eval(self):
        return self


class RuntimeDa3PrefillTests(unittest.TestCase):
    def test_exact_64_prefill_partition_is_13_13_13_13_12(self):
        counts = [0] * 5
        for ordinal in range(64):
            counts[runtime_da3._prefill_owner(ordinal, 5)] += 1
        self.assertEqual(counts, [13, 13, 13, 13, 12])
        self.assertEqual(
            {
                ordinal
                for worker_id in range(5)
                for ordinal in range(64)
                if runtime_da3._prefill_owner(ordinal, 5) == worker_id
            },
            set(range(64)),
        )

    def test_depth_records_skip_gt_without_changing_canonical_order(self):
        steps = [
            {
                "logical_samples": [
                    {"name": "zero", "depth_source": "gt"},
                    {"name": "one", "depth_source": "estimated"},
                ]
            },
            {
                "logical_samples": [
                    {"name": "two", "depth_source": "cleaned"},
                    {"name": "three", "depth_source": "gt"},
                ]
            },
        ]
        reader = SimpleNamespace(steps=lambda: iter(steps))
        with (
            patch.object(runtime_da3, "RecipeReader", return_value=reader),
            patch.object(
                runtime_da3.RecipeRecord,
                "from_dict",
                side_effect=lambda value: SimpleNamespace(**value),
            ),
        ):
            records = list(runtime_da3._depth_records(Path("recipe")))
        self.assertEqual(
            [(ordinal, record.name) for ordinal, record in records],
            [(0, "one"), (1, "two")],
        )

    def test_depth_records_stop_at_requested_training_prefix(self):
        steps = [
            {
                "step": step,
                "logical_samples": [
                    {"name": f"sample-{step}", "depth_source": "estimated"}
                ],
            }
            for step in range(3)
        ]
        reader = SimpleNamespace(steps=lambda: iter(steps))
        with (
            patch.object(runtime_da3, "RecipeReader", return_value=reader),
            patch.object(
                runtime_da3.RecipeRecord,
                "from_dict",
                side_effect=lambda value: SimpleNamespace(**value),
            ),
        ):
            records = list(runtime_da3._depth_records(Path("recipe"), max_steps=2))
        self.assertEqual([record.name for _, record in records], ["sample-0", "sample-1"])

    def test_expected_prefill_paths_are_recipe_exact(self):
        records = [_record(ordinal) for ordinal in range(4)]
        with patch.object(
            runtime_da3,
            "_depth_records",
            return_value=enumerate(records),
        ):
            paths = runtime_da3.prefill_ready_paths(
                Path("recipe"), Path("runtime"), 3
            )
        self.assertEqual(
            paths,
            (
                Path("runtime/step-000000/sample-00/ready"),
                Path("runtime/step-000000/sample-01/ready"),
                Path("runtime/step-000000/sample-02/ready"),
            ),
        )

    def test_prefill_workers_only_generate_their_modulo_shard(self):
        produced = []
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            self._run_worker(
                output_root,
                worker_id=0,
                continue_after_prefill=False,
                produced=produced,
            )
            self.assertEqual(produced, list(range(0, 64, 5)))
            self.assertTrue((output_root / "worker-0.prefill.ready").is_file())
            self.assertTrue((output_root / "worker-0.complete").is_file())
            self.assertFalse((output_root / "prefill.ready").exists())

    def test_worker_four_waits_for_handoff_then_owns_all_later_records(self):
        produced = []
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            (output_root / "prefill.ready").touch()
            self._run_worker(
                output_root,
                worker_id=4,
                continue_after_prefill=True,
                produced=produced,
            )
            self.assertEqual(produced, [*range(4, 64, 5), 64, 65])
            self.assertTrue((output_root / "complete").is_file())

    def test_five_gpu_launcher_reaps_burst_workers_before_handoff(self):
        source = (ROOT / "tools/modal_continual_training.py").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            source.count("prefill_devices=(0, 1, 2, 3, 4)"),
            3,
        )
        recipe_launcher = source[
            source.index("def _coordinate_recipe_da3(") : source.index(
                "def _run_recipe_da3("
            )
        ]
        wait_for_workers = recipe_launcher.index(
            "for worker_id, (process, thread) in enumerate(workers):"
        )
        publish_handoff = recipe_launcher.index("    prefill.touch()", wait_for_workers)
        launch_training = recipe_launcher.index(
            "    training_environment =", publish_handoff
        )
        self.assertLess(wait_for_workers, publish_handoff)
        self.assertLess(publish_handoff, launch_training)
        self.assertIn("if ready_samples != expected_ready_samples:", source)
        self.assertIn('"prefill_workers": prefill_worker_metrics', source)
        self.assertIn('"event": "da3_prefill_complete"', source)
        self.assertIn("finally:\n        _terminate_logged_processes(workers)", source)
        for name in ("OMP", "MKL", "OPENBLAS", "NUMEXPR"):
            self.assertIn(f'"{name}_NUM_THREADS": "1"', source)

    def test_steady_producer_prefetches_reads_and_h200_uses_capacity_80(self):
        producer = (ROOT / "mvtracker/preprocessing/runtime_da3.py").read_text(
            encoding="utf-8"
        )
        launcher = (ROOT / "tools/modal_continual_training.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("ThreadPoolExecutor(max_workers=1) as loader", producer)
        self.assertIn("loaded_future = loader.submit(load_next)", producer)
        self.assertIn("loaded=loaded", producer)
        self.assertEqual(launcher.count("da3_image_capacity=80"), 3)
        self.assertEqual(launcher.count("da3_image_capacity=64"), 1)

    def _run_worker(
        self,
        output_root: Path,
        *,
        worker_id: int,
        continue_after_prefill: bool,
        produced: list[int],
    ) -> None:
        records = [_record(ordinal) for ordinal in range(66)]
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(
            synchronize=lambda: None,
            get_device_name=lambda index: "test-gpu",
        )
        fake_api = types.ModuleType("depth_anything_3.api")
        fake_api.DepthAnything3 = SimpleNamespace(from_pretrained=lambda model: _Model())
        fake_package = types.ModuleType("depth_anything_3")
        fake_package.api = fake_api

        def produce(model, reader, record, root, *, loaded=None):
            ordinal = int(record.scene.removeprefix("scene-"))
            produced.append(ordinal)
            sample_root = (
                root
                / f"step-{record.step:06d}"
                / f"sample-{record.logical_index:02d}"
            )
            sample_root.mkdir(parents=True)
            (sample_root / "ready").touch()
            return 1.0, 1

        with (
            patch.dict(
                sys.modules,
                {
                    "torch": fake_torch,
                    "depth_anything_3": fake_package,
                    "depth_anything_3.api": fake_api,
                },
            ),
            patch.object(runtime_da3, "_depth_records", return_value=enumerate(records)),
            patch.object(
                runtime_da3,
                "_PackedScenes",
                return_value=SimpleNamespace(load=lambda record: record.scene),
            ),
            patch.object(
                runtime_da3,
                "_MVKubricScenes",
                return_value=SimpleNamespace(load=lambda record: record.scene),
            ),
            patch.object(runtime_da3, "_produce_record", side_effect=produce),
            patch.object(runtime_da3, "_wait_for_consumer"),
            patch.object(runtime_da3, "_log"),
        ):
            runtime_da3.run(
                Path("recipe"),
                Path("data"),
                output_root,
                64,
                worker_id=worker_id,
                worker_count=5,
                prefill_samples=64,
                continue_after_prefill=continue_after_prefill,
            )


if __name__ == "__main__":
    unittest.main()
