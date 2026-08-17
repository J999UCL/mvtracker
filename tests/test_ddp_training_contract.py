import ast
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.optim as optim


TRAIN_PATH = Path(__file__).resolve().parents[1] / "mvtracker" / "cli" / "train.py"
TRAIN_TREE = ast.parse(TRAIN_PATH.read_text(encoding="utf-8"), filename=str(TRAIN_PATH))


def _function_node(name):
    return next(
        node
        for node in TRAIN_TREE.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _class_node(name):
    return next(
        node
        for node in TRAIN_TREE.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _load_functions(*names, namespace=None):
    nodes = [_function_node(name) for name in names]
    module = ast.Module(body=nodes, type_ignores=[])
    values = {} if namespace is None else dict(namespace)
    exec(compile(ast.fix_missing_locations(module), str(TRAIN_PATH), "exec"), values)
    return [values[name] for name in names]


class DDPTrainingContractTests(unittest.TestCase):
    def test_container_metrics_use_cgroup_cpu_and_memory(self):
        ticks = iter((10.0, 12.0))
        module = ast.Module(
            body=[_class_node("_ContainerHardwareMonitor")], type_ignores=[]
        )
        namespace = {
            "Path": Path,
            "time": SimpleNamespace(monotonic=lambda: next(ticks)),
            "os": SimpleNamespace(sched_getaffinity=lambda _pid: range(8)),
        }
        exec(compile(ast.fix_missing_locations(module), str(TRAIN_PATH), "exec"), namespace)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cpuacct").mkdir()
            (root / "cpu").mkdir()
            (root / "memory").mkdir()
            (root / "cpuacct/cpuacct.usage").write_text("1000000000\n")
            (root / "cpu/cpu.cfs_quota_us").write_text("400000\n")
            (root / "cpu/cpu.cfs_period_us").write_text("100000\n")
            (root / "memory/memory.usage_in_bytes").write_text(str(8 * 1024 ** 3))
            (root / "memory/memory.limit_in_bytes").write_text(str(16 * 1024 ** 3))
            monitor = namespace["_ContainerHardwareMonitor"](root)
            (root / "cpuacct/cpuacct.usage").write_text("5000000000\n")

            metrics = monitor.sample()

        self.assertEqual(metrics["hardware/container/cpu_cores_used"], 2.0)
        self.assertEqual(metrics["hardware/container/cpu_utilization_percent"], 50.0)
        self.assertEqual(metrics["hardware/container/memory_used_gib"], 8.0)
        self.assertEqual(metrics["hardware/container/memory_utilization_percent"], 50.0)

    def test_throughput_uses_global_work_and_step_wall_time(self):
        (throughput_metrics,) = _load_functions("_throughput_metrics")

        metrics = throughput_metrics(4.0, 8, 4096)

        self.assertEqual(metrics["performance/global_samples"], 8)
        self.assertEqual(metrics["performance/global_trajectories"], 4096)
        self.assertEqual(metrics["performance/samples_per_second"], 2.0)
        self.assertEqual(metrics["performance/trajectories_per_second"], 1024.0)

    def test_gpu_metrics_keep_both_ddp_ranks_separate(self):
        (gather_rank_metrics,) = _load_functions(
            "_gather_rank_metrics", namespace={"torch": torch}
        )

        class Fabric:
            device = torch.device("cpu")
            world_size = 2

            @staticmethod
            def all_gather(_local):
                return torch.tensor([[75.0, 20.0], [62.0, 18.0]])

        metrics = gather_rank_metrics(
            Fabric(), {"utilization_percent": 0.0, "memory_used_gib": 0.0}
        )

        self.assertEqual(metrics["hardware/gpu_0/utilization_percent"], 20.0)
        self.assertEqual(metrics["hardware/gpu_1/utilization_percent"], 18.0)
        self.assertEqual(metrics["hardware/gpu_0/memory_used_gib"], 75.0)
        self.assertEqual(metrics["hardware/gpu_1/memory_used_gib"], 62.0)

    def test_onecycle_uses_independent_schedule_horizon(self):
        (fetch_optimizer,) = _load_functions(
            "fetch_optimizer",
            namespace={"optim": optim, "torch": torch},
        )
        cfg = SimpleNamespace(
            lr=1e-4,
            wdecay=1e-5,
            anneal_strategy="linear",
            num_steps=1000,
            lr_schedule_steps=2000,
        )
        model = torch.nn.Linear(2, 1)

        optimizer, scheduler = fetch_optimizer(cfg, model)

        self.assertEqual(scheduler.total_steps, 2000)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 4e-6)

    def test_onecycle_rejects_schedule_shorter_than_stop(self):
        (fetch_optimizer,) = _load_functions(
            "fetch_optimizer",
            namespace={"optim": optim, "torch": torch},
        )
        cfg = SimpleNamespace(
            lr=1e-4,
            wdecay=1e-5,
            anneal_strategy="linear",
            num_steps=1000,
            lr_schedule_steps=999,
        )
        with self.assertRaisesRegex(ValueError, "lr_schedule_steps"):
            fetch_optimizer(cfg, torch.nn.Linear(2, 1))

    def test_latest_checkpoint_manifest_is_canonical(self):
        latest_path, write_manifest = _load_functions(
            "_latest_checkpoint_path",
            "_write_latest_checkpoint_manifest",
            namespace={
                "Path": Path,
                "json": json,
                "LATEST_CHECKPOINT_MANIFEST": "latest_checkpoint.json",
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model_000250.pth"
            checkpoint.touch()
            write_manifest(directory, checkpoint, 250)

            self.assertEqual(latest_path(directory), checkpoint)
            manifest = json.loads(
                (Path(directory) / "latest_checkpoint.json").read_text()
            )
            self.assertEqual(manifest["completed_steps"], 250)

    def test_rank_zero_eval_wraps_validation_in_barriers(self):
        source = ast.unparse(_function_node("_run_rank_zero_eval"))
        self.assertEqual(source.count("fabric.barrier()"), 2)
        self.assertIn("if fabric.global_rank == 0", source)
        self.assertIn("_unwrap_objects(model)", source)

    def test_main_uses_canonical_resume_and_rank_zero_writer(self):
        source = ast.unparse(_function_node("main"))
        self.assertIn("_latest_checkpoint_path(cfg.experiment_path)", source)
        self.assertNotIn("sorted(folder_ckpts)", source)
        self.assertIn("if fabric.global_rank == 0 else None", source)
        self.assertIn("resume='allow'", source)


if __name__ == "__main__":
    unittest.main()
