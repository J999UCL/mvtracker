from pathlib import Path
import tempfile
import unittest

from mvtracker.profiling.t4_loader_benchmark import (
    CASES,
    SOURCE_SCHEDULE,
    ContainerHardwareMonitor,
    run_case_matrix,
)


ROOT = Path(__file__).resolve().parents[1]


class T4LoaderBenchmarkTests(unittest.TestCase):
    def test_case_matrix_and_alternating_schedule_are_fixed(self):
        calls = []

        def profile(**kwargs):
            calls.append(kwargs)
            sources = kwargs.get("source_schedule", (kwargs["source"],))
            return {
                "view_count": kwargs["view_count"],
                "source_schedule": list(sources),
                "samples_per_second": 1.0,
                "sample_seconds_median": 1.0,
                "sample_seconds_p95": 1.0,
                "exposed_wait_seconds_p50": 0.1,
                "exposed_wait_seconds_p95": 0.2,
                "max_exposed_wait_seconds": 0.3,
            }

        result = run_case_matrix(profile, warmup=4, measured=2)

        self.assertEqual(
            [case.name for case in CASES],
            ["diegesis-views1", "diegesis-views2", "diegesis-views4",
             "mvkubric-views1", "mvkubric-views2", "mvkubric-views4", "mvkubric-views6"],
        )
        self.assertEqual(result["source_schedule"], list(SOURCE_SCHEDULE))
        self.assertEqual(result["alternating_schedule_label"], "representative-fixed-view4")
        self.assertEqual(len(calls), 16)
        self.assertEqual(calls[-1]["source_schedule"], SOURCE_SCHEDULE)
        self.assertEqual(result["cases"]["mvkubric-views6"]["cold"]["view_count"], 6)

    def test_container_monitor_reports_cpu_and_ram(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cpuacct").mkdir()
            (root / "cpu").mkdir()
            (root / "memory").mkdir()
            (root / "cpuacct/cpuacct.usage").write_text("1000000000")
            (root / "cpu/cpu.cfs_quota_us").write_text("200000")
            (root / "cpu/cpu.cfs_period_us").write_text("100000")
            (root / "memory/memory.usage_in_bytes").write_text(str(2 * 1024**3))
            (root / "memory/memory.limit_in_bytes").write_text(str(4 * 1024**3))
            monitor = ContainerHardwareMonitor(root)
            (root / "cpuacct/cpuacct.usage").write_text("3000000000")
            metrics = monitor.sample()

        self.assertGreater(metrics["cpu_cores_used"], 0.0)
        self.assertEqual(metrics["ram_used_gib"], 2.0)
        self.assertEqual(metrics["ram_limit_gib"], 4.0)

    def test_modal_function_has_one_t4_and_billing_tags(self):
        source = (ROOT / "tools/modal_t4_loader_benchmark.py").read_text(encoding="utf-8")
        self.assertIn('gpu=T4_GPU_REQUEST', source)
        self.assertIn('T4_GPU_REQUEST = "T4"', (ROOT / "mvtracker/profiling/t4_loader_benchmark.py").read_text())
        self.assertIn("BASE_TAGS", source)
        self.assertIn("run_volume.commit()", source)
        self.assertIn("wandb.init", source)


if __name__ == "__main__":
    unittest.main()
