import json
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

from scripts import pointodyssey_training_dashboard as dashboard


class TrainingLogReaderTests(unittest.TestCase):
    def test_incrementally_parses_samples_failures_tracks_and_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text(
                "[INFO] batch is None: failed 1 / 1 (100.00%) batches\n"
                "[INFO] Datapoint: ['000027'] (microbatch 1/2, waited  0.40s)\n"
                "[INFO] FWD pass: num_views=4 num_frames=24 num_points=256 height=384\n",
                encoding="utf-8",
            )
            reader = dashboard.TrainingLogReader(path)
            reader.refresh()
            self.assertEqual(len(reader.samples), 1)
            self.assertEqual(reader.samples[0]["tracks"], 256)
            self.assertEqual(len(reader.failure_events), 1)

            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    "[INFO] Datapoint: ['000030'] (microbatch 2/2, waited  0.20s)\n"
                    "[INFO] FWD pass: num_views=4 num_frames=24 num_points=128 height=384\n"
                    "[INFO] [timing:000001] Total:  12.00s | Data:   3.00s | "
                    "Fwd:   4.00s | Sync:   0.00s | Bwd:   5.00s | \n"
                )
            reader.refresh()
            series = reader.pipeline_series(accumulation_steps=2)
            self.assertEqual(len(reader.samples), 2)
            self.assertEqual(len(reader.timing_rows), 1)
            self.assertEqual(series[0]["failed"], 1)
            self.assertAlmostEqual(series[0]["rejection_percent"], 100 / 3)
            self.assertEqual(series[0]["tracks_mean"], 192)
            self.assertEqual(series[0]["scenes_cumulative"], 2)

    def test_waits_for_complete_appended_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text(
                "Datapoint: ['000001'] (microbatch 1/8, waited  0.25s)",
                encoding="utf-8",
            )
            reader = dashboard.TrainingLogReader(path)
            reader.refresh()
            self.assertEqual(reader.samples, [])
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            reader.refresh()
            self.assertEqual(len(reader.samples), 1)

    def test_detects_completion_and_fatal_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text("CUDA out of memory\nFINISHED TRAINING\n", encoding="utf-8")
            reader = dashboard.TrainingLogReader(path)
            reader.refresh()
            self.assertIn("CUDA out of memory", reader.fatal_error)
            self.assertTrue(reader.finished)

    def test_computes_trailing_clipped_step_rate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.log"
            path.write_text(
                "[optimizer:000000] elements_clipped=10 clipped=1\n"
                "[optimizer:000001] elements_clipped=0 clipped=0\n"
                "[optimizer:000002] elements_clipped=2 clipped=1\n"
                "[optimizer:000003] elements_clipped=0 clipped=0\n",
                encoding="utf-8",
            )
            reader = dashboard.TrainingLogReader(path)
            reader.refresh()

            series = reader.rolling_clipped_step_rate(window_size=3)

            self.assertEqual([point["step"] for point in series], [0, 1, 2, 3])
            self.assertEqual(
                [point["value"] for point in series],
                [1.0, 0.5, 2 / 3, 1 / 3],
            )


class HydraConfigReaderTests(unittest.TestCase):
    def test_reads_dashboard_fields_from_real_hydra_yaml(self):
        try:
            import omegaconf  # noqa: F401
        except ImportError as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "trainer:\n"
                "  num_steps: 200\n"
                "  gradient_accumulation_steps: 8\n"
                "  eval_freq: 20\n"
                "datasets:\n"
                "  train:\n"
                "    traj_per_sample: 256\n"
                "    num_workers: 8\n"
                "    sequence_len: 24\n"
                "  eval:\n"
                "    names: [validation-a, validation-b]\n",
                encoding="utf-8",
            )
            config = dashboard.HydraConfigReader(path).read()
            self.assertEqual(config.total_steps, 200)
            self.assertEqual(config.gradient_accumulation_steps, 8)
            self.assertEqual(config.trajectory_cap, 256)
            self.assertEqual(config.eval_datasets, ("validation-a", "validation-b"))


class TensorBoardScalarReaderTests(unittest.TestCase):
    def test_reads_losses_timing_and_validation_from_live_event_file(self):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as directory:
            event_dir = Path(directory) / "runs_0"
            writer = SummaryWriter(event_dir)
            writer.add_scalar("live_total_loss", 0.3, 0)
            writer.add_scalar("timing/step", 12.5, 1)
            writer.add_scalar("eval_validation/model__survival__any", 81.2, 1)
            writer.flush()
            reader = dashboard.TensorBoardScalarReader(event_dir)
            scalars = reader.read()
            self.assertAlmostEqual(scalars["live_total_loss"][0]["value"], 0.3, places=5)
            self.assertEqual(scalars["timing/step"][0]["step"], 1)
            self.assertIn("eval_validation/model__survival__any", scalars)

            writer.add_scalar("live_total_loss", 0.2, 1)
            writer.flush()
            scalars = reader.read()
            self.assertEqual(len(scalars["live_total_loss"]), 2)
            writer.close()


class WandbRunReaderTests(unittest.TestCase):
    def test_merges_sparse_wandb_rows_by_optimizer_step(self):
        class FakeRun:
            state = "running"
            config = {
                "trainer": {
                    "num_steps": 37,
                    "gradient_accumulation_steps": 3,
                },
                "datasets": {"eval": {"names": ["held-out"]}},
            }

            def load(self, force=False):
                return {}

            def scan_history(self, **kwargs):
                self.scan_kwargs = kwargs
                return iter(
                    [
                        {
                            "_step": 4,
                            "global_step": 2,
                            "_timestamp": 100.0,
                            "live_total_loss": 0.4,
                        },
                        {
                            "_step": 5,
                            "global_step": 2,
                            "_timestamp": 100.1,
                            "hardware/gpu_1/utilization_percent": 81,
                        },
                    ]
                )

        run = FakeRun()
        api = SimpleNamespace(run=lambda path: run)
        reader = dashboard.WandbRunReader("entity/project/run", 10, api=api)

        scalars = reader.read()

        self.assertEqual(scalars["live_total_loss"][0]["step"], 2)
        self.assertEqual(
            scalars["hardware/gpu_1/utilization_percent"][0]["value"], 81
        )
        self.assertEqual(reader.config().total_steps, 37)
        self.assertEqual(reader.status(), "running")
        self.assertEqual(run.scan_kwargs["min_step"], 0)

    def test_builds_separate_gpu_series_from_logged_rank_metrics(self):
        scalars = {
            "hardware/gpu_0/utilization_percent": [
                {"step": 1, "value": 72.0, "wall_time": 10.0}
            ],
            "hardware/gpu_1/memory_used_gib": [
                {"step": 1, "value": 64.0, "wall_time": 10.0}
            ],
        }

        series = dashboard.gpu_series_from_scalars(scalars)

        self.assertEqual({point["gpu_index"] for point in series}, {0, 1})
        self.assertEqual(series[1]["vram_used_gib"], 64.0)


class SourceSeriesTests(unittest.TestCase):
    def test_groups_combined_and_per_source_losses(self):
        point = [{"step": 1, "value": 0.2}]
        losses = dashboard.loss_series_from_scalars(
            {
                "live_flow_loss": point,
                "source/diegesis/component/flow": point,
                "source/mvkubric/component/flow": point,
            }
        )

        self.assertEqual(losses["combined"]["flow"], point)
        self.assertEqual(losses["diegesis"]["flow"], point)
        self.assertEqual(losses["mvkubric"]["flow"], point)

    def test_keeps_full_and_subset_mvkubric_validation_separate(self):
        diegesis = [{"step": 250, "value": 80.0}]
        mvkubric = [{"step": 250, "value": 70.0}]
        validation = dashboard.validation_series_from_scalars(
            {
                "eval/tapvid3d-multiview-validation/"
                "eval_tapvid3d-multiview-validation/model__average_jaccard__any": diegesis,
                "eval/kubric-multiview-v3-validation-full/"
                "eval_kubric-multiview-v3-validation-full/model__average_jaccard__any": mvkubric,
                "eval/kubric-multiview-v3-validation-subset/"
                "eval_kubric-multiview-v3-validation-subset/model__average_jaccard__any": mvkubric,
            }
        )

        self.assertEqual(
            set(validation),
            {
                "combined_full",
                "combined_subset",
                "diegesis",
                "mvkubric_full",
                "mvkubric_subset",
            },
        )
        self.assertEqual(
            validation["mvkubric_subset"]["model__average_jaccard__any"],
            mvkubric,
        )
        self.assertEqual(
            validation["combined_subset"]["model__average_jaccard__any"],
            [{"step": 250, "value": 75.0}],
        )


class GPUHistoryTests(unittest.TestCase):
    def test_samples_at_bounded_frequency_and_capacity(self):
        calls = []

        def read_gpu():
            calls.append(len(calls))
            return {
                "utilization_percent": 90,
                "vram_used_gib": 20.0,
                "vram_total_gib": 24.0,
                "vram_percent": 83.3,
                "power_watts": 300.0,
                "temperature_c": 62,
            }

        history = dashboard.GPUHistory(read_gpu, interval_seconds=0.001, max_samples=2)
        for _ in range(3):
            history.sample_if_due(enabled=True)
            time.sleep(0.002)
        self.assertEqual(len(calls), 3)
        self.assertEqual(len(history.series()), 2)
        self.assertEqual(history.series()[0]["elapsed_seconds"], 0)


class DashboardStateTests(unittest.TestCase):
    def test_builds_live_snapshot_without_mutating_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            log_path = run_dir / "train.log"
            log_path.write_text(
                "Datapoint: ['000001'] (microbatch 1/1, waited  0.25s)\n"
                "FWD pass: num_points=64\n"
                "[optimizer:000001] elements_clipped=3 clipped=1\n"
                "[timing:000001] Total:  2.00s | Data:  0.50s | Fwd:  0.60s | "
                "Sync:  0.00s | Bwd:  0.90s |\n",
                encoding="utf-8",
            )
            gpu = dashboard.GPUHistory(
                lambda: {
                    "utilization_percent": 100,
                    "vram_used_gib": 22.0,
                    "vram_total_gib": 24.0,
                    "vram_percent": 91.7,
                    "power_watts": 300.0,
                    "temperature_c": 63,
                },
                interval_seconds=1,
                max_samples=10,
            )
            state = dashboard.TrainingDashboardState(
                run_dir,
                log_path,
                run_dir / "runs_0",
                gpu,
            )
            state.config_reader = SimpleNamespace(
                read=lambda: dashboard.RunConfig(2, 1, 256, 8, 24, 10, ()),
                error=None,
            )
            state.scalar_reader = SimpleNamespace(
                read=lambda: {
                    "live_total_loss": [{"step": 0, "value": 0.3, "wall_time": 1.0}],
                    "live_flow_loss": [{"step": 0, "value": 0.1, "wall_time": 1.0}],
                    "live_visibility_loss": [{"step": 0, "value": 0.2, "wall_time": 1.0}],
                    "source/diegesis/loss": [
                        {"step": 0, "value": 0.25, "wall_time": 1.0}
                    ],
                    "source/mvkubric/loss": [
                        {"step": 0, "value": 0.35, "wall_time": 1.0}
                    ],
                    "baseline/stationary_trajectory_loss": [
                        {"step": 0, "value": 0.5, "wall_time": 1.0}
                    ],
                    "optimization/grad_norm_pre_clip": [
                        {"step": 0, "value": 2.0, "wall_time": 1.0}
                    ],
                    "timing/step": [{"step": 1, "value": 2.0, "wall_time": 1.0}],
                    "sampling/motion_window_mean_m": [
                        {"step": 1, "value": 0.2, "wall_time": 1.0}
                    ],
                    "sampling/motion_full_dynamic_window_static_count": [
                        {"step": 1, "value": 3.0, "wall_time": 1.0}
                    ],
                },
                error=None,
            )
            snapshot = state.snapshot()
            self.assertEqual(snapshot["schema_version"], 2)
            self.assertEqual(snapshot["status"], "running")
            self.assertEqual(snapshot["progress"]["completed_steps"], 1)
            self.assertEqual(snapshot["summary"]["accepted_samples"], 1)
            self.assertEqual(snapshot["series"]["pipeline"][0]["tracks_mean"], 64)
            self.assertEqual(snapshot["series"]["gpu"][0]["utilization_percent"], 100)
            self.assertEqual(snapshot["series"]["baseline"]["stationary"][0]["value"], 0.5)
            self.assertEqual(snapshot["series"]["losses"]["combined"]["total"][0]["value"], 0.3)
            self.assertEqual(snapshot["series"]["losses"]["diegesis"]["total"][0]["value"], 0.25)
            self.assertEqual(snapshot["series"]["losses"]["mvkubric"]["total"][0]["value"], 0.35)
            self.assertEqual(snapshot["series"]["gradients"]["pre_clip"][0]["value"], 2.0)
            self.assertEqual(snapshot["series"]["motion"]["window_mean"][0]["value"], 0.2)
            self.assertEqual(
                snapshot["series"]["motion"]["full_dynamic_window_static"][0]["value"],
                3.0,
            )
            self.assertEqual(
                snapshot["series"]["gradients"]["clipped_step_rate_50"],
                [{"step": 1, "value": 1.0}],
            )


class DashboardHTTPTests(unittest.TestCase):
    def setUp(self):
        self.state = SimpleNamespace(
            snapshot=lambda: {
                "schema_version": 1,
                "format": "mvtracker_training_dashboard",
                "status": "running",
            }
        )
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            dashboard.make_handler(self.state, stream_interval_seconds=0.01),
        )
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_root_keeps_the_approved_graph_set_and_uses_sse(self):
        with urllib.request.urlopen(f"{self.base_url}/", timeout=2) as response:
            html = response.read().decode("utf-8")
        for chart_id in (
            "loss-combined",
            "loss-diegesis",
            "loss-mvkubric",
            "stationary-baseline",
            "stationary-ratio",
            "validation-combined",
            "validation-diegesis",
            "validation-mvkubric",
            "step-timing",
            "learning-rate",
            "gradient-norms",
            "gradient-cosine",
            "gradient-clipping",
            "rejection-rate",
            "track-count",
            "scene-coverage",
            "motion-path-length",
            "motion-window-buckets",
            "motion-window-mismatch",
            "gpu-util",
            "gpu-vram",
            "gpu-thermal",
            "training-throughput",
            "container-cpu",
            "container-memory",
        ):
            self.assertIn(f'id="{chart_id}"', html)
        self.assertIn("new EventSource('/api/stream')", html)
        self.assertIn("movingAveragePoints", html)
        self.assertIn("rawPoints", html)
        self.assertIn('id="raw-opacity"', html)
        self.assertIn("setRawOpacity", html)
        self.assertIn("Trailing 50-sample means", html)
        self.assertIn("`${label} · 50-sample mean`", html)
        self.assertIn("lossLines", html)
        self.assertIn("loss-large", html)
        self.assertIn("toggleLossSeries", html)
        self.assertIn("seriesKey:'trajectory'", html)
        self.assertIn("combined_full:'Combined · full 27'", html)
        self.assertIn("Full (27 scenes) and subset (101–102) remain separate", html)
        self.assertIn("movingAverageXY(trackMean)", html)
        self.assertNotIn("emaPoints", html)
        self.assertIn("Clipped steps (last 50)", html)
        self.assertIn("latest 50 optimizer steps", html)
        self.assertNotIn("setInterval", html)

    def test_state_endpoint_is_uncached_json(self):
        with urllib.request.urlopen(f"{self.base_url}/api/state", timeout=2) as response:
            payload = json.load(response)
            self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertEqual(payload["status"], "running")

    def test_stream_endpoint_pushes_an_sse_snapshot(self):
        with urllib.request.urlopen(f"{self.base_url}/api/stream", timeout=2) as response:
            self.assertEqual(response.headers["Content-Type"], "text/event-stream; charset=utf-8")
            line = response.readline().decode("utf-8")
        self.assertTrue(line.startswith("data: "))
        payload = json.loads(line.removeprefix("data: "))
        self.assertEqual(payload["format"], "mvtracker_training_dashboard")


class CommandLineTests(unittest.TestCase):
    def test_safe_loopback_defaults_and_explicit_gpu(self):
        args = dashboard.parse_args(["--run-dir", "/tmp/run", "--gpu-index", "2"])
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8766)
        self.assertEqual(args.gpu_index, 2)

    def test_gpu_index_is_required(self):
        with self.assertRaises(SystemExit):
            dashboard.parse_args(["--run-dir", "/tmp/run"])

    def test_wandb_mode_does_not_require_a_local_gpu(self):
        args = dashboard.parse_args(
            ["--wandb-run", "entity/project/run-id"]
        )
        self.assertEqual(args.wandb_run, "entity/project/run-id")
        self.assertIsNone(args.gpu_index)


if __name__ == "__main__":
    unittest.main()
