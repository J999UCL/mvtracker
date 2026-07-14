import json
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from scripts import pointodyssey_preprocessing_dashboard as dashboard


def _snapshot(status: str = "converting") -> dict:
    unit_progress = {"completed": 1, "total": 2, "percent": 50.0}
    return {
        "schema_version": 1,
        "format": "pointodyssey_preprocessing_progress",
        "status": status,
        "started_at": "2026-07-14T12:00:00Z",
        "updated_at": "2026-07-14T12:00:01Z",
        "elapsed_seconds": 1.0,
        "source_root": "/source",
        "output_root": "/output",
        "workers": 4,
        "active": {
            "stage": "source_conversion",
            "phase": "rgb",
            "layout": "raw",
            "source_sequence": "candidate_empty_office",
            "split": "train",
            "scene_id": "000000",
            "view": 0,
        },
        "progress": {
            key: dict(unit_progress)
            for key in (
                "sources",
                "scenes",
                "frames",
                "camera_frames",
                "jpegs",
                "validated_jpegs",
            )
        }
        | {"output_bytes": {"completed": 1048576, "total": None, "percent": None}},
        "rates": {
            "sources_per_second": 1.0,
            "scenes_per_second": 2.0,
            "frames_per_second": 3.0,
            "camera_frames_per_second": 4.0,
            "jpegs_per_second": 5.0,
            "validated_jpegs_per_second": 6.0,
            "output_mib_per_second": 7.0,
        },
        "diagnostics": {
            "semantic_validation_failures": 0,
            "invalid_rgb_frames": 0,
        },
        "timing": {
            "current_stage_elapsed_seconds": 0.5,
            "stages_seconds": {"preflight": 0.25},
        },
        "statistics": {
            "tracks": {"finite_coordinate_values": 12},
            "depth": {"positive": 10, "finite_min": 0.1},
            "visibility": {"accepted": 8},
            "rgb": {"output_frames": 1, "output_bytes": 1048576},
            "io": {"source_files": 22, "output_files": 6},
        },
        "error": None,
    }


class ProgressFileTests(unittest.TestCase):
    def test_reads_exact_progress_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            snapshot = _snapshot()
            path.write_text(json.dumps(snapshot), encoding="utf-8")
            self.assertEqual(dashboard.read_progress(path), snapshot)

    def test_rejects_wrong_schema_and_format(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "progress.json"
            for mutation in (
                {"schema_version": 2},
                {"format": "not_pointodyssey_progress"},
            ):
                snapshot = _snapshot() | mutation
                path.write_text(json.dumps(snapshot), encoding="utf-8")
                with self.assertRaises(dashboard.ProgressFileError):
                    dashboard.read_progress(path)


class DashboardHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.progress_path = Path(self.temporary_directory.name) / "progress.json"
        self.server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            dashboard.make_handler(self.progress_path),
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary_directory.cleanup()

    def test_root_serves_detailed_live_dashboard(self):
        with urllib.request.urlopen(f"{self.base_url}/", timeout=2) as response:
            html = response.read().decode("utf-8")
        self.assertIn("PointOdyssey preprocessing", html)
        self.assertIn("validated_jpegs_per_second", html)
        self.assertIn("Detailed running statistics", html)
        self.assertIn("item.total === null", html)
        self.assertIn("setInterval(refresh, 1000)", html)
        self.assertNotIn("setInterval(() => refresh(), 10000)", html)

    def test_api_rereads_each_atomic_snapshot(self):
        self.progress_path.write_text(json.dumps(_snapshot("starting")), encoding="utf-8")
        with urllib.request.urlopen(f"{self.base_url}/api/progress", timeout=2) as response:
            first = json.load(response)
            self.assertEqual(response.headers["Cache-Control"], "no-store")
        replacement = self.progress_path.with_suffix(".tmp")
        replacement.write_text(json.dumps(_snapshot("completed")), encoding="utf-8")
        replacement.replace(self.progress_path)
        with urllib.request.urlopen(f"{self.base_url}/api/progress", timeout=2) as response:
            second = json.load(response)
        self.assertEqual(first["status"], "starting")
        self.assertEqual(second["status"], "completed")

    def test_api_reports_missing_snapshot_without_crashing_server(self):
        with self.assertRaises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(f"{self.base_url}/api/progress", timeout=2)
        self.assertEqual(raised.exception.code, 503)
        payload = json.loads(raised.exception.read().decode("utf-8"))
        self.assertEqual(payload["error"]["type"], "ProgressFileError")


class CommandLineTests(unittest.TestCase):
    def test_safe_loopback_defaults(self):
        args = dashboard.parse_args(["--progress-json", "/tmp/progress.json"])
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8765)

    def test_invalid_port_is_rejected(self):
        with self.assertRaises(SystemExit):
            dashboard.parse_args(["--progress-json", "/tmp/progress.json", "--port", "0"])


if __name__ == "__main__":
    unittest.main()
