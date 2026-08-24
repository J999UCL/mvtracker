import json
import unittest
from pathlib import Path

from mvtracker.preprocessing.materialization_manifests import (
    load_mvkubric_manifest,
    load_syn4d_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


class ManifestMaterializationTests(unittest.TestCase):
    def test_syn4d_manifest_deduplicates_rows(self):
        path = ROOT / "manifests/syn4d-stride1-backfill.json"
        payload, rows = load_syn4d_manifest(path)
        self.assertEqual(payload["name"], "syn4d-stride1-backfill")
        self.assertEqual(rows, [{
            "environment": "countryside",
            "sequence": "seq_000000",
            "split": "train",
        }])

    def test_mvkubric_backfill_range_excludes_validation(self):
        payload = json.loads(
            (ROOT / "manifests/mvkubric-0031-1000-backfill.json").read_text()
        )
        self.assertEqual(payload["name"], "mvkubric-0031-1000-backfill")
        source = (ROOT / "tools/modal_mvkubric_0031_backfill.py").read_text()
        self.assertIn("0031-1000", source)
        self.assertNotIn("1001-2000", source)
        self.assertNotIn("2001-3000", source)

    def test_launchers_log_heartbeats_and_use_cpu(self):
        syn = (ROOT / "tools/modal_syn4d_stride1.py").read_text()
        mvk = (ROOT / "tools/modal_mvkubric_0031_backfill.py").read_text()
        for source in (syn, mvk):
            self.assertIn('"heartbeat"', source)
            self.assertIn("data_volume.commit()", source)
            self.assertIn('"gpu": "cpu"', source)

    def test_mvkubric_manifest_reader_keeps_numeric_scene_ids(self):
        path = ROOT / "manifests/mvkubric-0031-1000-backfill.json"
        payload, scenes = load_mvkubric_manifest(path)
        self.assertEqual(payload["name"], "mvkubric-0031-1000-backfill")
        self.assertEqual(len(scenes), 943)
        self.assertEqual(scenes[0], "31")
        self.assertEqual(scenes[-1], "1000")
        self.assertNotIn("101", scenes)


if __name__ == "__main__":
    unittest.main()
