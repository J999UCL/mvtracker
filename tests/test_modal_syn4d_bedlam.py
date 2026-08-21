import csv
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from mvtracker.profiling.modal_syn4d_bedlam import (
    CLOTHING_ROOT,
    build_dependency_plan,
    parse_tar_header,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (ROOT / "tools/modal_syn4d_bedlam_setup.py").read_text(encoding="utf-8")


class ModalSyn4DBedlamTests(unittest.TestCase):
    def test_plan_keeps_only_exact_required_members(self):
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=("scene", "sequence_name", "asset_type", "asset"),
        )
        writer.writeheader()
        motions = [f"it_{index:04d}_M_2000" for index in range(298)]
        for index in range(299):
            writer.writerow(
                {
                    "scene": "arena",
                    "sequence_name": f"seq_{index:06d}_0",
                    "asset_type": "bedlam2_body",
                    "asset": motions[index % len(motions)],
                }
            )
        archive_map = {f"b2_clothing_npz_{index:03d}": [] for index in range(54)}
        for index, motion in enumerate(motions):
            subject, animation = motion.rsplit("_", 1)
            archive_map[f"b2_clothing_npz_{index % 54:03d}"].append(
                f"{subject}/{animation}/{animation}.npz"
            )
        archive_map["b2_clothing_npz_275"] = ["unused/9999/9999.npz"]

        plan = build_dependency_plan(output.getvalue(), archive_map)
        self.assertEqual(plan["sequence_count"], 299)
        self.assertEqual(len(plan["motions"]), 298)
        self.assertEqual(len(plan["required_members"]), 54)
        self.assertNotIn("b2_clothing_npz_275", plan["required_members"])

    def test_remote_tar_header_parser_matches_tarfile(self):
        payload = b"PK\x03\x04synthetic-npz"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.tar"
            with tarfile.open(path, "w") as archive:
                info = tarfile.TarInfo("subject/2000/2000.npz")
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            parsed = parse_tar_header(path.read_bytes()[:512], 0)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.name, "subject/2000/2000.npz")
        self.assertEqual(parsed.data_offset, 512)
        self.assertEqual(parsed.size, len(payload))

    def test_modal_job_is_cpu_tagged_and_uses_ephemeral_secret(self):
        self.assertIn('"owner": "jeet"', (ROOT / "tools/modal_training_profile.py").read_text())
        self.assertIn('"experiment": "syn4d-bedlam-selective-dependencies"', LAUNCHER)
        self.assertIn("modal.Secret.from_dict(_bedlam_credentials())", LAUNCHER)
        self.assertIn("cpu=8", LAUNCHER)
        self.assertNotIn("gpu=", LAUNCHER)
        self.assertNotIn("BEDLAM_EMAIL=", LAUNCHER)
        self.assertEqual(
            CLOTHING_ROOT.as_posix(),
            "datasets/syn4d/v1-stride1-12train-4validation/metadata/"
            "b2_assetdata_download/clothing/npz",
        )


if __name__ == "__main__":
    unittest.main()
