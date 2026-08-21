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
    copy_selected_tar_members,
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

    def test_streaming_copy_keeps_only_selected_members(self):
        payloads = {
            "subject/2000/2000.npz": b"PK\x03\x04selected",
            "unused/9999/9999.npz": b"PK\x03\x04unused",
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.tar"
            destination = Path(directory) / "selected.tar"
            with tarfile.open(source, "w") as archive:
                for name, payload in payloads.items():
                    info = tarfile.TarInfo(name)
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
            with source.open("rb") as stream:
                sizes = copy_selected_tar_members(
                    stream, destination, ["subject/2000/2000.npz"]
                )
            with tarfile.open(destination, "r") as archive:
                names = archive.getnames()
        self.assertEqual(names, ["subject/2000/2000.npz"])
        self.assertEqual(sizes, {"subject/2000/2000.npz": len(payloads[names[0]])})

    def test_modal_job_is_cpu_tagged_and_uses_named_secret(self):
        self.assertIn('"owner": "jeet"', (ROOT / "tools/modal_training_profile.py").read_text())
        self.assertIn('"experiment": "syn4d-bedlam-selective-dependencies"', LAUNCHER)
        self.assertIn('BEDLAM_SECRET_NAME = "jeet-mvtracker-bedlam"', LAUNCHER)
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
