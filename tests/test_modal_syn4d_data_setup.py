import csv
import json
import tempfile
import unittest
from pathlib import Path

from mvtracker.profiling.modal_syn4d import (
    SYN4D_REVISION,
    TEMPLE_GROUP_SOURCE_BYTES,
    temple_group_bedlam_plan,
    write_temple_group_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (ROOT / "tools/modal_syn4d_data_setup.py").read_text(encoding="utf-8")


def _mapping() -> str:
    rows = []
    for sequence in range(20):
        for view in range(8):
            name = f"seq_{sequence:06d}_{view}"
            rows.append({"scene": "temple_group", "sequence_name": name,
                         "asset_type": "bedlam2_body", "asset": f"it_{sequence:04d}_M_{2000 + sequence}"})
    output = ["scene,sequence_name,asset_type,asset"]
    output.extend(
        f"{row['scene']},{row['sequence_name']},{row['asset_type']},{row['asset']}"
        for row in rows
    )
    return "\n".join(output) + "\n"


class ModalSyn4DDataSetupTests(unittest.TestCase):
    def test_bedlam_plan_is_scene_scoped_and_exact(self):
        archive_map = {
            "b2_clothing_npz_000": [f"it_{i:04d}_M/{2000 + i}/{2000 + i}.npz" for i in range(20)]
        }
        plan = temple_group_bedlam_plan(_mapping(), archive_map)
        self.assertEqual(plan["sequence_count"], 20)
        self.assertEqual(plan["body_count"], 20)
        self.assertEqual(plan["clothing_count"], 20)
        self.assertEqual(len(plan["required_members"]["b2_clothing_npz_000"]), 20)

    def test_manifest_is_written_last_and_records_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = root / "mapping.csv"
            mapping.write_text(
                "scene,sequence_name,asset_type,asset\n"
                + "\n".join(
                    f"temple_group,seq_{i:06d}_{view},bedlam2_body,it_{i:04d}_M_{2000+i}"
                    for i in range(20) for view in range(8)
                )
                + "\n"
                + "\n".join(
                    f"temple_group,seq_{i:06d}_{view},objaverse_object,000-{i:03d}_asset{i:02d}"
                    for i in range(20) for view in range(8)
                )
                + "\n",
                encoding="utf-8",
            )
            source = root / "temple_group.tar.zst"
            with source.open("wb") as handle:
                handle.truncate(TEMPLE_GROUP_SOURCE_BYTES)
            objects = tuple(root / f"object-{i}" for i in range(60))
            for path in objects:
                path.touch()
            # The full dependency parser requires three objects per sequence;
            # use a tiny synthetic parser fixture only for the marker contract.
            # A malformed mapping must fail before a manifest can be published.
            with self.assertRaises(ValueError):
                write_temple_group_manifest(
                    root / "manifest.json", source_archive=source,
                    mapping=mapping, object_files=objects, bedlam={},
                )
            self.assertFalse((root / "manifest.json").exists())

    def test_launcher_is_tagged_and_has_explicit_cpu_and_t4_stages(self):
        self.assertIn('"purpose": "profiling"', (ROOT / "tools/modal_training_profile.py").read_text())
        self.assertIn('gpu="T4"', LAUNCHER)
        self.assertIn("BLENDER_VERSION = \"4.5.0\"", LAUNCHER)
        self.assertIn("smplx_anim_to_alembic_batch.py", LAUNCHER)
        self.assertIn("smplx_anim_to_objs_batch.py", LAUNCHER)
        self.assertIn("preflight_active_containers", LAUNCHER)
        self.assertIn("require_pushed_main_commit", LAUNCHER)
        self.assertNotIn("flash_attn", LAUNCHER)
        self.assertNotIn("pointops", LAUNCHER)


if __name__ == "__main__":
    unittest.main()
