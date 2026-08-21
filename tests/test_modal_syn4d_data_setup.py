import tempfile
import unittest
from pathlib import Path

from mvtracker.profiling.modal_syn4d import (
    SYN4D_REVISION,
    SYN4D_SOURCE_BYTES,
    sequence_bedlam_plan,
    sequence_dependencies,
    sequence_object_paths,
    write_sequence_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (ROOT / "tools/modal_syn4d_data_setup.py").read_text(encoding="utf-8")
PROFILE = (ROOT / "mvtracker/profiling/modal_syn4d.py").read_text(encoding="utf-8")


def _mapping() -> str:
    rows = ["scene,sequence_name,asset_type,asset"]
    for view in range(8):
        name = f"seq_000000_{view}"
        rows.append(f"lab_bald,{name},bedlam2_body,it_0000_M_2000")
        rows.extend(
            f"lab_bald,{name},objaverse_object,000-{index:03d}_asset{index:02d}"
            for index in range(3)
        )
    return "\n".join(rows) + "\n"


class ModalSyn4DDataSetupTests(unittest.TestCase):
    def test_dependency_plan_is_exactly_one_lab_bald_sequence(self):
        plan = sequence_dependencies(_mapping())
        self.assertEqual(plan["scene"], "lab_bald")
        self.assertEqual(plan["sequence"], "seq_000000")
        self.assertEqual(plan["views"], list(range(8)))
        self.assertEqual(plan["motions"], ["it_0000_M_2000"])
        self.assertEqual(plan["clothing_member"], "it_0000_M/2000/2000.npz")
        self.assertEqual(plan["object_count"], 3)
        self.assertEqual(len(sequence_object_paths(_mapping())), 3)

    def test_bedlam_plan_selects_one_cached_clothing_archive(self):
        archive_map = {
            "b2_clothing_npz_000": ["it_0000_M/2000/2000.npz"],
            "unused": ["other/2001/2001.npz"],
        }
        plan = sequence_bedlam_plan(_mapping(), archive_map)
        self.assertEqual(plan["body_count"], 1)
        self.assertEqual(plan["clothing_count"], 1)
        self.assertEqual(plan["required_members"], {
            "b2_clothing_npz_000": ["it_0000_M/2000/2000.npz"]
        })

    def test_manifest_is_not_published_when_bedlam_cache_is_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mapping = root / "mapping.csv"
            mapping.write_text(_mapping(), encoding="utf-8")
            source = root / "lab_bald.tar.zst"
            with source.open("wb") as handle:
                handle.truncate(SYN4D_SOURCE_BYTES)
            objects = tuple(root / f"object-{index}" for index in range(3))
            for path in objects:
                path.touch()
            plan = sequence_bedlam_plan(
                _mapping(), {"b2_clothing_npz_000": ["it_0000_M/2000/2000.npz"]}
            )
            with self.assertRaises(FileNotFoundError):
                write_sequence_manifest(
                    root / "manifest.json", source_archive=source, mapping=mapping,
                    object_files=objects, bedlam=plan, bedlam_root=root / "bedlam",
                )
            self.assertFalse((root / "manifest.json").exists())

    def test_launcher_is_single_sequence_and_all_jobs_are_billing_tagged(self):
        self.assertIn('"purpose": "profiling"', (ROOT / "tools/modal_training_profile.py").read_text())
        self.assertIn('SYN4D_HF_SOURCE = "data/syn4d_v1_stride_1/lab_bald.tar.zst"', PROFILE)
        self.assertIn("convert_syn4d_scene", LAUNCHER)
        self.assertIn('gpu="T4"', LAUNCHER)
        self.assertIn('"gpu": "cpu"', LAUNCHER)
        self.assertIn("for index in range(5)", LAUNCHER)
        self.assertIn("len(sample.jpeg_bytes) != 4 * 24", LAUNCHER)
        self.assertIn("(4, 24, 1, 384, 683)", LAUNCHER)
        self.assertIn("plan.output_size != (384, 512)", LAUNCHER)


if __name__ == "__main__":
    unittest.main()
