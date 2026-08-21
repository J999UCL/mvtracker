import unittest
from pathlib import Path

from mvtracker.profiling.modal_syn4d_split import (
    ARCHIVE_BYTES,
    SHARD_A_ENVIRONMENTS,
    SHARD_B_ENVIRONMENTS,
    SPLIT_MANIFEST,
    SPLIT_ROOTS,
    SPLIT_SEED,
    jobs,
    jobs_for,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (ROOT / "tools/modal_syn4d_split_setup.py").read_text(encoding="utf-8")
PROFILE = (ROOT / "mvtracker/profiling/modal_syn4d_split.py").read_text(encoding="utf-8")


class ModalSyn4DSplitTests(unittest.TestCase):
    def test_manifest_is_the_exact_fixed_split(self):
        self.assertEqual(len(SPLIT_MANIFEST), 20)
        self.assertEqual(sum(row["split"] == "train" for row in SPLIT_MANIFEST), 16)
        self.assertEqual(sum(row["split"] == "validation" for row in SPLIT_MANIFEST), 4)
        self.assertEqual(len({row["environment"] for row in SPLIT_MANIFEST}), 20)
        self.assertEqual(SPLIT_SEED, 72)
        self.assertTrue(all(row["sequence"].startswith("seq_") for row in SPLIT_MANIFEST))

    def test_archive_sizes_and_workers_are_balanced(self):
        all_jobs = jobs()
        self.assertEqual(len(all_jobs), 20)
        self.assertEqual(sum(job.archive_bytes for job in all_jobs), sum(ARCHIVE_BYTES.values()))
        a = jobs_for(SHARD_A_ENVIRONMENTS)
        b = jobs_for(SHARD_B_ENVIRONMENTS)
        self.assertEqual(len(a), 10)
        self.assertEqual(len(b), 10)
        self.assertEqual({job.environment for job in a} & {job.environment for job in b}, set())
        self.assertEqual({job.environment for job in a} | {job.environment for job in b}, set(ARCHIVE_BYTES))
        difference = abs(sum(job.archive_bytes for job in a) - sum(job.archive_bytes for job in b))
        self.assertEqual(difference, 100_986_048)

    def test_split_roots_are_separate(self):
        self.assertEqual(SPLIT_ROOTS["train"].as_posix(), "datasets/syn4d-mvtracker/train")
        self.assertEqual(SPLIT_ROOTS["validation"].as_posix(), "datasets/syn4d-mvtracker/validation")

    def test_launcher_uses_existing_dependency_interfaces_and_two_t4_workers(self):
        self.assertIn("_inputs", LAUNCHER)
        self.assertIn("download_body_motions", LAUNCHER)
        self.assertIn("download_sparse_clothing_tar", LAUNCHER)
        self.assertIn("datasets/syn4d/temple_group/private/smplx_addon_parts", PROFILE)
        for package in ("nvidia-dali-cuda120", "safetensors", "pandas", "matplotlib", "pypng", "kornia", "rerun-sdk", "tqdm", "pause"):
            self.assertIn(package, LAUNCHER)
        for filename in ("syn4d_track.py", "base_dataset.py", "utils.py"):
            self.assertIn(filename, LAUNCHER)
        self.assertIn("local_archive_map", LAUNCHER)
        self.assertIn("stale", LAUNCHER)
        self.assertIn("set(local_archive_map.get(archive, [])) | set(members)", LAUNCHER)
        self.assertEqual(LAUNCHER.count('gpu="T4"'), 2)
        self.assertIn('name="convert-shard-a"', LAUNCHER)
        self.assertIn('name="convert-shard-b"', LAUNCHER)
        self.assertNotIn(".spawn()", LAUNCHER)
        self.assertNotIn(".get()", LAUNCHER)
        self.assertIn("data_volume.commit()", LAUNCHER)
        self.assertIn("shutil.rmtree(work, ignore_errors=True)", LAUNCHER)
        self.assertIn("LEGACY_ARCHIVES.get(job.environment)", LAUNCHER)
        self.assertIn("archive = Path(source)", LAUNCHER)
        self.assertIn("HF_XET_CACHE", LAUNCHER)
        self.assertIn("rglob(f\"{motion}.npz\")", LAUNCHER)
        self.assertNotIn("range(1, 10)", LAUNCHER)
        self.assertNotIn("range(10, 20)", LAUNCHER)


if __name__ == "__main__":
    unittest.main()
