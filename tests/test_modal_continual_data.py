import json
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np

from mvtracker.profiling.modal_continual_data import (
    CHECKPOINT_FILE,
    CHECKPOINT_REVISION,
    CHECKPOINT_SHA256,
    EXPECTED_DIEGESIS_SPLITS,
    EXPECTED_MVKUBRIC_POOL_SCENES,
    MVKUBRIC_INDEX_RELATIVE,
    _require_existing_profile_data,
    stage_continual_training_data,
    stage_mvkubric_profile_shard,
)


class ModalContinualDataTests(unittest.TestCase):
    def test_checkpoint_is_pinned_and_checksummed(self):
        self.assertEqual(CHECKPOINT_FILE, "mvtracker_200000_june2025.pth")
        self.assertEqual(len(CHECKPOINT_REVISION), 40)
        self.assertEqual(len(CHECKPOINT_SHA256), 64)

    def test_setup_refuses_to_rebuild_missing_profile_data(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "does not download or rebuild"):
                _require_existing_profile_data(Path(directory))

    def test_setup_accepts_only_exact_existing_scene_sets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "mvkubric_revision": "pinned",
                "diegesis": {"splits": EXPECTED_DIEGESIS_SPLITS},
                "mvkubric": {"scene_count": 100},
            }
            (root / "profile-data-manifest.json").write_text(json.dumps(manifest))
            raw_root = root / "datasets/diegesis-mvtracker/TAPVid3D_raw"
            cache_root = root / "datasets/diegesis-mvtracker/TAPVid3D_MVTracker_cache"
            for split, count in EXPECTED_DIEGESIS_SPLITS.items():
                for index in range(count):
                    (raw_root / split / str(index)).mkdir(parents=True)
                    (cache_root / split / str(index)).mkdir(parents=True)
            mvkubric = root / "datasets/kubric-multiview/train"
            for scene in EXPECTED_MVKUBRIC_POOL_SCENES:
                (mvkubric / scene).mkdir(parents=True)
            index_root = root / MVKUBRIC_INDEX_RELATIVE
            (index_root / "scenes").mkdir(parents=True)
            index_entries = {}
            for scene in EXPECTED_MVKUBRIC_POOL_SCENES:
                np.savez(index_root / "scenes" / f"{scene}.npz", index=np.zeros(1))
                index_entries[scene] = {"arrays": f"scenes/{scene}.npz"}
            (index_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "source_fingerprint": hashlib.sha256().hexdigest(),
                        "scenes": index_entries,
                    }
                )
            )

            self.assertEqual(_require_existing_profile_data(root), manifest)

            (cache_root / "train" / "0").rmdir()
            with self.assertRaisesRegex(RuntimeError, "existing DIEGESIS train data is incomplete"):
                _require_existing_profile_data(root)
            (cache_root / "train" / "0").mkdir()

            (mvkubric / "1001").rename(mvkubric / "900")
            with self.assertRaisesRegex(RuntimeError, "1001..3000"):
                _require_existing_profile_data(root)

    def test_archive_staging_is_disabled_for_the_cached_image(self):
        with self.assertRaisesRegex(RuntimeError, "legacy archive staging is disabled"):
            stage_continual_training_data(Path("unused"), local_data_root=Path("unused"))

    def test_profile_shard_staging_is_disabled_for_the_cached_image(self):
        with self.assertRaisesRegex(RuntimeError, "legacy MV-Kubric shard staging is disabled"):
            stage_mvkubric_profile_shard(Path("unused"), local_data_root=Path("unused"))


if __name__ == "__main__":
    unittest.main()
