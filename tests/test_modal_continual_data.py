import json
from pathlib import Path
import tempfile
import unittest

from mvtracker.profiling.modal_continual_data import (
    CHECKPOINT_FILE,
    CHECKPOINT_REVISION,
    CHECKPOINT_SHA256,
    EXPECTED_DIEGESIS_SPLITS,
    EXPECTED_MVKUBRIC_SCENES,
    _require_existing_profile_data,
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
            for scene in EXPECTED_MVKUBRIC_SCENES:
                (mvkubric / scene).mkdir(parents=True)

            self.assertEqual(_require_existing_profile_data(root), manifest)

            (mvkubric / "999").rename(mvkubric / "1000")
            with self.assertRaisesRegex(RuntimeError, "exactly scenes 900..999"):
                _require_existing_profile_data(root)


if __name__ == "__main__":
    unittest.main()
