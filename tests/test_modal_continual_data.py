import json
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from mvtracker.profiling.modal_continual_data import (
    CHECKPOINT_FILE,
    CHECKPOINT_REVISION,
    CHECKPOINT_SHA256,
    DIEGESIS_ARCHIVE_RELATIVE,
    EXPECTED_DIEGESIS_SPLITS,
    EXPECTED_MVKUBRIC_POOL_SCENES,
    EXPECTED_MVKUBRIC_SCENES,
    MVKUBRIC_INDEX_RELATIVE,
    MVKUBRIC_SHARDS,
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

            (mvkubric / "999").rename(mvkubric / "1000")
            with self.assertRaisesRegex(RuntimeError, "contain scenes 900..999"):
                _require_existing_profile_data(root)

    def test_archive_staging_extracts_sources_and_copies_sidecars(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            extracted = Path(directory) / "extracted"
            root.mkdir()
            manifest = {
                "mvkubric_revision": "pinned",
                "diegesis": {"splits": EXPECTED_DIEGESIS_SPLITS},
                "mvkubric": {"scene_count": 100},
            }
            (root / "profile-data-manifest.json").write_text(json.dumps(manifest))
            (root / "continual-training-data-manifest.json").write_text(json.dumps({}))
            (root / DIEGESIS_ARCHIVE_RELATIVE).parent.mkdir(parents=True)
            (root / DIEGESIS_ARCHIVE_RELATIVE).write_bytes(b"diegesis")
            for shard in MVKUBRIC_SHARDS:
                (root / shard).parent.mkdir(parents=True, exist_ok=True)
                (root / shard).write_bytes(b"mvkubric")
            (root / "checkpoints").mkdir()
            (root / "checkpoints/mvtracker_200000_june2025.pth").write_bytes(b"checkpoint")
            split_document = json.loads(
                (Path(__file__).resolve().parents[1] / "configs/diegesis_split_v1.json").read_text()
            )
            for split, scenes in split_document["splits"].items():
                for scene in scenes:
                    (root / "datasets/diegesis-mvtracker/TAPVid3D_MVTracker_cache" / split / scene).mkdir(parents=True)
            index_root = root / MVKUBRIC_INDEX_RELATIVE
            (index_root / "scenes").mkdir(parents=True)
            entries = {}
            for scene in EXPECTED_MVKUBRIC_POOL_SCENES:
                np.savez(index_root / "scenes" / f"{scene}.npz", index=np.zeros(1))
                entries[scene] = {"arrays": f"scenes/{scene}.npz"}
            (index_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "version": 1,
                        "source_fingerprint": hashlib.sha256().hexdigest(),
                        "scenes": entries,
                    }
                )
            )

            def extract(command, check):
                self.assertTrue(check)
                destination = Path(command[command.index("--directory") + 1])
                if "--strip-components=3" in command:
                    for scene in EXPECTED_MVKUBRIC_SCENES:
                        (destination / scene).mkdir(parents=True, exist_ok=True)
                    for scene in ("101", "102"):
                        (root / "datasets/kubric-multiview/train" / scene).mkdir(
                            parents=True, exist_ok=True
                        )
                else:
                    for scenes in split_document["splits"].values():
                        for scene in scenes:
                            (destination / "scenes" / scene / "tracking/sequence").mkdir(
                                parents=True, exist_ok=True
                            )

            with mock.patch(
                "mvtracker.profiling.modal_continual_data.subprocess.run",
                side_effect=extract,
            ):
                staging = stage_continual_training_data(root, local_data_root=extracted)
            self.assertEqual(staging["local_data_root"], str(extracted))
            self.assertTrue(
                (extracted / "checkpoints/mvtracker_200000_june2025.pth").is_file()
            )
            self.assertEqual(
                {
                    path.name
                    for path in (extracted / "datasets/kubric-multiview/train").iterdir()
                    if path.name.isdigit()
                },
                EXPECTED_MVKUBRIC_POOL_SCENES,
            )
            self.assertTrue(
                (extracted / "datasets/diegesis-mvtracker/TAPVid3D_raw/train").is_dir()
            )

    def test_profile_staging_uses_only_one_mvkubric_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            destination = Path(directory) / "local"
            shard = root / MVKUBRIC_SHARDS[0]
            shard.parent.mkdir(parents=True)
            shard.write_bytes(b"shard")
            index = root / MVKUBRIC_INDEX_RELATIVE
            index.mkdir(parents=True)
            (index / "manifest.json").write_text("{}")

            def extract(command, check):
                self.assertTrue(check)
                train_root = Path(command[command.index("--directory") + 1])
                for scene in range(900, 925):
                    (train_root / str(scene)).mkdir()

            with mock.patch(
                "mvtracker.profiling.modal_continual_data.subprocess.run",
                side_effect=extract,
            ):
                result = stage_mvkubric_profile_shard(
                    root, local_data_root=destination
                )

            self.assertEqual(result["scene_ids"], [str(scene) for scene in range(900, 925)])
            self.assertEqual(result["copied_size_bytes"], 5)


if __name__ == "__main__":
    unittest.main()
