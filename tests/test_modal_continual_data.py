import json
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mvtracker.profiling.modal_continual_data import (
    CHECKPOINT_FILE,
    CHECKPOINT_REVISION,
    CHECKPOINT_SHA256,
    EXPECTED_DIEGESIS_SPLITS,
    EXPECTED_MVKUBRIC_POOL_SCENES,
    MVKUBRIC_INDEX_RELATIVE,
    SYN4D_LAB_BALD_SEQUENCES,
    _require_existing_profile_data,
    profile_encoded_loader,
    stage_continual_training_data,
    stage_mvkubric_profile_shard,
)


class ModalContinualDataTests(unittest.TestCase):
    def test_profile_rejects_unknown_source_before_setup(self):
        with self.assertRaisesRegex(ValueError, "diegesis, mvkubric, or syn4d"):
            profile_encoded_loader(Path("unused"), source="unknown", measured=1)

    def test_syn4d_profile_uses_all_lab_bald_sequences(self):
        class Dataset:
            real_len = 20

            def __getitem__(self, index):
                return SimpleNamespace(
                    jpeg_bytes=(b"jpeg",) * 96,
                    metadata={"worker_prepare_seconds": 0.1},
                ), True

        config = SimpleNamespace(
            datasets=SimpleNamespace(train=SimpleNamespace())
        )
        omega_conf = SimpleNamespace(
            load=lambda _: config,
            merge=lambda *_: config,
        )
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            sys.modules,
            {"omegaconf": SimpleNamespace(OmegaConf=omega_conf)},
        ), patch(
            "mvtracker.datasets.syn4d_multiview_dataset.Syn4DMultiViewDataset.from_name",
            return_value=Dataset(),
        ) as factory:
            result = profile_encoded_loader(
                Path(directory), source="syn4d", warmup=0, measured=1,
                workers=0, use_cuda=False, view_count=4,
            )

        self.assertEqual(result["measured_sources"], ["syn4d"])
        self.assertEqual(result["encoded_frames"], 96)
        args, kwargs = factory.call_args
        self.assertEqual(args[:2], (
            "syn4d-multiview-training",
            str(Path(directory) / "datasets/syn4d-mvtracker"),
        ))
        self.assertEqual(kwargs["include_scene_ids"], list(SYN4D_LAB_BALD_SEQUENCES))
        self.assertEqual(SYN4D_LAB_BALD_SEQUENCES[0], "lab_bald__seq_000000")
        self.assertEqual(SYN4D_LAB_BALD_SEQUENCES[-1], "lab_bald__seq_000019")

    def test_checkpoint_is_pinned_and_checksummed(self):
        self.assertEqual(CHECKPOINT_FILE, "mvtracker_200000_june2025.pth")
        self.assertEqual(len(CHECKPOINT_REVISION), 40)
        self.assertEqual(len(CHECKPOINT_SHA256), 64)

    def test_setup_refuses_to_rebuild_missing_profile_data(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "does not download or rebuild"):
                _require_existing_profile_data(Path(directory))

    def test_setup_accepts_the_indexed_scene_inventory(self):
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
            (index_root / "scenes/1001.npz").rename(index_root / "scenes/900.npz")
            index_entries["900"] = index_entries.pop("1001")
            index_entries["900"]["arrays"] = "scenes/900.npz"
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

    def test_archive_staging_is_disabled_for_the_cached_image(self):
        with self.assertRaisesRegex(RuntimeError, "legacy archive staging is disabled"):
            stage_continual_training_data(Path("unused"), local_data_root=Path("unused"))

    def test_profile_shard_staging_is_disabled_for_the_cached_image(self):
        with self.assertRaisesRegex(RuntimeError, "legacy MV-Kubric shard staging is disabled"):
            stage_mvkubric_profile_shard(Path("unused"), local_data_root=Path("unused"))


if __name__ == "__main__":
    unittest.main()
