import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "tools/modal_mvkubric_full_conversion.py").read_text(encoding="utf-8")
PROFILE_SOURCE = (ROOT / "tools/modal_training_profile.py").read_text(encoding="utf-8")


class ModalMvKubricFullConversionContractTests(unittest.TestCase):
    def test_cpu_modal_shape_and_billing_tags(self):
        self.assertIn("BASE_TAGS", SOURCE)
        self.assertIn("MODAL_TAGS", SOURCE)
        self.assertIn('"purpose": "profiling"', PROFILE_SOURCE)
        self.assertIn("cpu=16", SOURCE)
        self.assertIn("memory=32768", SOURCE)
        self.assertIn("ephemeral_disk=1024 * 1024", SOURCE)
        self.assertIn("timeout=24 * 60 * 60", SOURCE)
        self.assertIn("retries=2", SOURCE)
        self.assertIn("max_containers=1", SOURCE)
        self.assertNotIn('gpu=', SOURCE)

    def test_staging_is_sequential_and_local(self):
        self.assertIn("for archive_name, start, end in TRAIN_ARCHIVES", SOURCE)
        self.assertIn("LOCAL_ROOT / archive_name", SOURCE)
        self.assertIn("rapidgzip", SOURCE)
        self.assertIn('"--strip-components=2"', SOURCE)
        self.assertIn("local_archive.unlink()", SOURCE)
        self.assertIn("shutil.rmtree(local_extract)", SOURCE)

    def test_progress_and_resume_contract(self):
        self.assertIn("HEARTBEAT_SECONDS = 30.0", SOURCE)
        self.assertIn("COPY_PROGRESS_BYTES = 10 * (1 << 30)", SOURCE)
        self.assertIn('self.emit("heartbeat"', SOURCE)
        self.assertIn('"scene_complete"', SOURCE)
        self.assertIn('"shard_complete"', SOURCE)
        self.assertIn("_clear_partials(staging_root)", SOURCE)
        self.assertIn("finalize=False", SOURCE)
        self.assertIn("finalize_shards", SOURCE)

    def test_train_and_validation_publish_together(self):
        self.assertIn("validation_source: str = str(VALIDATION_SOURCE)", SOURCE)
        publish = SOURCE.index('progress.emit("published"')
        validation = SOURCE.index('validation_manifest = finalize_shards')
        self.assertGreater(publish, validation)


if __name__ == "__main__":
    unittest.main()
