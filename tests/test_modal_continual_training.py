import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_launch_contract():
    path = ROOT / "mvtracker/profiling/modal_continual_training.py"
    spec = importlib.util.spec_from_file_location("modal_continual_training", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_launch_contract()


class ModalContinualTrainingContractTests(unittest.TestCase):
    def test_exact_gpu_container_and_billing_contract(self):
        self.assertEqual(contract.GPU_REQUEST, "H200:2")
        self.assertEqual(contract.GPU_COUNT, 2)
        self.assertEqual(contract.TRAIN_MEMORY_REQUEST_MIB, 64 * 1024)
        self.assertEqual(contract.MAX_CONTAINERS, 1)
        self.assertEqual(contract.EPHEMERAL_DISK_MIB, 512 * 1024)
        self.assertEqual(contract.CONTINUAL_RUN_SUBDIR, "continual-training")
        self.assertEqual(
            contract.MODAL_TAGS,
            {"owner": "jeet", "project": "mvtracker", "purpose": "training"},
        )

    def test_results_volume_is_rw_at_the_canonical_run_root(self):
        profile_source = (
            ROOT / "tools/modal_training_profile.py"
        ).read_text(encoding="utf-8")
        continual_source = (
            ROOT / "tools/modal_continual_training.py"
        ).read_text(encoding="utf-8")

        self.assertIn('RUN_VOLUME_NAME = "jeet-mvtracker-runs-v2"', profile_source)
        self.assertIn('RUN_ROOT = Path("/mnt/mvtracker-runs")', profile_source)
        self.assertIn("str(RUN_ROOT): run_volume,", continual_source)
        self.assertNotIn("str(RUN_ROOT): run_volume.with_mount_options", continual_source)
        self.assertNotIn("ephemeral_disk=", continual_source)

    def test_training_reads_data_volume_v2_directly(self):
        source = (
            ROOT / "tools/modal_continual_training.py"
        ).read_text(encoding="utf-8")
        train_start = source.index("def train_remote(")
        train_end = source.index("\ndef _default_run_name", train_start)
        training = source[train_start:train_end]

        self.assertIn("image=training_image", source)
        self.assertIn("training_image = _source_image(_dependency_image())", source)
        self.assertNotIn(".run_function(", source)
        self.assertEqual(
            source.count(
                "memory=(TRAIN_MEMORY_REQUEST_MIB, TRAIN_MEMORY_LIMIT_MIB)"
            ),
            1,
        )
        self.assertIn('"MVTRACKER_DATA_ROOT": DATA_VOLUME_ROOT', training)
        self.assertNotIn("stage_continual_training_data", training)
        self.assertIn(
            "str(DATA_ROOT): data_volume.with_mount_options(read_only=True)",
            source,
        )

    def test_volume_ingestion_extracts_each_archive_once(self):
        source = (ROOT / "tools/modal_volume_ingestion.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"--strip-components=2"', source)
        self.assertIn('"rapidgzip", "-d", "-c", "-P", "16"', source)
        self.assertNotIn("--exclude-from", source)

    def test_main_launch_requires_explicit_confirmation(self):
        contract.require_main_confirmation("smoke", False)
        contract.require_main_confirmation("memory_profile", False)
        contract.require_main_confirmation("main", True)
        contract.require_main_confirmation("syn4d_main", True)
        with self.assertRaisesRegex(RuntimeError, "--confirm-main"):
            contract.require_main_confirmation("main", False)
        contract.require_remote_main_confirmation("main", contract.MAIN_CONFIRMATION)
        contract.require_remote_main_confirmation(
            "syn4d_main", contract.SYN4D_MAIN_CONFIRMATION
        )
        with self.assertRaisesRegex(RuntimeError, "explicit confirmation"):
            contract.require_remote_main_confirmation("main", "")

    def test_preflight_refuses_to_interrupt_active_apps(self):
        active = tuple(
            contract.ActiveContainer(str(index), f"other-{index}")
            for index in range(contract.WORKSPACE_CONTAINER_LIMIT - 1)
        )
        with self.assertRaisesRegex(RuntimeError, "other-0, other-1"):
            contract.require_training_capacity(active)

    def test_preflight_reads_inventory_without_mutating_it(self):
        calls = []

        def runner(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(
                command,
                0,
                stdout='[{"container_id":"one","app_name":"other-user"}]',
                stderr="",
            )

        containers = contract.preflight_active_containers(runner)

        self.assertEqual(containers[0].app_name, "other-user")
        self.assertEqual(calls[0][0], ["modal", "container", "list", "--json"])
        self.assertNotIn("stop", calls[0][0])

    def test_commit_must_match_pushed_main(self):
        commit = "a" * 40

        def matching(command, **kwargs):
            return subprocess.CompletedProcess(
                command, 0, stdout=f"{commit}\trefs/heads/main\n", stderr=""
            )

        contract.require_pushed_main_commit(commit, matching)
        with self.assertRaisesRegex(RuntimeError, "not the pushed origin/main"):
            contract.require_pushed_main_commit(
                commit,
                lambda command, **kwargs: subprocess.CompletedProcess(
                    command, 0, stdout=f"{'b' * 40}\trefs/heads/main\n", stderr=""
                ),
            )

    def test_training_entrypoints_spawn_durable_function_calls(self):
        source = (ROOT / "tools/modal_continual_training.py").read_text(
            encoding="utf-8"
        )
        entrypoints = source[source.index('@app.local_entrypoint(name="smoke")') :]
        self.assertIn(
            'modal.Function.from_name(APP_NAME, "train_remote")', source
        )
        self.assertIn("deployed_training.spawn(", source)
        self.assertIn('"function_call_id": call.object_id', source)
        self.assertNotIn("train_remote.remote(", entrypoints)

    def test_syn4d_launch_uses_the_environment_disjoint_split(self):
        source = (ROOT / "tools/modal_continual_training.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("TRAIN_ENVIRONMENTS", source)
        self.assertIn("VALIDATION_ENVIRONMENTS", source)
        self.assertIn('"gt-depth-replay-syn4d-v2"', source)
        self.assertNotIn("expected 20", source)

    def test_resume_preserves_existing_identity_and_skips_duplicate_eval(self):
        source = (ROOT / "tools/modal_continual_training.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('seed = int(existing_manifest["master_seed"])', source)
        self.assertIn('wandb_run_id = str(existing_manifest["wandb_run_id"])', source)
        self.assertIn('command.append("modes.validate_at_start=false")', source)
        self.assertIn('"resume_source_commits"', source)
        self.assertIn("--resume-existing requires --run-name", source)


if __name__ == "__main__":
    unittest.main()
