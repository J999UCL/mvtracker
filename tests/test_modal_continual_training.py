import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _load_launch_contract():
    path = ROOT / "mvtracker/profiling/modal_continual_training.py"
    spec = importlib.util.spec_from_file_location("modal_continual_training", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


contract = _load_launch_contract()


class ContinualTrainingConfigTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "configs/experiment/diegesis_mvkubric_gt_ddp.yaml"
        self.config = yaml.safe_load(path.read_text(encoding="utf-8"))

    def test_scientific_recipe_is_encoded_in_config(self):
        train = self.config["datasets"]["train"]
        trainer = self.config["trainer"]

        self.assertEqual(train["name"], "mixed-diegesis-mvkubric-training")
        self.assertEqual(train["source_schedule"], ["diegesis", "mvkubric"] * 2)
        self.assertEqual(train["batch_size"], 1)
        self.assertEqual(train["sequence_len"], 24)
        self.assertEqual(train["traj_per_sample"], 2048)
        self.assertEqual(trainer["num_steps"], 1000)
        self.assertEqual(trainer["lr_schedule_steps"], 2000)
        self.assertEqual(trainer["gradient_accumulation_steps"], 4)
        self.assertEqual(trainer["lr"], 1e-4)
        self.assertEqual(trainer["anneal_strategy"], "linear")

    def test_base_config_preserves_legacy_scheduler_horizon(self):
        base = yaml.safe_load((ROOT / "configs/train.yaml").read_text(encoding="utf-8"))

        self.assertEqual(base["trainer"]["num_steps"], 200000)
        self.assertEqual(base["trainer"]["lr_schedule_steps"], 200100)

    def test_gt_depth_and_full_training_scene_policy(self):
        sources = self.config["datasets"]["train"]["sources"]

        self.assertEqual(len(sources["diegesis"]["include_scene_ids"]), 17)
        self.assertEqual(
            sources["mvkubric"]["include_scene_ids"],
            [str(scene) for scene in range(900, 998)],
        )
        self.assertFalse(self.config["augmentations"]["variable_depth_type"])
        self.assertTrue(self.config["augmentations"]["depth"])
        self.assertEqual(
            sources["mvkubric"]["view_count_probabilities"],
            [0.20, 0.10, 0.10, 0.25, 0.10, 0.25],
        )
        self.assertEqual(sources["mvkubric"]["trajectory_caps_by_view"], {5: 819, 6: 512})

    def test_wandb_policy_is_explicit(self):
        logging = self.config["logging"]

        self.assertTrue(logging["log_wandb"])
        self.assertEqual(logging["wandb_entity"], contract.WANDB_ENTITY)
        self.assertEqual(logging["wandb_project"], contract.WANDB_PROJECT)
        self.assertEqual(logging["wandb_group"], contract.WANDB_GROUP)
        self.assertIn("ddp2-h100", logging["tags"])
        self.assertEqual(
            self.config["evaluation"]["evaluator"],
            {
                "rerun_viz_indices": None,
                "forward_pass_log_indices": None,
                "mp4_track_viz_indices": None,
            },
        )

    def test_smoke_config_is_exactly_five_steps(self):
        path = ROOT / "configs/experiment/diegesis_mvkubric_gt_ddp_smoke.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(config["trainer"]["num_steps"], 5)
        self.assertFalse(config["modes"]["validate_at_start"])
        self.assertEqual(config["datasets"]["eval"]["names"], [])

        phase1_path = ROOT / "configs/experiment/diegesis_mvkubric_gt_ddp_smoke_phase1.yaml"
        phase1 = yaml.safe_load(phase1_path.read_text(encoding="utf-8"))
        self.assertEqual(phase1["trainer"]["num_steps"], 3)
        self.assertEqual(phase1["trainer"]["save_ckpt_freq"], 3)


class ModalContinualTrainingContractTests(unittest.TestCase):
    def test_exact_gpu_container_and_billing_contract(self):
        self.assertEqual(contract.GPU_REQUEST, "H100!:2")
        self.assertEqual(contract.GPU_COUNT, 2)
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
        self.assertGreaterEqual(continual_source.count("ephemeral_disk=EPHEMERAL_DISK_MIB"), 2)

    def test_main_launch_requires_explicit_confirmation(self):
        contract.require_main_confirmation("smoke", False)
        contract.require_main_confirmation("main", True)
        with self.assertRaisesRegex(RuntimeError, "--confirm-main"):
            contract.require_main_confirmation("main", False)
        contract.require_remote_main_confirmation("main", contract.MAIN_CONFIRMATION)
        with self.assertRaisesRegex(RuntimeError, "1000-step confirmation"):
            contract.require_remote_main_confirmation("main", "")

    def test_preflight_refuses_to_interrupt_active_apps(self):
        active = tuple(
            contract.ActiveContainer(str(index), f"other-{index}")
            for index in range(2)
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


if __name__ == "__main__":
    unittest.main()
