import unittest
from pathlib import Path

import yaml


class PointOdysseyTrainingConfigTests(unittest.TestCase):
    def setUp(self):
        self.config_root = Path(__file__).resolve().parents[1] / "configs"

    def test_upstream_default_keeps_one_microbatch_per_step(self):
        config = yaml.safe_load((self.config_root / "train.yaml").read_text())

        self.assertEqual(config["trainer"]["gradient_accumulation_steps"], 1)

    def test_pointodyssey_uses_eight_serial_microbatches(self):
        config = yaml.safe_load(
            (self.config_root / "experiment" / "pointodyssey.yaml").read_text()
        )

        self.assertEqual(config["trainer"]["gradient_accumulation_steps"], 8)
        self.assertEqual(
            config["datasets"]["pointodyssey_prepared_dir"],
            "PointOdyssey_MVTracker_v5",
        )

    def test_pointodyssey_long_run_settings(self):
        config = yaml.safe_load(
            (self.config_root / "experiment" / "pointodyssey.yaml").read_text()
        )

        self.assertEqual(config["trainer"]["num_steps"], 2000)
        self.assertEqual(config["trainer"]["lr"], 0.00005)
        self.assertEqual(config["trainer"]["eval_freq"], 500)
        self.assertEqual(config["trainer"]["viz_freq"], 500)
        self.assertEqual(config["trainer"]["save_ckpt_freq"], 500)
        self.assertFalse(config["augmentations"]["rgb"])
        self.assertTrue(config["augmentations"]["variable_trajpersample"])
        self.assertTrue(config["modes"]["validate_at_start"])
        self.assertTrue(config["logging"]["log_wandb"])
        self.assertEqual(
            config["logging"]["wandb_project"],
            "mvtracker-pointodyssey",
        )

    def test_pointodyssey_registers_only_the_validation_split_for_evaluation(self):
        config = yaml.safe_load(
            (self.config_root / "experiment" / "pointodyssey.yaml").read_text()
        )

        self.assertEqual(
            config["datasets"]["eval"]["names"],
            ["pointodyssey-multiview-validation"],
        )

    def test_pointodyssey_overfit_repeats_one_unaugmented_sample(self):
        config = yaml.safe_load(
            (self.config_root / "experiment" / "pointodyssey_overfit.yaml").read_text()
        )

        self.assertEqual(config["datasets"]["train"]["traj_per_sample"], 64)
        self.assertEqual(config["datasets"]["train"]["num_workers"], 0)
        self.assertEqual(config["trainer"]["gradient_accumulation_steps"], 1)
        self.assertEqual(config["augmentations"]["probability"], 0.0)
        self.assertTrue(config["modes"]["tune_per_scene"])
        self.assertFalse(config["modes"]["validate_at_start"])
        for key in (
            "rgb",
            "depth",
            "cropping",
            "variable_trajpersample",
            "scene_transform",
            "camera_params_noise",
        ):
            self.assertFalse(config["augmentations"][key])


if __name__ == "__main__":
    unittest.main()
