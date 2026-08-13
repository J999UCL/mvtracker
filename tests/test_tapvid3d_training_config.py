import unittest
from pathlib import Path

import yaml


class TapVid3DTrainingConfigTests(unittest.TestCase):
    def test_procedural_config_uses_gpu_loader_defaults(self):
        path = Path(__file__).resolve().parents[1] / "configs/experiment/tapvid3d_procedural.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        self.assertEqual(config["datasets"]["train"]["name"], "tapvid3d-multiview-training")
        self.assertEqual(config["datasets"]["train"]["sequence_len"], 24)
        self.assertEqual(config["datasets"]["train"]["num_workers"], 4)
        self.assertEqual(config["datasets"]["train"]["prefetch_factor"], 2)
        self.assertFalse(config["augmentations"]["rgb"])
        self.assertTrue(config["logging"]["log_wandb"])


if __name__ == "__main__":
    unittest.main()
