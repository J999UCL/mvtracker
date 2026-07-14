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


if __name__ == "__main__":
    unittest.main()
