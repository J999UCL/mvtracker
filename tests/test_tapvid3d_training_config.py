import hashlib
import json
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

    def test_diegesis_config_uses_clean_depth_3090_recipe(self):
        path = Path(__file__).resolve().parents[1] / "configs/experiment/diegesis.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertEqual(config["datasets"]["root"], "${oc.env:DIEGESIS_MVTRACKER_ROOT}")
        self.assertEqual(config["datasets"]["train"]["name"], "tapvid3d-multiview-training")
        self.assertEqual(config["datasets"]["train"]["traj_per_sample"], 256)
        self.assertEqual(
            config["datasets"]["eval"]["names"],
            ["tapvid3d-multiview-validation"],
        )
        self.assertEqual(config["trainer"]["gradient_accumulation_steps"], 8)
        self.assertEqual(config["trainer"]["num_steps"], 2000)
        self.assertEqual(config["trainer"]["eval_freq"], 250)
        self.assertTrue(config["modes"]["validate_at_start"])
        self.assertTrue(config["augmentations"]["rgb"])
        self.assertTrue(config["augmentations"]["depth"])
        self.assertTrue(config["augmentations"]["variable_num_views"])
        self.assertFalse(config["augmentations"]["variable_depth_type"])
        self.assertIn("cleandepth", config["restore_ckpt_path"])
        self.assertFalse(config["logging"]["log_wandb"])
        self.assertEqual(config["logging"]["wandb_project"], "mvtracker-diegesis")
        self.assertNotIn("model", config)

    def test_diegesis_split_is_complete_disjoint_and_matches_its_algorithm(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "configs/diegesis_split_v1.json").read_text())
        splits = manifest["splits"]

        self.assertEqual({name: len(scenes) for name, scenes in splits.items()}, {
            "train": 17,
            "validation": 2,
            "test": 2,
        })
        all_scenes = [scene for scenes in splits.values() for scene in scenes]
        self.assertEqual(len(all_scenes), len(set(all_scenes)))

        room_scenes = {
            "Bathroom": [f"bathroom{index:02d}" for index in range(1, 5)],
            "Bedroom": [f"bedroom{index:02d}" for index in range(1, 5)],
            "DiningRoom": [f"diningroom{index:02d}" for index in range(1, 5)],
            "Kitchen": [f"kitchen{index:02d}" for index in range(1, 5)],
            "LivingRoom": [f"livingroom{index:02d}" for index in range(1, 6)],
        }
        digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
        ranked_rooms = sorted(
            room_scenes,
            key=lambda room: digest(f"diegesis-split-v1:room:{room}"),
        )
        selected = [
            min(
                room_scenes[room],
                key=lambda scene: digest(f"diegesis-split-v1:scene:{scene}"),
            )
            for room in ranked_rooms[:4]
        ]
        self.assertEqual(splits["validation"], selected[:2])
        self.assertEqual(splits["test"], selected[2:])


if __name__ == "__main__":
    unittest.main()
