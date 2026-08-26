import unittest
from types import SimpleNamespace

import torch

from tools.audit_recipe_optimizer_steps import _concentration


class RecipeOptimizerStepAuditTests(unittest.TestCase):
    def test_per_track_contributions_reconstruct_scene_loss(self):
        frames, tracks = 12, 10
        target = torch.zeros((1, frames, tracks, 3), dtype=torch.float32)
        target[..., 2] = 0.1
        prediction = torch.zeros_like(target)
        prediction[..., 0] = torch.arange(1, tracks + 1, dtype=torch.float32)
        expected = float(torch.arange(1, tracks + 1).float().mean() / 3.0)
        batch = SimpleNamespace(
            trajectory_3d=target,
            query_points_3d=torch.cat(
                (
                    torch.zeros((1, tracks, 1)),
                    target[:, 0],
                ),
                dim=-1,
            ),
            valid=torch.ones((1, frames, tracks)),
            track_upscaling_factor=torch.ones(1),
        )
        output = {
            "training_trace": {
                "coordinates": [prediction],
                "visibility_logits": [],
            },
            "scene_losses": {"flow": torch.tensor([expected])},
        }
        cfg = SimpleNamespace(
            model=SimpleNamespace(sliding_window_len=frames),
            trainer=SimpleNamespace(train_iters=1, gamma=0.8),
        )

        result = _concentration(batch, output, cfg)

        self.assertAlmostEqual(result["total_trajectory_loss"], expected, places=6)
        self.assertAlmostEqual(result["scene_loss_difference"], 0.0, places=6)
        self.assertGreater(result["worst_10_percent_share"], 0.1)


if __name__ == "__main__":
    unittest.main()
