"""RED contract tests for ragged scene batching.

These tests describe the intended padded-trajectory API.  They intentionally
fail until collation, loss reduction, and spatial attention support
``track_padding_mask``.
"""

import ast
import inspect
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_dataset_api():
    path = ROOT / "mvtracker" / "datasets" / "utils.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_node = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Datapoint"
    )
    collate_node = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "collate_fn"
    )
    namespace = {
        "Any": Any,
        "List": List,
        "Optional": Optional,
        "dataclass": dataclass,
        "torch": torch,
    }
    module = ast.Module(body=[class_node, collate_node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["Datapoint"], namespace["collate_fn"]


def _load_sequence_loss():
    path = ROOT / "mvtracker" / "models" / "core" / "losses.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "sequence_loss_3d"
    )

    def reduce_masked_mean(value, mask):
        return (value * mask).sum() / mask.sum().clamp_min(1e-6)

    namespace = {"torch": torch, "reduce_masked_mean": reduce_masked_mean}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["sequence_loss_3d"]


Datapoint, collate_fn = _load_dataset_api()
sequence_loss_3d = _load_sequence_loss()


def _scene(n_tracks: int):
    views, channels, height, width = 2, 3, 4, 4
    frames = 3
    return Datapoint(
        video=torch.full((views, frames, channels, height, width), float(n_tracks)),
        videodepth=torch.ones((views, frames, 1, height, width)),
        segmentation=torch.ones((views, frames, 1, height, width)),
        trajectory=torch.arange(views * n_tracks * 2, dtype=torch.float32).reshape(views, n_tracks, 2),
        trajectory_3d=torch.arange(views * n_tracks * 3, dtype=torch.float32).reshape(views, n_tracks, 3),
        visibility=torch.ones((views, n_tracks), dtype=torch.bool),
        valid=torch.ones((views, n_tracks)),
        query_points=torch.zeros((n_tracks, 4)),
        query_points_3d=torch.zeros((n_tracks, 4)),
        intrs=torch.eye(3).repeat(views, frames, 1, 1),
        extrs=torch.zeros((views, frames, 3, 4)),
        seq_name=f"scene-{n_tracks}",
    )


class RaggedCollationRedTests(unittest.TestCase):
    def test_homogeneous_views_pad_only_trajectory_axes_and_emit_mask(self):
        scenes = [_scene(n) for n in (3, 5, 8)]
        batch, _ = collate_fn([(scene, True) for scene in scenes])

        self.assertEqual(batch.video.shape[:2], (3, 2))
        self.assertEqual(batch.trajectory.shape, (3, 2, 8, 2))
        self.assertEqual(batch.visibility.shape, (3, 2, 8))
        self.assertEqual(batch.track_padding_mask.shape, (3, 8))
        torch.testing.assert_close(batch.track_padding_mask, torch.tensor([
            [False, False, False, True, True, True, True, True],
            [False, False, False, False, False, True, True, True],
            [False, False, False, False, False, False, False, False],
        ]))
        torch.testing.assert_close(batch.video[:, :, 0, 0, 0, 0], torch.tensor([[3., 3.], [5., 5.], [8., 8.]]))
        self.assertTrue(torch.equal(batch.visibility[0, :, 3:], torch.zeros(2, 5, dtype=torch.bool)))

    def test_padded_tracks_are_excluded_from_validity_mask(self):
        scenes = [_scene(n) for n in (3, 5, 8)]
        batch, _ = collate_fn([(scene, True) for scene in scenes])
        self.assertTrue(torch.equal(batch.valid[0, :, 3:], torch.zeros(2, 5)))
        self.assertTrue(torch.equal(batch.valid[1, :, 5:], torch.zeros(2, 3)))


class PerSceneLossWeightingRedTests(unittest.TestCase):
    def test_each_scene_contributes_equally_despite_different_track_counts(self):
        # Scene 0 has three valid tracks with unit error; scene 1 has eight
        # valid tracks with error three. Equal scene weighting is (1 + 3) / 2.
        gt = torch.zeros(2, 1, 8, 3)
        pred = torch.zeros_like(gt)
        gt[0, :, :3, :2] = 1
        gt[0, :, :3, 2] = 0.5  # relative to transformed zero prediction: unit error
        gt[1, :, :, :2] = 3
        gt[1, :, :, 2] = 1.5  # relative to transformed zero prediction: triple error
        valid = torch.zeros(2, 1, 8)
        valid[0, :, :3] = 1
        valid[1, :, :] = 1
        visibility = valid.clone()
        actual = sequence_loss_3d([[pred.clone()]], [gt], [visibility], [valid], gamma=1.0)
        serial_losses = [
            sequence_loss_3d(
                [[pred[index:index + 1].clone()]],
                [gt[index:index + 1]],
                [visibility[index:index + 1]],
                [valid[index:index + 1]],
                gamma=1.0,
            )
            for index in range(2)
        ]
        torch.testing.assert_close(actual, torch.stack(serial_losses).mean())


class SpatialPaddingIsolationRedTests(unittest.TestCase):
    def test_model_forward_declares_track_padding_mask(self):
        path = ROOT / "mvtracker" / "models" / "core" / "mvtracker" / "mvtracker.py"
        source = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        model = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "MVTracker")
        forward = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
        argument_names = {arg.arg for arg in forward.args.args}
        self.assertIn("track_padding_mask", argument_names)

    def test_spatial_update_path_consumes_track_padding_mask(self):
        path = ROOT / "mvtracker" / "models" / "core" / "mvtracker" / "mvtracker.py"
        source = path.read_text(encoding="utf-8")
        self.assertIn("track_padding_mask", source)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for the full-model parity contract")
class FullModelSerialBatchedParityRedTests(unittest.TestCase):
    def test_serial_and_padded_batched_outputs_match_on_valid_tracks(self):
        from mvtracker.models.core.mvtracker.mvtracker import MVTracker

        torch.manual_seed(11)
        model = MVTracker(
            fmaps_dim=32, hidden_size=64, num_heads=4, space_depth=1,
            time_depth=1, num_virtual_tracks=8, sliding_window_len=4, stride=2,
            corr_n_levels=1, corr_neighbors=1,
        ).cuda().eval()
        views, frames, height, width = 1, 4, 32, 32
        counts = (3, 5, 8)
        rgb = torch.randn(1, views, frames, 3, height, width, device="cuda")
        depth = torch.ones(1, views, frames, 1, height, width, device="cuda")
        intrs = torch.eye(3, device="cuda").view(1, 1, 1, 3, 3).expand(1, views, frames, -1, -1)
        extrs = torch.zeros(1, views, frames, 3, 4, device="cuda")
        extrs[..., :3, :3] = torch.eye(3, device="cuda")
        padded = torch.zeros(1, 8, 4, device="cuda")
        padded[:, :, 0] = 0
        padded[:, :, 1:] = torch.randn(1, 8, 3, device="cuda")
        for count in counts:
            mask = torch.zeros(1, 8, dtype=torch.bool, device="cuda")
            mask[:, count:] = True
            poisoned = padded.clone()
            poisoned[:, count:, 1:] = torch.randn_like(poisoned[:, count:, 1:]) * 10000
            batched = model(rgb, depth, padded, intrs, extrs, iters=1, track_padding_mask=mask)
            poisoned_batched = model(
                rgb, depth, poisoned, intrs, extrs, iters=1,
                track_padding_mask=mask,
            )
            serial = model(rgb, depth, padded[:, :count], intrs, extrs, iters=1)
            torch.testing.assert_close(
                batched["traj_e"][:, :, :count], serial["traj_e"], rtol=1e-3, atol=1e-3
            )
            torch.testing.assert_close(
                batched["vis_e"][:, :, :count], serial["vis_e"], rtol=1e-3, atol=1e-3
            )
            torch.testing.assert_close(
                poisoned_batched["traj_e"][:, :, :count],
                batched["traj_e"][:, :, :count],
                rtol=1e-3,
                atol=1e-3,
            )
            torch.testing.assert_close(
                poisoned_batched["vis_e"][:, :, :count],
                batched["vis_e"][:, :, :count],
                rtol=1e-3,
                atol=1e-3,
            )

    def test_three_scene_ragged_batch_matches_serial_forwards(self):
        from mvtracker.models.core.mvtracker.mvtracker import MVTracker

        torch.manual_seed(17)
        model = MVTracker(
            fmaps_dim=32, hidden_size=64, num_heads=4, space_depth=1,
            time_depth=1, num_virtual_tracks=8, sliding_window_len=4, stride=2,
            corr_n_levels=1, corr_neighbors=1,
        ).cuda().eval()
        batch_size, views, frames, height, width = 3, 1, 4, 32, 32
        counts = (3, 5, 8)
        rgb = torch.randn(
            batch_size, views, frames, 3, height, width, device="cuda"
        )
        depth = torch.ones(
            batch_size, views, frames, 1, height, width, device="cuda"
        )
        intrs = torch.eye(3, device="cuda").view(1, 1, 1, 3, 3).expand(
            batch_size, views, frames, -1, -1
        )
        extrs = torch.zeros(batch_size, views, frames, 3, 4, device="cuda")
        extrs[..., :3, :3] = torch.eye(3, device="cuda")
        queries = torch.zeros(batch_size, max(counts), 4, device="cuda")
        queries[..., 1:] = torch.randn_like(queries[..., 1:])
        padding = torch.ones(
            batch_size, max(counts), dtype=torch.bool, device="cuda"
        )
        for scene_index, count in enumerate(counts):
            padding[scene_index, :count] = False

        with torch.no_grad():
            batched = model(
                rgb, depth, queries, intrs, extrs, iters=1,
                track_padding_mask=padding,
            )
            for scene_index, count in enumerate(counts):
                serial = model(
                    rgb[scene_index:scene_index + 1],
                    depth[scene_index:scene_index + 1],
                    queries[scene_index:scene_index + 1, :count],
                    intrs[scene_index:scene_index + 1],
                    extrs[scene_index:scene_index + 1],
                    iters=1,
                )
                torch.testing.assert_close(
                    batched["traj_e"][scene_index:scene_index + 1, :, :count],
                    serial["traj_e"],
                    rtol=1e-3,
                    atol=1e-3,
                )
                torch.testing.assert_close(
                    batched["vis_e"][scene_index:scene_index + 1, :, :count],
                    serial["vis_e"],
                    rtol=1e-3,
                    atol=1e-3,
                )


if __name__ == "__main__":
    unittest.main()
