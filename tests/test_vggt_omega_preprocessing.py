import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from PIL import Image

from mvtracker.preprocessing.vggt_omega import (
    MVKubricSceneSource,
    SceneDescription,
    SceneSource,
    TapVid3DSceneSource,
    _manifest_matches,
    cleaned_depth_mask,
    infer_temporal_chunk,
    infer_temporal_chunks,
    metric_scale_from_camera_baselines,
)


def _jpeg_bytes(value: int) -> np.ndarray:
    buffer = io.BytesIO()
    Image.new("RGB", (8, 6), (value, value, value)).save(buffer, format="JPEG")
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


class VGGTOmegaPreprocessingTests(unittest.TestCase):
    def test_temporal_chunk_uses_timestamp_major_sequence_and_returns_canonical_shapes(self):
        class FakeSource(SceneSource):
            @property
            def description(self):
                return SceneDescription("fake", 2, (0, 1), (6, 8), "fingerprint")

            def load_rgb(self, view_id, frame_index):
                value = frame_index * 10 + view_id
                return Image.new("RGB", (8, 6), (value, value, value))

            def extrinsics_w2c(self, frame_indices):
                result = np.repeat(
                    np.eye(4, dtype=np.float32)[None, None],
                    len(frame_indices),
                    axis=0,
                ).repeat(2, axis=1)
                result[:, 1, 0, 3] = -1
                return result

        def fake_model_batch(_model, images, _device):
            self.assertEqual(tuple(images.shape), (1, 4, 3, 32, 32))
            observed = images[0, :, 0, 0, 0].numpy() * 255
            np.testing.assert_allclose(observed, [0, 1, 10, 11], atol=1e-5)
            depth = np.stack(
                [np.full((32, 32, 1), value, dtype=np.float32) for value in (1, 2, 3, 4)]
            )[None]
            confidence = np.ones_like(depth)
            intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, None], 4, axis=1)
            extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, None], 4, axis=1)
            extrinsics[:, 1::2, 0, 3] = -1
            return depth, confidence, intrinsics, extrinsics[:, :, :3]

        with mock.patch(
            "mvtracker.preprocessing.vggt_omega._model_batch",
            side_effect=fake_model_batch,
        ):
            result = infer_temporal_chunk(
                FakeSource(Path(".")),
                [0, 1],
                object(),
                device=torch.device("cpu"),
                image_resolution=32,
            )

        self.assertEqual(result.depth.shape, (2, 2, 6, 8))
        self.assertEqual(result.cleaned_mask.shape, result.depth.shape)
        self.assertEqual(result.intrinsics.shape, (2, 2, 3, 3))
        self.assertEqual(result.extrinsics_w2c.shape, (2, 2, 4, 4))
        np.testing.assert_allclose(result.depth.mean(axis=(-1, -2)), [[1, 2], [3, 4]])

    def test_temporal_batch_processes_each_scene_independently(self):
        class FakeSource(SceneSource):
            def __init__(self, name, value):
                super().__init__(Path("."))
                self.name = name
                self.value = value

            @property
            def description(self):
                return SceneDescription(self.name, 1, (0, 1), (6, 8), self.name)

            def load_rgb(self, view_id, frame_index):
                value = self.value + view_id
                return Image.new("RGB", (8, 6), (value, value, value))

            def extrinsics_w2c(self, frame_indices):
                result = np.repeat(
                    np.eye(4, dtype=np.float32)[None, None],
                    len(frame_indices),
                    axis=0,
                ).repeat(2, axis=1)
                result[:, 1, 0, 3] = -2
                return result

        sources = [FakeSource(f"scene-{index}", index * 10) for index in range(3)]

        def fake_model_batch(_model, images, _device):
            self.assertEqual(tuple(images.shape), (3, 2, 3, 32, 32))
            observed = images[:, :, 0, 0, 0].numpy() * 255
            np.testing.assert_allclose(observed, [[0, 1], [10, 11], [20, 21]], atol=1e-5)
            depth = np.stack(
                [
                    np.full((2, 32, 32, 1), index + 1, dtype=np.float32)
                    for index in range(3)
                ]
            )
            confidence = np.ones_like(depth)
            intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None, None], 3, axis=0)
            intrinsics = np.repeat(intrinsics, 2, axis=1)
            extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, None], 3, axis=0)
            extrinsics = np.repeat(extrinsics, 2, axis=1)
            extrinsics[:, 1, 0, 3] = -1
            return depth, confidence, intrinsics, extrinsics[:, :, :3]

        with mock.patch(
            "mvtracker.preprocessing.vggt_omega._model_batch",
            side_effect=fake_model_batch,
        ):
            batch = infer_temporal_chunks(
                sources,
                [0],
                object(),
                device=torch.device("cpu"),
                image_resolution=32,
            )

        self.assertEqual(len(batch.scenes), 3)
        self.assertEqual(batch.timings.scene_count, 3)
        self.assertEqual(batch.timings.image_count, 6)
        for index, result in enumerate(batch.scenes):
            self.assertEqual(result.depth.shape, (1, 2, 6, 8))
            np.testing.assert_allclose(result.depth.mean(), 2 * (index + 1))
            self.assertEqual(result.scale, 2.0)

    def test_metric_scale_uses_corresponding_camera_baselines(self):
        predicted = np.repeat(np.eye(4, dtype=np.float32)[None], 3, axis=0)
        known = predicted.copy()
        predicted[:, 0, 3] = [0, -1, -2]
        known[:, 0, 3] = [0, -2, -4]

        scale, residual = metric_scale_from_camera_baselines(predicted, known)

        self.assertEqual(scale, 2.0)
        self.assertEqual(residual, 0.0)

    def test_clean_mask_combines_percentile_and_depth_edges(self):
        depth = np.ones((2, 8, 8), dtype=np.float32)
        confidence = np.arange(1, depth.size + 1, dtype=np.float32).reshape(depth.shape)
        depth[:, 4, 4] = 2.0

        mask = cleaned_depth_mask(depth, confidence)

        threshold = np.percentile(confidence, 20)
        self.assertTrue(np.all(~mask[confidence < threshold]))
        self.assertFalse(mask[:, 4, 4].any())
        self.assertGreater(mask.mean(), 0.5)

    def test_manifest_only_matches_complete_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {"format": "mvtracker_estimated_depth", "schema_version": 1}
            manifest = {**expected, "complete": False, "arrays": {"depth.npy": {}}}
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "depth.npy").touch()
            self.assertFalse(_manifest_matches(root, expected))

            manifest["complete"] = True
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            self.assertTrue(_manifest_matches(root, expected))
            (root / "depth.npy").unlink()
            self.assertFalse(_manifest_matches(root, expected))

    def test_tapvid_source_reads_jpegs_and_cameras(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scene"
            root.mkdir()
            np.save(root / "tracks_xyz.npy", np.zeros((2, 1, 3), dtype=np.float32))
            for view in (0, 1):
                view_root = root / str(view)
                view_root.mkdir()
                images = np.empty(2, dtype=object)
                images[:] = [_jpeg_bytes(view * 20 + frame) for frame in range(2)]
                np.save(view_root / "images_jpeg_bytes.npy", images, allow_pickle=True)
                extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
                extrinsics[:, 0, 3] = -view
                np.save(view_root / "extrinsics_w2c.npy", extrinsics)

            source = TapVid3DSceneSource(root)

            self.assertEqual(source.description.view_ids, (0, 1))
            self.assertEqual(source.description.resolution_hw, (6, 8))
            self.assertEqual(source.load_rgb(1, 1).size, (8, 6))
            self.assertEqual(source.extrinsics_w2c([0, 1]).shape, (2, 2, 4, 4))

    def test_kubric_source_matches_cv_camera_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scene"
            root.mkdir()
            for view in (0, 1):
                view_root = root / f"view_{view}"
                view_root.mkdir()
                for frame in range(2):
                    Image.new("RGBA", (8, 6), (20, 30, 40, 255)).save(
                        view_root / f"rgba_{frame:05d}.png"
                    )
                metadata = {
                    "camera": {
                        "positions": [[float(view), 0.0, 0.0]] * 2,
                        "quaternions": [[1.0, 0.0, 0.0, 0.0]] * 2,
                    }
                }
                (view_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            source = MVKubricSceneSource(root)
            extrinsics = source.extrinsics_w2c([0])

            self.assertEqual(source.description.view_ids, (0, 1))
            self.assertEqual(extrinsics.shape, (1, 2, 4, 4))
            np.testing.assert_allclose(extrinsics[0, 1, :3, 3], [-1, 0, 0])
            np.testing.assert_allclose(np.diag(extrinsics[0, 1, :3, :3]), [1, -1, -1])


if __name__ == "__main__":
    unittest.main()
