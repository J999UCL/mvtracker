import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from mvtracker.datasets.kubric_gpu_dataset import GpuDecodedKubricMultiViewDataset
from mvtracker.datasets.kubric_metadata_index import build_kubric_metadata_index


class KubricGpuDatasetTests(unittest.TestCase):
    def test_plan_is_metadata_only_and_materialization_reads_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "900"
            view = scene / "view_0"
            view.mkdir(parents=True)
            tracks = np.zeros((2, 4, 3), dtype=np.float32)
            tracks[..., 2] = -2
            np.savez(scene / "tracks_3d.npz", tracks_3d=tracks)
            (scene / "scene.json").write_text("{}", encoding="utf-8")
            metadata = {
                "camera": {
                    "K": np.eye(3).tolist(),
                    "positions": [[0, 0, 0]] * 2,
                    "quaternions": [[1, 0, 0, 0]] * 2,
                    "sensor_width": 1.0,
                    "focal_length": 1.0,
                },
                "metadata": {"resolution": [4, 3]},
            }
            (view / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            np.savez(
                view / "tracks_2d.npz",
                tracks_2d=np.zeros((2, 4, 2), dtype=np.float32),
                occlusion=np.zeros((2, 4), dtype=np.bool_),
            )
            for frame in range(2):
                Image.fromarray(np.full((3, 4, 4), frame, dtype=np.uint8)).save(
                    view / f"rgba_{frame:05d}.png"
                )
                Image.fromarray(np.ones((3, 4), dtype=np.float32)).save(
                    view / f"depth_{frame:05d}.tiff"
                )
            index_root = build_kubric_metadata_index(root).parent
            dataset = GpuDecodedKubricMultiViewDataset(
                data_root=str(root),
                metadata_index_root=str(index_root),
                seq_len=2,
                num_views=1,
                traj_per_sample=1,
                ratio_dynamic=0,
                ratio_very_dynamic=0,
                seed=72,
            )

            with mock.patch.object(Path, "read_bytes", side_effect=AssertionError):
                plan = dataset.plan_sample(0)
            self.assertIsNotNone(plan)
            self.assertEqual(plan.track_count, len(plan.selected_track_indices))
            self.assertEqual(len(plan.rgb_sources), len(plan.depth_sources))
            self.assertEqual(plan.image_codec, "nvimagecodec")
            sample, valid = dataset.materialize_sample(plan)
            self.assertTrue(valid)
            self.assertEqual(len(sample.jpeg_bytes), 2)
            self.assertEqual(len(sample.depth_bytes), 2)

    def test_workers_return_lossless_encoded_png_and_tiff(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene = root / "900"
            view = scene / "view_0"
            view.mkdir(parents=True)
            tracks = np.zeros((2, 4, 3), dtype=np.float32)
            tracks[..., 2] = -2
            np.savez(scene / "tracks_3d.npz", tracks_3d=tracks)
            (scene / "scene.json").write_text("{}", encoding="utf-8")
            metadata = {
                "camera": {
                    "K": np.eye(3).tolist(),
                    "positions": [[0, 0, 0]] * 2,
                    "quaternions": [[1, 0, 0, 0]] * 2,
                    "sensor_width": 1.0,
                    "focal_length": 1.0,
                },
                "metadata": {"resolution": [4, 3]},
            }
            (view / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            np.savez(
                view / "tracks_2d.npz",
                tracks_2d=np.zeros((2, 4, 2), dtype=np.float32),
                occlusion=np.zeros((2, 4), dtype=np.bool_),
            )
            for frame in range(2):
                Image.fromarray(np.full((3, 4, 4), frame, dtype=np.uint8)).save(
                    view / f"rgba_{frame:05d}.png"
                )
                Image.fromarray(np.ones((3, 4), dtype=np.float32)).save(
                    view / f"depth_{frame:05d}.tiff"
                )
            index_root = build_kubric_metadata_index(root).parent
            dataset = GpuDecodedKubricMultiViewDataset(
                data_root=str(root),
                metadata_index_root=str(index_root),
                seq_len=2,
                num_views=1,
                traj_per_sample=1,
                ratio_dynamic=0,
                ratio_very_dynamic=0,
                seed=72,
            )

            sample, valid = dataset[0]

            self.assertTrue(valid)
            self.assertEqual(sample.image_codec, "nvimagecodec")
            self.assertIsNone(sample.depth)
            self.assertEqual(len(sample.jpeg_bytes), 2)
            self.assertEqual(len(sample.depth_bytes), 2)
            self.assertTrue(sample.jpeg_bytes[0].startswith(b"\x89PNG"))
            self.assertIn(sample.depth_bytes[0][:2], (b"II", b"MM"))


if __name__ == "__main__":
    unittest.main()
