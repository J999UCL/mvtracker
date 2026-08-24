import csv
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from mvtracker.preprocessing import syn4d_conversion


def _write_one_frame_scene(root: Path) -> Path:
    scene_root = root / "lab_bald"
    (scene_root / "mp4").mkdir(parents=True)
    camera_root = scene_root / "ground_truth" / "meta_exr_csv"
    camera_root.mkdir(parents=True)
    for view in range(8):
        (scene_root / "mp4" / f"seq_000000_{view}.mp4").touch()
        depth_root = scene_root / "exr_layers" / "depth" / f"seq_000000_{view}"
        depth_root.mkdir(parents=True)
        (depth_root / "seq_000000_000000_depth.exr").touch()
        with (camera_root / f"seq_000000_{view}_camera.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "name",
                    "focal_length",
                    "sensor_width",
                    "sensor_height",
                    "yaw",
                    "pitch",
                    "roll",
                    "x",
                    "y",
                    "z",
                ),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "name": "seq_000000_000000.png",
                    "focal_length": "50",
                    "sensor_width": "100",
                    "sensor_height": "100",
                    "yaw": "0",
                    "pitch": "0",
                    "roll": "0",
                    "x": "0",
                    "y": "0",
                    "z": "0",
                }
            )
    return scene_root


class Syn4DConversionTests(unittest.TestCase):
    def test_official_reader_constructor_uses_exact_runtime_contract(self):
        calls = []
        items = [{"idx": np.array([0])}]

        class FakeDataset:
            annotation_paths = ["/scene/seq_000000_0_camera.csv"]

            def __getitem__(self, index):
                self.requested_index = index
                return items

        def construct(**kwargs):
            calls.append(kwargs)
            return FakeDataset()

        module = SimpleNamespace(Syn4D_Track=construct)
        result = syn4d_conversion._official_sequence_items(
            module,
            scene_root=Path("/data/lab_bald"),
            metadata_root=Path("/metadata"),
            sequence_base="seq_000000",
            frame_count=1,
        )

        self.assertIs(result[0], items[0])
        self.assertEqual(len(calls), 1)
        kwargs = calls[0]
        self.assertEqual(kwargs["dataset_root"], "/data")
        self.assertEqual(kwargs["metadata_root"], "/metadata")
        self.assertIsNone(kwargs["fallback_metadata_root"])
        self.assertEqual(kwargs["scene_name_list"], ["lab_bald"])
        self.assertEqual(kwargs["track_query_idx"], 0)
        self.assertEqual(kwargs["S"], 1)
        self.assertEqual(kwargs["N"], 65_536)
        self.assertEqual(kwargs["strides"], [1])
        self.assertEqual(kwargs["rgb_source"], "mp4")
        self.assertEqual(kwargs["tracking_format"], "safetensor")
        self.assertEqual(kwargs["resolution"], ((878, 494), 1))
        self.assertFalse(kwargs["allow_repeat"])

    def test_query_selection_applies_quarter_before_cap(self):
        valid = np.ones((40, 40), dtype=bool)
        track = np.zeros((40, 40, 3), dtype=np.float32)
        selected, xs, ys, quarter_count = syn4d_conversion._quarter_query_pixels(
            valid, track, seed=3, cap=100
        )
        self.assertEqual(quarter_count, 400)
        self.assertEqual(selected.shape, (100,))
        self.assertEqual(np.unique(selected).size, 100)
        np.testing.assert_array_equal(selected, ys * 40 + xs)

        valid.reshape(-1)[399:] = False
        selected, _, _, quarter_count = syn4d_conversion._quarter_query_pixels(
            valid, track, seed=3, cap=100
        )
        self.assertEqual(quarter_count, 99)
        self.assertEqual(selected.shape, (99,))
        self.assertEqual(np.unique(selected).size, 99)

    def test_single_sequence_conversion_fills_direct_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene_root = _write_one_frame_scene(root)
            primary = root / "primary"
            clothing = root / "clothing"
            visualizer = root / "visualizer"
            output = root / "output"
            primary.mkdir()
            clothing.mkdir()
            visualizer.mkdir()

            dense_track = np.zeros((494, 878, 3), dtype=np.float32)
            dense_track[..., 2] = 2.0
            camera_to_world = np.eye(4, dtype=np.float32)
            camera_to_world[0, 3] = 100.0  # must not be applied again
            query_valid = np.zeros((494, 878), dtype=bool)
            query_valid.reshape(-1)[:400] = True
            official_items = [
                {
                    "track": dense_track,
                    "track_valid_mask": query_valid,
                    "camera_pose": camera_to_world,
                }
            ]
            progress = []
            depth_calls = []
            rgb_calls = []
            jpeg_threads = []
            write_jpeg_store = syn4d_conversion._write_jpeg_store

            def fake_depth(paths, *, official_module, workers):
                depth_calls.append((tuple(paths), workers, threading.get_ident()))
                return np.full((1, 4, 6), 200.0, dtype=np.float32)

            def fake_rgb(path, *, frame_count, device):
                rgb_calls.append((path, frame_count, device))
                return np.zeros((1, 384, 683, 3), dtype=np.uint8)

            def fake_visibility(tracks, valid, depths, intrinsics, extrinsics, **kwargs):
                self.assertEqual(depths.shape, (1, 1, 4, 6))
                return np.ones((1, 1, tracks.shape[1]), dtype=bool)

            def fake_resize(depths, **kwargs):
                return np.broadcast_to(
                    depths[:, :1, :1], (len(depths), 384, 683)
                ).copy()

            def observed_jpeg_store(frames, destination, *, workers):
                jpeg_threads.append((workers, threading.get_ident()))
                write_jpeg_store(frames, destination, workers=workers)

            with (
                patch.object(
                    syn4d_conversion,
                    "_import_official_syn4d",
                    return_value=SimpleNamespace(),
                ),
                patch.object(
                    syn4d_conversion,
                    "_official_sequence_items",
                    return_value=official_items,
                ),
                patch.object(
                    syn4d_conversion,
                    "_probe_video",
                    return_value={"width": 6, "height": 4, "frames": 1, "fps": 24.0},
                ),
                patch.object(
                    syn4d_conversion, "_read_view_depths_cm", side_effect=fake_depth
                ),
                patch.object(
                    syn4d_conversion, "_decode_rgb_video_dali", side_effect=fake_rgb
                ),
                patch.object(
                    syn4d_conversion, "compute_depth_visibility", side_effect=fake_visibility
                ),
                patch.object(
                    syn4d_conversion,
                    "resize_depth_validity_weighted",
                    side_effect=fake_resize,
                ),
                patch.object(
                    syn4d_conversion,
                    "_write_jpeg_store",
                    side_effect=observed_jpeg_store,
                ),
            ):
                result = syn4d_conversion.convert_syn4d_sequence(
                    scene_root,
                    primary,
                    output,
                    official_visualizer_root=visualizer,
                    sequence="seq_000000",
                    device="cpu",
                    progress=progress.append,
                )

            self.assertEqual(result["scene"], "lab_bald")
            self.assertEqual(result["sequence"], "seq_000000")
            destination = Path(result["output_path"])
            manifest = json.loads((destination / "manifest.json").read_text())
            tracks = np.load(destination / "tracks_xyz.npy", mmap_mode="r")
            valid = np.load(destination / "track_valid.npy", mmap_mode="r")
            queries = np.load(destination / "queries_xytv.npy", mmap_mode="r")
            path_length = np.load(destination / "motion_path_length.npy", mmap_mode="r")
            depth = np.load(destination / "0" / "depth.npy", mmap_mode="r")
            visibility = np.load(destination / "7" / "visibility.npy", mmap_mode="r")
            jpeg_offsets = np.load(destination / "view_0" / "jpeg_offsets.npy")

            self.assertEqual(manifest["queries"], "cache_pixel_xytv")
            self.assertEqual(manifest["tracks"], 100)
            self.assertEqual(manifest["views"], 8)
            self.assertEqual(tracks.shape, (1, 100, 3))
            np.testing.assert_allclose(tracks[0, 0], [0.0, 0.0, 2.0])
            self.assertTrue(valid.all())
            self.assertTrue(visibility.all())
            self.assertTrue(np.all(queries[:, 2:] == 0.0))
            self.assertTrue(np.all(path_length == 0.0))
            self.assertEqual(depth.dtype, np.float32)
            self.assertEqual(depth.shape, (1, 384, 683))
            np.testing.assert_allclose(depth[0, 0, 0], 2.0)
            np.testing.assert_array_equal(jpeg_offsets.shape, (2,))
            self.assertGreater((destination / "view_0" / "jpeg_bytes.bin").stat().st_size, 0)
            self.assertFalse((destination / "0" / "rgb.npy").exists())
            self.assertEqual(len(depth_calls), 8)
            self.assertEqual(len(rgb_calls), 8)
            self.assertTrue(all(workers <= 3 for _, workers, _ in depth_calls))
            self.assertTrue(all(workers <= 3 for workers, _ in jpeg_threads))
            main_thread = threading.get_ident()
            self.assertTrue(all(thread != main_thread for _, _, thread in depth_calls))
            self.assertTrue(all(thread != main_thread for _, thread in jpeg_threads))
            self.assertEqual(
                [event["stage"] for event in progress],
                [
                    "sequence_started",
                    "tracks_ready",
                    *("view_ready" for _ in range(8)),
                    "sequence_complete",
                ],
            )


if __name__ == "__main__":
    unittest.main()
