import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from mvtracker.preprocessing.waymo_rerun import (
    fit_rigid_transform,
    load_tapvid3d_annotation,
    match_annotation_frames,
    points_inside_boxes,
    select_tracks,
    time_colors,
    trajectory_segments,
    transform_points,
    voxel_downsample,
)


def jpeg(value: int) -> bytes:
    image = np.full((32, 48, 3), value, dtype=np.uint8)
    handle = io.BytesIO()
    Image.fromarray(image).save(handle, format="JPEG", quality=95)
    return handle.getvalue()


class WaymoRerunTest(unittest.TestCase):
    def test_load_annotation_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.npz"
            np.savez(
                path,
                images_jpeg_bytes=np.asarray([jpeg(10), jpeg(20)]),
                queries_xyt=np.zeros((3, 3)),
                tracks_XYZ=np.zeros((2, 3, 3)),
                visibility=np.ones((2, 3), dtype=bool),
                fx_fy_cx_cy=np.ones(4),
                extrinsics_w2c=np.repeat(np.eye(4)[None], 2, axis=0),
            )
            loaded = load_tapvid3d_annotation(path)
        self.assertEqual(loaded.tracks_xyz.shape, (2, 3, 3))
        self.assertEqual(len(loaded.images_jpeg), 2)

    def test_exact_frame_and_camera_matching(self):
        values = [jpeg(value) for value in range(8)]
        camera, indices, score = match_annotation_frames(
            [values[1], values[4], values[7]],
            {1: [jpeg(100)] * 8, 2: values, 3: [jpeg(200)] * 8},
        )
        self.assertEqual(camera, 2)
        np.testing.assert_array_equal(indices, [1, 4, 7])
        self.assertEqual(score, 0.0)

    def test_rigid_alignment(self):
        source = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        angle = np.pi / 2
        rotation = np.asarray([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        target = source @ rotation.T + [3, 4, 5]
        transform, rmse = fit_rigid_transform(source, target)
        np.testing.assert_allclose(transform_points(transform, source), target, atol=1e-6)
        self.assertLess(rmse, 1e-6)

        camera_tracks = np.asarray([[1.0, 2.0, 3.0]])
        camera_to_world = np.eye(4)
        camera_to_world[:3, 3] = [10, 20, 30]
        np.testing.assert_allclose(transform_points(camera_to_world, camera_tracks), [[11, 22, 33]])

    def test_boxes_voxels_tracks_and_colors(self):
        points = np.asarray([[0, 0, 0], [0.4, 0, 0], [2, 0, 0]], dtype=float)
        inside = points_inside_boxes(points, [(0, 0, 0, 1, 1, 1, 0)])
        np.testing.assert_array_equal(inside, [True, True, False])
        down_points, down_colors = voxel_downsample(points, np.arange(9).reshape(3, 3), 1.0)
        self.assertEqual(len(down_points), 2)
        self.assertEqual(len(down_colors), 2)

        tracks = np.zeros((4, 3, 3), dtype=np.float32)
        tracks[:, 0, 0] = [0, 1, 2, 3]
        tracks[:, 1, 1] = [0, 0.5, 1.0, 1.5]
        visibility = np.ones((4, 3), dtype=bool)
        selected = select_tracks(tracks, visibility, 2)
        self.assertEqual(set(selected.tolist()), {0, 1})
        segments, colors = trajectory_segments(tracks, selected)
        self.assertEqual(segments.shape, (6, 2, 3))
        self.assertEqual(colors.shape, (6, 3))
        palette = time_colors(3)
        np.testing.assert_array_equal(palette[0], [0, 220, 255])
        np.testing.assert_array_equal(palette[-1], [255, 40, 20])


if __name__ == "__main__":
    unittest.main()
