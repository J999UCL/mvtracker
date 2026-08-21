import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mvtracker.preprocessing.syn4d import (
    EntityMesh,
    SurfaceCandidate,
    apply_syn4d_body_transform,
    build_surface_bank,
    camera_from_syn4d_row,
    compute_depth_visibility,
    create_sequence_cache,
    depth_centimetres_to_metres,
    dynamic_surface_coordinates,
    finalize_sequence_cache,
    motion_path_length,
    reconstruct_surface_tracks,
    resize_depth_validity_weighted,
    resize_intrinsics,
    sample_candidate_quarter,
    sequence_dependencies,
    syn4d_actor_world_vertices,
    syn4d_moving_object_world_vertices,
    syn4d_static_object_world_vertices,
    temple_group_dependencies,
    tile_syn4d_object_animation,
)


class Syn4DPreprocessingTests(unittest.TestCase):
    def test_single_sequence_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            mapping = Path(directory) / "sequence_to_asset_mapping.csv"
            with mapping.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("scene", "sequence_name", "asset_type", "asset"),
                )
                writer.writeheader()
                for view in range(8):
                    common = {"scene": "lab_bald", "sequence_name": f"seq_000000_{view}"}
                    writer.writerow(common | {"asset_type": "bedlam2_body", "asset": "subject_M_1234"})
                    for index in range(3):
                        writer.writerow(common | {"asset_type": "objaverse_object", "asset": f"group{index}_object{index}"})

            dependencies = sequence_dependencies(
                mapping, scene="lab_bald", sequence_base="seq_000000"
            )

        self.assertEqual(dependencies.body_motion, "subject_M_1234")
        self.assertEqual(dependencies.clothing_member, "subject_M/1234/1234.npz")
        self.assertEqual(len(dependencies.objects), 3)

    def test_temple_group_dependency_plan_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            mapping = Path(directory) / "sequence_to_asset_mapping.csv"
            with mapping.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("scene", "sequence_name", "asset_type", "asset"),
                )
                writer.writeheader()
                for sequence_index in range(20):
                    for view in range(8):
                        common = {
                            "scene": "temple_group",
                            "sequence_name": f"seq_{sequence_index:06d}_{view}",
                        }
                        writer.writerow(
                            common
                            | {
                                "asset_type": "bedlam2_body",
                                "asset": f"it_{sequence_index:04d}_M_{2000 + sequence_index}",
                            }
                        )
                        for object_index in range(3):
                            writer.writerow(
                                common
                                | {
                                    "asset_type": "objaverse_object",
                                    "asset": (
                                        f"{sequence_index:03d}-{object_index:03d}_"
                                        f"object{sequence_index:02d}{object_index}"
                                    ),
                                }
                            )
                writer.writerow(
                    {
                        "scene": "hallway",
                        "sequence_name": "seq_000000_0",
                        "asset_type": "bedlam2_body",
                        "asset": "unused_M_9999",
                    }
                )

            plan = temple_group_dependencies(mapping)

        self.assertEqual(len(plan.sequences), 20)
        self.assertEqual(len(plan.body_motions), 20)
        self.assertEqual(len(plan.clothing_members), 20)
        self.assertEqual(len(plan.objects), 60)
        self.assertEqual(plan.sequences[0].clothing_member, "it_0000_M/2000/2000.npz")
        self.assertEqual(
            plan.sequences[-1].objects,
            (
                ("019-000", "object190"),
                ("019-001", "object191"),
                ("019-002", "object192"),
            ),
        )

    def test_signed_surface_coordinates_reconstruct_without_quantization(self):
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        points = np.array([[0.25, 0.5, 0.05]], dtype=np.float32)
        barycentric, signed_offset, usable = dynamic_surface_coordinates(
            points, np.array([0]), vertices, faces
        )
        np.testing.assert_allclose(barycentric, [[0.25, 0.25, 0.5]], atol=1e-7)
        np.testing.assert_allclose(signed_offset, [0.05], atol=1e-7)
        np.testing.assert_array_equal(usable, [True])

        candidates = [
            SurfaceCandidate(
                entity_name="body_00",
                face_id=0,
                barycentric=tuple(float(value) for value in barycentric[0]),
                normal_offset_m=float(signed_offset[0]),
                env_world_xyz=(0.0, 0.0, 0.0),
                query_xytv=(3.0, 4.0, 0.0, 2.0),
            ),
            SurfaceCandidate(
                entity_name=None,
                face_id=-1,
                barycentric=(0.0, 0.0, 0.0),
                normal_offset_m=0.0,
                env_world_xyz=(5.0, 6.0, 7.0),
                query_xytv=(1.0, 2.0, 0.0, 0.0),
            ),
        ]
        bank = build_surface_bank(candidates, count=2)
        animated = np.stack([vertices, vertices + np.array([1.0, 0.0, 0.0])])
        tracks, valid = reconstruct_surface_tracks(
            bank,
            {"body_00": EntityMesh(animated, faces)},
            point_batch_size=1,
            frame_batch_size=1,
        )
        body_index = int(np.flatnonzero(bank.entity_id == 0)[0])
        environment_index = int(np.flatnonzero(bank.entity_id == -1)[0])
        np.testing.assert_allclose(tracks[:, body_index], [points[0], points[0] + [1, 0, 0]])
        np.testing.assert_allclose(tracks[:, environment_index], [[5, 6, 7], [5, 6, 7]])
        self.assertTrue(valid.all())
        np.testing.assert_allclose(motion_path_length(tracks, valid), [1.0, 0.0])

    def test_body_and_clothing_transforms_match_official_order(self):
        vertices = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
        trans = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        pose = np.diag([2.0, 3.0, 4.0]).astype(np.float32)
        translation = np.array([[[10.0, 20.0, 30.0]]], dtype=np.float32).reshape(1, 3)

        body = apply_syn4d_body_transform(
            vertices, pose, translation, trans=trans
        )
        clothing = apply_syn4d_body_transform(
            vertices, pose, translation, trans=None
        )

        expected_body = vertices @ trans.T @ pose.T + translation[:, None, :]
        expected_clothing = vertices @ pose.T + translation[:, None, :]
        np.testing.assert_allclose(body, expected_body)
        np.testing.assert_allclose(clothing, expected_clothing)

        body_world = syn4d_actor_world_vertices(
            vertices,
            yaw_degrees=0.0,
            pitch_degrees=0.0,
            roll_degrees=0.0,
            position_cm=(100.0, 200.0, 300.0),
            global_translation_m=(4.0, 5.0, 6.0),
            naked_body=True,
        )
        clothing_world = syn4d_actor_world_vertices(
            vertices,
            yaw_degrees=0.0,
            pitch_degrees=0.0,
            roll_degrees=0.0,
            position_cm=(100.0, 200.0, 300.0),
            global_translation_m=(4.0, 5.0, 6.0),
            naked_body=False,
        )
        self.assertFalse(np.array_equal(body_world, clothing_world))
        np.testing.assert_allclose(body_world[0, 0], [6.0, 5.0, 12.0])
        np.testing.assert_allclose(clothing_world[0, 0], [6.0, 10.0, 11.0])

    def test_object_transforms_match_pinned_static_and_keyframe_paths(self):
        source = np.array(
            [
                [[1.0, 2.0, 3.0]],
                [[4.0, 5.0, 6.0]],
            ],
            dtype=np.float32,
        )
        tiled = tile_syn4d_object_animation(source, frame_count=3)
        np.testing.assert_allclose(
            tiled[:, 0], [[4.0, -5.0, 6.0], [1.0, -2.0, 3.0], [4.0, -5.0, 6.0]]
        )

        static_world = syn4d_static_object_world_vertices(
            source,
            frame_count=3,
            yaw_degrees=0.0,
            pitch_degrees=0.0,
            roll_degrees=0.0,
            position_cm=(100.0, 200.0, 300.0),
            global_translation_m=(10.0, 20.0, 30.0),
            scale=2.0,
            shift_cm=100.0,
        )
        np.testing.assert_allclose(static_world[0, 0], [19.0, 12.0, 45.0])
        np.testing.assert_allclose(static_world[-1, 0], [19.0, 13.0, 45.0])

        keyframe_poses = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
            ],
            dtype=np.float32,
        )
        moving_world = syn4d_moving_object_world_vertices(
            source,
            keyframe_frames=np.array([0, 2]),
            keyframe_poses=keyframe_poses,
            frame_count=3,
            global_translation_m=(1.0, 2.0, 3.0),
        )
        np.testing.assert_allclose(moving_world[0, 0], [5.0, -3.0, 9.0])
        np.testing.assert_allclose(moving_world[1, 0], [4.0, -2.0, 9.0])
        np.testing.assert_allclose(moving_world[2, 0], [15.0, -13.0, 21.0])

    def test_float_depth_camera_resize_and_three_by_three_visibility(self):
        depth_cm = np.array([[200.0, np.nan, -1.0]], dtype=np.float32)
        converted = depth_centimetres_to_metres(depth_cm)
        np.testing.assert_array_equal(converted, [[2.0, 0.0, 0.0]])
        self.assertEqual(converted.dtype, np.float32)

        intrinsics = np.array(
            [[100.0, 0.0, 20.0], [0.0, 200.0, 30.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        resized_intrinsics = resize_intrinsics(
            intrinsics,
            source_width=100,
            source_height=100,
            target_width=50,
            target_height=25,
        )
        np.testing.assert_allclose(
            resized_intrinsics,
            [[50.0, 0.0, 10.0], [0.0, 50.0, 7.5], [0.0, 0.0, 1.0]],
        )

        tracks = np.array([[[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]], dtype=np.float32)
        track_valid = np.array([[True, False]])
        depths = np.zeros((1, 1, 5, 5), dtype=np.float32)
        depths[0, 0, 2, 3] = 2.0  # one pixel beside the projected center
        cameras = np.array([[[[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]]]])
        extrinsics = np.broadcast_to(np.eye(4, dtype=np.float32), (1, 1, 4, 4)).copy()
        visibility = compute_depth_visibility(
            tracks,
            track_valid,
            depths,
            cameras,
            extrinsics,
            point_batch_size=1,
            frame_batch_size=1,
        )
        np.testing.assert_array_equal(visibility, [[[True, False]]])

    def test_weighted_depth_resize_does_not_blend_invalid_zero(self):
        depth = np.array([[[1.0, 0.0]]], dtype=np.float32)
        resized = resize_depth_validity_weighted(depth, height=1, width=4)
        self.assertEqual(resized.dtype, np.float32)
        np.testing.assert_allclose(resized[0, 0, :3], [1.0, 1.0, 1.0])
        self.assertEqual(float(resized[0, 0, 3]), 0.0)

    def test_camera_csv_conversion_and_direct_cache_contract(self):
        row = {
            "focal_length": "50",
            "sensor_width": "100",
            "sensor_height": "50",
            "yaw": "0",
            "pitch": "0",
            "roll": "0",
            "x": "100",
            "y": "200",
            "z": "300",
        }
        intrinsics, world_to_camera = camera_from_syn4d_row(
            row, source_width=200, source_height=100
        )
        np.testing.assert_allclose(np.diag(intrinsics), [100.0, 100.0, 1.0])
        np.testing.assert_allclose(
            np.linalg.inv(world_to_camera)[:3, 3], [1.0, 2.0, 3.0], atol=1e-6
        )

        with tempfile.TemporaryDirectory() as directory:
            writer = create_sequence_cache(
                Path(directory) / "temple_group__seq_000000",
                scene="temple_group",
                sequence_base="seq_000000",
                frame_count=2,
                track_count=3,
                height=4,
                width=5,
                view_count=2,
            )
            writer.array("tracks_xyz")[:] = 1.0
            writer.view_array(1, "depth")[:] = 2.0
            for view in range(2):
                (writer.destination / f"view_{view}" / "jpeg_bytes.bin").write_bytes(b"ab")
                np.save(
                    writer.destination / f"view_{view}" / "jpeg_offsets.npy",
                    np.array([0, 1, 2], dtype=np.int64),
                )
            manifest_path = finalize_sequence_cache(
                writer, fps=24.0, source_width=10, source_height=8
            )
            manifest = json.loads(manifest_path.read_text())

        self.assertEqual(manifest["format"], "syn4d-tapvid-mv")
        self.assertEqual(manifest["cache_resolution"], [4, 5])
        self.assertEqual(manifest["depth"], "float32_optical_z_metres_zero_invalid")
        self.assertEqual(manifest["rgb"], "jpeg_quality_95_rgb")
        self.assertNotIn("version", manifest)
        self.assertEqual(manifest["views"], 2)

    def test_candidate_quarter_is_exact_without_replacement(self):
        candidates = [
            SurfaceCandidate(
                entity_name=None,
                face_id=-1,
                barycentric=(0.0, 0.0, 0.0),
                normal_offset_m=0.0,
                env_world_xyz=(float(index), 0.0, 0.0),
                query_xytv=(float(index), 0.0, 0.0, 0.0),
            )
            for index in range(20)
        ]
        selected = sample_candidate_quarter(candidates, np.random.default_rng(7))
        self.assertEqual(len(selected), 5)
        self.assertEqual(len({candidate.identity for candidate in selected}), 5)


if __name__ == "__main__":
    unittest.main()
