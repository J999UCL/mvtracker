import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from scripts import pointodyssey_repair_preprocessed as repair


class TargetedRepairTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source_root = self.root / "source"
        self.prepared_root = self.root / "PointOdyssey_MVTracker_v4"
        self.output_root = self.root / "PointOdyssey_MVTracker_v5"
        self.spec = repair.preprocessing.SceneSpec(
            split="train",
            scene_id="000000",
            layout="raw",
            source_sequence="synthetic",
            environment_family="synthetic",
            source_frame_start=0,
            source_frame_end=4,
            source_frame_count=4,
            source_fps=30,
        )
        self._write_fixture()

    def _write_fixture(self):
        source_view = self.source_root / "raw" / "synthetic" / "0"
        source_view.mkdir(parents=True)
        np.save(source_view / "visibility.npy", np.ones((4, 2), dtype=np.bool_))

        for split in ("train", "validation", "test"):
            (self.prepared_root / split).mkdir(parents=True)
        scene = self.prepared_root / "train" / "000000"
        view = scene / "view_0"
        view.mkdir(parents=True)

        tracks = np.asarray(
            [[[0.0, 0.0, 1.0], [4.0, 4.0, 1.0]]] * 4,
            dtype=np.float32,
        )
        intrinsics = np.eye(3, dtype=np.float32)
        extrinsics = np.zeros((4, 3, 4), dtype=np.float32)
        extrinsics[:, :3, :3] = np.eye(3, dtype=np.float32)
        depth = np.full((4, 5, 5), 2.0, dtype=np.float32)
        depth[1, 3:5, 3:5] = 1.0  # Exactly half the candidates match.
        depth[2:, 0:2, 0:2] = 1.0
        depth[2:, 3:5, 3:5] = 1.0

        np.save(scene / "tracks_3d.npy", tracks)
        np.save(view / "depth.npy", depth)
        np.save(view / "intrinsics.npy", intrinsics)
        np.save(view / "extrinsics_w2c.npy", extrinsics)
        np.save(view / "visibility.npy", np.zeros((4, 2), dtype=np.bool_))
        for frame in range(4):
            (view / f"rgba_{frame:05d}.jpg").write_bytes(b"immutable-jpeg")

        scene_metadata = {
            "schema_version": 4,
            "format": "pointodyssey_mvtracker_preprocessed",
            "split": "train",
            "scene_id": "000000",
            "source": {
                "layout": "raw",
                "sequence": "synthetic",
                "environment_family": "synthetic",
                "relative_scene_path": "raw/synthetic",
                "frame_range_half_open": [0, 4],
                "frame_count": 4,
                "fps_provenance": 30,
            },
            "output": {
                "frame_count": 4,
                "views": [0],
                "resolution_hw": [5, 5],
                "rgb": {
                    "format": "jpeg",
                    "quality": 95,
                    "resize_interpolation": "cv2.INTER_LINEAR",
                    "invalid_frame_indices": [],
                },
                "depth": {
                    "format": "npy",
                    "dtype": "float32",
                    "semantics": "optical_z_meters",
                    "invalid_value": 0.0,
                    "clipped": False,
                    "resize_interpolation": "cv2.INTER_NEAREST",
                },
                "visibility": {
                    "format": "npy",
                    "dtype": "bool",
                    "depth_track_consistency": {"tolerance_metres": 0.05},
                },
                "intrinsic_scale_xy": [1.0, 1.0],
            },
            "validation": {"failure_count": 0, "failures": []},
            "statistics": {"views": {"0": {"rgb": {}}}},
        }
        (scene / "scene.json").write_text(
            json.dumps(scene_metadata),
            encoding="utf-8",
        )
        root_report = {
            "schema_version": 4,
            "format": repair.REPORT_FORMAT,
            "status": "completed",
            "failures": [],
        }
        (self.prepared_root / "validation_report.json").write_text(
            json.dumps(root_report),
            encoding="utf-8",
        )

    def _patch_contract(self):
        return mock.patch.multiple(
            repair.preprocessing,
            VIEW_IDS=(0,),
            POINT_COUNT=2,
            OUTPUT_HEIGHT=5,
            OUTPUT_WIDTH=5,
            WINDOW_LENGTH=2,
            SOURCE_SUBROOTS={"raw": Path("raw")},
            build_scene_specs=mock.Mock(return_value=[self.spec]),
        )

    def test_repair_hard_links_immutable_data_and_restores_visibility(self):
        with self._patch_contract():
            repair.repair_preprocessed(
                self.source_root,
                self.prepared_root,
                self.output_root,
            )

        old_scene = self.prepared_root / "train" / "000000"
        new_scene = self.output_root / "train" / "000000"
        for relative in (
            Path("tracks_3d.npy"),
            Path("view_0/depth.npy"),
            Path("view_0/intrinsics.npy"),
            Path("view_0/extrinsics_w2c.npy"),
            Path("view_0/rgba_00000.jpg"),
        ):
            self.assertEqual(
                (old_scene / relative).stat().st_ino,
                (new_scene / relative).stat().st_ino,
            )

        np.testing.assert_array_equal(
            np.load(old_scene / "view_0/visibility.npy"),
            np.zeros((4, 2), dtype=np.bool_),
        )
        np.testing.assert_array_equal(
            np.load(new_scene / "view_0/visibility.npy"),
            np.ones((4, 2), dtype=np.bool_),
        )
        self.assertNotEqual(
            (old_scene / "view_0/visibility.npy").stat().st_ino,
            (new_scene / "view_0/visibility.npy").stat().st_ino,
        )

        metadata = json.loads((new_scene / "scene.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["schema_version"], 5)
        self.assertFalse(metadata["output"]["visibility"]["depth_gated"])
        self.assertEqual(
            metadata["output"]["depth_track_consistency"]["invalid_frame_indices"],
            [0],
        )
        self.assertEqual(
            metadata["output"]["window_exclusion"],
            {
                "window_length": 2,
                "invalid_frame_indices": [0],
                "reasons": {
                    "rgb_decode": [],
                    "depth_track_majority_mismatch": [0],
                },
                "total_start_count": 3,
                "excluded_start_count": 1,
                "legal_start_count": 2,
            },
        )
        self.assertEqual(
            metadata["output"]["depth_track_consistency"]["per_frame"][1][
                "failure_fraction"
            ],
            0.5,
        )
        report = json.loads(
            (self.output_root / "validation_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(report["schema_version"], 5)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(
            report["repair"]["immutable_artifact_policy"],
            "hard_links_only_no_copy_fallback",
        )
        self.assertTrue(self.prepared_root.is_dir())

    def test_hard_link_failure_has_no_copy_fallback_or_publication(self):
        with self._patch_contract(), mock.patch.object(
            repair.os,
            "link",
            side_effect=OSError("synthetic hard-link failure"),
        ):
            with self.assertRaisesRegex(OSError, "Hard-linking is required"):
                repair.repair_preprocessed(
                    self.source_root,
                    self.prepared_root,
                    self.output_root,
                )
        self.assertFalse(self.output_root.exists())
        self.assertTrue((self.prepared_root / "train/000000/tracks_3d.npy").is_file())
        self.assertFalse(list(self.root.glob(".PointOdyssey_MVTracker_v5.tmp-*")))

    def test_zero_legal_windows_abort_without_publication(self):
        scene_json = self.prepared_root / "train/000000/scene.json"
        metadata = json.loads(scene_json.read_text(encoding="utf-8"))
        metadata["output"]["rgb"]["invalid_frame_indices"] = [0, 1, 2, 3]
        scene_json.write_text(json.dumps(metadata), encoding="utf-8")
        with self._patch_contract():
            with self.assertRaisesRegex(ValueError, "no legal 2-frame windows"):
                repair.repair_preprocessed(
                    self.source_root,
                    self.prepared_root,
                    self.output_root,
                )
        self.assertFalse(self.output_root.exists())
        self.assertFalse(list(self.root.glob(".PointOdyssey_MVTracker_v5.tmp-*")))

    def test_refuses_existing_output(self):
        self.output_root.mkdir()
        with self._patch_contract():
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                repair.repair_preprocessed(
                    self.source_root,
                    self.prepared_root,
                    self.output_root,
                )

    def test_requires_clean_schema_v4_input(self):
        report_path = self.prepared_root / "validation_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["schema_version"] = 3
        report_path.write_text(json.dumps(report), encoding="utf-8")
        with self._patch_contract():
            with self.assertRaisesRegex(ValueError, "schema-v4"):
                repair.repair_preprocessed(
                    self.source_root,
                    self.prepared_root,
                    self.output_root,
                )


if __name__ == "__main__":
    unittest.main()
