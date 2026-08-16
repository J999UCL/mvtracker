import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_MODULE_PATH = REPO_ROOT / "mvtracker" / "datasets" / "kubric_metadata_index.py"
LOADER_PATH = REPO_ROOT / "mvtracker" / "datasets" / "kubric_multiview_dataset.py"


def _load_index_module():
    spec = importlib.util.spec_from_file_location("kubric_metadata_index", INDEX_MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INDEX = _load_index_module()


def _indexed_view_loader(read_calls):
    source = LOADER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(LOADER_PATH))
    dataset_class = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    method = next(
        node for node in dataset_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "getitem_indexed_views"
    )
    method.decorator_list = []

    def read_image(path):
        read_calls.append(Path(path))
        return np.zeros((2, 3, 1), dtype=np.float32)

    class DatasetStub:
        @staticmethod
        def depth_from_euclidean_to_z(depth, sensor_width, focal_length):
            return depth

    namespace = {
        "np": np,
        "torch": torch,
        "os": __import__("os"),
        "read_png": read_image,
        "read_tiff": read_image,
        "KubricMultiViewDataset": DatasetStub,
    }
    exec(compile(ast.Module([method], type_ignores=[]), str(LOADER_PATH), "exec"), namespace)
    return namespace[method.name]


class KubricMetadataIndexTests(unittest.TestCase):
    def _make_scene(self, root, name="900", n_views=2, n_frames=3):
        scene = root / name
        scene.mkdir()
        np.savez(scene / "tracks_3d.npz", tracks_3d=np.zeros((n_frames, 4, 3)))
        (scene / "scene.json").write_text(
            json.dumps({"output": {"rgb": {"invalid_frame_indices": [1]}}}),
            encoding="utf-8",
        )
        for view_index in range(n_views):
            view = scene / f"view_{view_index}"
            view.mkdir()
            metadata = {
                "camera": {
                    "K": np.eye(3).tolist(),
                    "positions": [[0, 0, 0]] * n_frames,
                    "quaternions": [[1, 0, 0, 0]] * n_frames,
                    "sensor_width": 1.0,
                    "focal_length": 1.0,
                },
                "metadata": {"resolution": [3, 2]},
            }
            (view / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            np.savez(
                view / "tracks_2d.npz",
                tracks_2d=np.zeros((n_frames, 4, 2)),
                occlusion=np.zeros((n_frames, 4), dtype=bool),
            )
            for frame in range(n_frames):
                (view / f"rgba_{frame:05d}.png").touch()
                (view / f"depth_{frame:05d}.tiff").touch()
        return scene

    def test_builds_relocatable_camera_and_file_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_scene(root)
            manifest_path = INDEX.build_kubric_metadata_index(root)

            index = INDEX.KubricMetadataIndex(manifest_path.parent)
            scene, arrays = index.scene("900")
            self.assertEqual(scene["view_names"], ["view_0", "view_1"])
            self.assertEqual(scene["invalid_frame_indices"], [1])
            self.assertEqual(scene["rgba_files"][0][2], "rgba_00002.png")
            self.assertEqual(arrays["intrinsics"].shape, (2, 3, 3))
            self.assertEqual(arrays["extrinsics"].shape, (2, 3, 3, 4))
            index.validate_source(root)

    def test_indexed_loader_opens_only_requested_view_and_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_scene(root)
            index_root = INDEX.build_kubric_metadata_index(root).parent
            entry, arrays = INDEX.KubricMetadataIndex(index_root).scene("900")
            calls = []
            load = _indexed_view_loader(calls)

            views = load(root / "900", entry, arrays, [1], frame_indices=[0, 2])

            self.assertIsNone(views[0])
            self.assertEqual(views[1]["rgba"].shape[0], 2)
            self.assertEqual(views[1]["tracks_2d"].shape[0], 2)
            self.assertEqual(len(calls), 4)
            self.assertTrue(all("view_1" in str(path) for path in calls))
            self.assertFalse(any("00001" in path.name for path in calls))

    def test_fixed_seed_indexed_view_payload_matches_native_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_path = self._make_scene(root)
            index_root = INDEX.build_kubric_metadata_index(root).parent
            entry, arrays = INDEX.KubricMetadataIndex(index_root).scene("900")
            selected_view = int(np.random.RandomState(72).choice([0, 1], 1)[0])
            calls = []
            indexed = _indexed_view_loader(calls)(
                scene_path, entry, arrays, [selected_view]
            )[selected_view]

            view_path = scene_path / f"view_{selected_view}"
            with np.load(view_path / "tracks_2d.npz") as tracks_file:
                expected_tracks = torch.from_numpy(tracks_file["tracks_2d"].copy())
                expected_occlusion = torch.from_numpy(tracks_file["occlusion"].copy())
            metadata = json.loads((view_path / "metadata.json").read_text(encoding="utf-8"))
            expected_intrinsics, expected_extrinsics = INDEX._camera_matrices(metadata)

            torch.testing.assert_close(indexed["tracks_2d"], expected_tracks)
            torch.testing.assert_close(indexed["occlusion"], expected_occlusion)
            torch.testing.assert_close(
                indexed["intrinsics"], torch.from_numpy(expected_intrinsics)
            )
            torch.testing.assert_close(
                indexed["extrinsics"], torch.from_numpy(expected_extrinsics)
            )
            self.assertEqual(indexed["rgba"].shape, (3, 2, 3, 1))
            self.assertEqual(indexed["depth"].shape, (3, 2, 3, 1))

    def test_explicit_index_mode_rejects_missing_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                INDEX.KubricMetadataIndex(Path(tmp) / "missing")

    def test_source_validation_rejects_stale_frame_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = self._make_scene(root)
            index = INDEX.KubricMetadataIndex(INDEX.build_kubric_metadata_index(root).parent)
            with (scene / "view_0" / "rgba_00000.png").open("ab") as handle:
                handle.write(b"changed")

            with self.assertRaisesRegex(ValueError, "index is stale"):
                index.validate_source(root)

    def test_camera_arrays_are_preloaded_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_scene(root)
            manifest = INDEX.build_kubric_metadata_index(root)
            index = INDEX.KubricMetadataIndex(manifest.parent)
            arrays_file = manifest.parent / "scenes" / "900.npz"
            arrays_file.unlink()

            _, first = index.scene("900")
            _, second = index.scene("900")
            self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
