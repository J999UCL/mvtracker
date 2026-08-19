import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mvtracker.preprocessing.mvkubric_webdataset import (
    SceneShard,
    _scene_components,
    build_wids_index,
    convert_shards,
    read_component,
    split_scene_ids,
    write_shard,
)


class MvKubricWebDatasetTests(unittest.TestCase):
    def _make_scene(self, root: Path, scene_id: str = "900", n_views: int = 3) -> Path:
        scene = root / scene_id
        scene.mkdir()
        frames, tracks = 3, 4
        tracks_3d = np.zeros((frames, tracks, 3), dtype=np.float32)
        tracks_3d[..., 0] = 0.1
        tracks_3d[..., 1] = 0.2
        tracks_3d[..., 2] = -2.0
        np.savez(scene / "tracks_3d.npz", tracks_3d=tracks_3d)
        (scene / "scene.json").write_text(
            json.dumps({"output": {"rgb": {"invalid_frame_indices": [1]}}})
        )
        for view in range(n_views):
            view_root = scene / f"view_{view}"
            view_root.mkdir()
            metadata = {
                "camera": {
                    "K": np.eye(3).tolist(),
                    "positions": [[0, 0, 0]] * frames,
                    "quaternions": [[1, 0, 0, 0]] * frames,
                    "sensor_width": 1.0,
                    "focal_length": 1.0,
                },
                "metadata": {"resolution": [3, 2]},
            }
            (view_root / "metadata.json").write_text(json.dumps(metadata))
            np.savez(
                view_root / "tracks_2d.npz",
                tracks_2d=np.zeros((frames, tracks, 2), dtype=np.float32),
                occlusion=np.zeros((frames, tracks), dtype=np.bool_),
            )
            for frame in range(frames):
                (view_root / f"rgba_{frame:05d}.png").write_bytes(bytes([view, frame, 1]))
                (view_root / f"depth_{frame:05d}.tiff").write_bytes(bytes([view, frame, 2]))
        return scene

    def test_scene_components_split_metadata_and_each_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_scene(root)
            components = _scene_components(root / "900", "900", read_workers=4)
            self.assertEqual(
                set(components),
                {
                    "scene-900.meta.npz",
                    "scene-900-view-00.rgb.npz",
                    "scene-900-view-00.depth.npz",
                    "scene-900-view-01.rgb.npz",
                    "scene-900-view-01.depth.npz",
                    "scene-900-view-02.rgb.npz",
                    "scene-900-view-02.depth.npz",
                },
            )
            metadata = read_component(components["scene-900.meta.npz"])
            np.testing.assert_array_equal(metadata["tracks_3d"].shape, [3, 4, 3])
            np.testing.assert_array_equal(metadata["visibility"].shape, [3, 3, 4])
            np.testing.assert_array_equal(metadata["invalid_frame_indices"], [1])
            rgb = read_component(components["scene-900-view-02.rgb.npz"])
            self.assertEqual(rgb["offsets"].tolist(), [0, 3, 6, 9])
            self.assertEqual(rgb["bytes"].tobytes(), bytes([2, 0, 1, 2, 1, 1, 2, 2, 1]))

    def test_write_shard_has_one_metadata_and_two_media_components_per_view(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_scene(root)
            output = root / "shard.tar"
            result = write_shard(root, SceneShard("mvkubric-00000", ("900",)), output, read_workers=2)
            self.assertEqual(result["nsamples"], 4)
            with tarfile.open(output, "r") as archive:
                self.assertEqual(
                    archive.getnames(),
                    [
                        "scene-900.meta.npz",
                        "scene-900-view-00.rgb.npz",
                        "scene-900-view-00.depth.npz",
                        "scene-900-view-01.rgb.npz",
                        "scene-900-view-01.depth.npz",
                        "scene-900-view-02.rgb.npz",
                        "scene-900-view-02.depth.npz",
                    ],
                )
                metadata = read_component(archive.extractfile("scene-900.meta.npz").read())
                np.testing.assert_array_equal(metadata["resolution_hw"], [2, 3])

    def test_convert_shards_writes_catalog_with_global_wids_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_scene(root, "1")
            self._make_scene(root, "2")
            output = root / "out"
            with patch(
                "mvtracker.preprocessing.mvkubric_webdataset.build_wids_index",
                side_effect=lambda archives, index, command: index.touch() or index,
            ):
                manifest = convert_shards(root, output, ["1", "2"], read_workers=2)
            catalog = json.loads((output / "catalog.json").read_text())
            self.assertEqual(manifest["wids_descriptor"], "shards.json")
            self.assertEqual(catalog["sample_count"], 8)
            self.assertEqual(catalog["scenes"]["1"]["metadata_index"], 0)
            self.assertEqual(catalog["scenes"]["1"]["views"]["2"]["media_index"], 3)
            self.assertEqual(catalog["scenes"]["2"]["metadata_index"], 4)

    def test_build_wids_index_invokes_standard_descriptor_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "mvkubric-00000.tar"
            archive.touch()
            index = root / "shards.json"
            with patch("mvtracker.preprocessing.mvkubric_webdataset.subprocess.run") as run:
                self.assertEqual(build_wids_index([archive], index, command="widsindex"), index.resolve())
            run.assert_called_once_with(
                ["widsindex", "create", "--output", str(index.resolve()), archive.name],
                cwd=archive.parent.resolve(),
                check=True,
            )

    def test_split_scene_ids_is_four_scene_shards(self):
        shards = split_scene_ids(["100", "2", "3", "1", "5"], scenes_per_shard=4)
        self.assertEqual([shard.name for shard in shards], ["mvkubric-00000", "mvkubric-00001"])
        self.assertEqual(shards[0].scene_ids, ("1", "2", "3", "5"))
        self.assertEqual(shards[1].scene_ids, ("100",))


if __name__ == "__main__":
    unittest.main()
