import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mvtracker.preprocessing.mvkubric_webdataset import (
    COMPONENTS,
    SceneShard,
    _scene_components,
    read_component,
    split_scene_ids,
    write_shard,
)

try:
    from mvtracker.datasets.kubric_dali_dataset import _record_from_outputs
except (ImportError, ModuleNotFoundError):
    _record_from_outputs = None


class MvKubricWebDatasetTests(unittest.TestCase):
    def _make_scene(self, root: Path, scene_id: str = "900") -> Path:
        scene = root / scene_id
        scene.mkdir()
        frames, tracks = 3, 4
        np.savez(scene / "tracks_3d.npz", tracks_3d=np.ones((frames, tracks, 3), dtype=np.float32))
        (scene / "scene.json").write_text(
            json.dumps({"output": {"rgb": {"invalid_frame_indices": [1]}}})
        )
        for view in range(6):
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

    def test_scene_components_preserve_arrays_and_encoded_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_scene(root)
            components = _scene_components(root / "900", "900", read_workers=4)
            self.assertEqual(set(components), set(COMPONENTS))
            metadata = read_component(components["meta.npz"])
            np.testing.assert_array_equal(metadata["tracks_3d"], np.ones((3, 4, 3), dtype=np.float32))
            np.testing.assert_array_equal(metadata["invalid_frame_indices"], [1])
            rgb = read_component(components["rgb2.npz"])
            offsets = rgb["offsets"]
            self.assertEqual(offsets.tolist(), [0, 3, 6, 9])
            self.assertEqual(rgb["bytes"].tobytes(), bytes([2, 0, 1, 2, 1, 1, 2, 2, 1]))

    def test_write_shard_uses_standard_key_components(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_scene(root)
            output = root / "shard.tar"
            result = write_shard(root, SceneShard("mvkubric-00000", ("900",)), output, read_workers=2)
            self.assertEqual(result["components_per_scene"], len(COMPONENTS))
            with tarfile.open(output, "r") as archive:
                names = archive.getnames()
                self.assertEqual(names, [f"900.{component}" for component in COMPONENTS])
                metadata = read_component(archive.extractfile("900.meta.npz").read())
                np.testing.assert_array_equal(metadata["resolution_hw"], [2, 3])

                if _record_from_outputs is not None:
                    outputs = [archive.extractfile(f"900.{component}").read() for component in COMPONENTS]
                    record = _record_from_outputs(outputs)
                    self.assertEqual(record.name, "900")
                    self.assertEqual(record.frame_count, 3)
                    self.assertEqual(record.view_count, 6)
                    self.assertEqual(record.rgb_frames[2][1], bytes([2, 1, 1]))

    def test_split_scene_ids_is_four_scene_shards(self):
        shards = split_scene_ids(["100", "2", "3", "1", "5"], scenes_per_shard=4)
        self.assertEqual([shard.name for shard in shards], ["mvkubric-00000", "mvkubric-00001"])
        self.assertEqual(shards[0].scene_ids, ("1", "2", "3", "5"))
        self.assertEqual(shards[1].scene_ids, ("100",))


if __name__ == "__main__":
    unittest.main()
