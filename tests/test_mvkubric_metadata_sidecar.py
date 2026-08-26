import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mvtracker.preprocessing.mvkubric_metadata_sidecar import (
    KubricMetadataSidecar,
    build_metadata_sidecar,
)


def _metadata(scene: str) -> bytes:
    stream = io.BytesIO()
    np.savez(
        stream,
        scene_name=np.asarray(scene),
        tracks_3d=np.zeros((2, 3, 3), dtype=np.float32),
        visibility=np.ones((2, 2, 3), dtype=np.bool_),
        intrinsics=np.repeat(np.eye(3, dtype=np.float32)[None], 2, axis=0),
        extrinsics=np.zeros((2, 3, 4), dtype=np.float32),
        sensor_widths=np.ones(2, dtype=np.float32),
        focal_lengths=np.ones(2, dtype=np.float32),
        invalid_frame_indices=np.asarray([1], dtype=np.int64),
        resolution_hw=np.asarray([32, 32], dtype=np.int32),
    )
    return stream.getvalue()


class MvKubricMetadataSidecarTests(unittest.TestCase):
    def _source(self, root: Path) -> dict[str, bytes]:
        payloads = {scene: _metadata(scene) for scene in ("1", "2", "3")}
        archive_path = root / "source.tar"
        with tarfile.open(archive_path, "w") as archive:
            for scene, payload in payloads.items():
                member = tarfile.TarInfo(f"scene-{scene}.meta.npz")
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
                media = tarfile.TarInfo(f"scene-{scene}-view-00.rgb.npz")
                media.size = 4
                archive.addfile(media, io.BytesIO(b"media"))
        with tarfile.open(archive_path) as archive:
            members = {member.name: member for member in archive.getmembers()}
        lines = ["v1.2 6"]
        for scene in payloads:
            member = members[f"scene-{scene}.meta.npz"]
            lines.append(f"meta.npz {member.offset_data} {member.size} {member.name}")
            member = members[f"scene-{scene}-view-00.rgb.npz"]
            lines.append(f"rgb.npz {member.offset_data} {member.size} {member.name}")
        (root / "source.idx").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "mvtracker-kubric-webdataset",
                    "scene_ids": list(payloads),
                    "shards": [{"tar": "source.tar", "nsamples": 6}],
                }
            ),
            encoding="utf-8",
        )
        return payloads

    def test_build_copies_metadata_verbatim_into_sixteen_indexed_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = self._source(root)
            manifest = build_metadata_sidecar(root, root / "sidecar")
            self.assertEqual(manifest["shard_count"], 16)
            self.assertEqual(len(manifest["shards"]), 16)
            for shard in manifest["shards"]:
                self.assertTrue((root / "sidecar" / str(shard["tar"])).is_file())
                self.assertTrue((root / "sidecar" / str(shard["tar"]).replace(".tar", ".idx")).is_file())
            sidecar = KubricMetadataSidecar(root / "sidecar")
            for scene, expected in payloads.items():
                entry = sidecar.scenes[scene]
                archive_path = root / "sidecar" / str(entry["shard"])
                with tarfile.open(archive_path) as archive:
                    observed = archive.extractfile(f"scene-{scene}.meta.npz").read()
                self.assertEqual(observed, expected)

    def test_stage_copies_only_requested_shard_and_loads_scene_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._source(root)
            build_metadata_sidecar(root, root / "sidecar")
            sidecar = KubricMetadataSidecar(root / "sidecar")
            staged = sidecar.stage(("2",), root / "staged", workers=2)
            required = sidecar.required_shards(("2",))[0]
            self.assertEqual(sorted(path.name for path in staged.iterdir()), [required, required.replace(".tar", ".idx")])
            loaded = sidecar.load("2", staged_root=staged)
            self.assertEqual(loaded.name, "2")
            self.assertEqual(loaded.frame_count, 2)
            self.assertEqual(loaded.view_count, 2)
            self.assertEqual(loaded.invalid_frame_indices, (1,))


if __name__ == "__main__":
    unittest.main()
