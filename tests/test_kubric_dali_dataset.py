import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mvtracker.datasets.kubric_dali_dataset import (
    KubricWebDatasetCatalog,
    _packed_frames,
)


def _packed(*frames):
    offsets = [0]
    payload = b""
    for frame in frames:
        payload += frame
        offsets.append(len(payload))
    stream = io.BytesIO()
    np.savez(stream, bytes=np.frombuffer(payload, dtype=np.uint8), offsets=np.asarray(offsets))
    return stream.getvalue()


class KubricDaliDatasetTests(unittest.TestCase):
    def test_catalog_reads_scene_and_view_names_without_opening_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            split = Path(temporary) / "train"
            split.mkdir()
            (split / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "mvtracker-kubric-webdataset",
                        "scenes": {
                            "1001": {
                                "view_names": [f"view_{i:02d}" for i in range(10)],
                                "views": {str(i): {} for i in range(10)},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            scene, _ = KubricWebDatasetCatalog(split / "manifest.json").scene("1001")

            self.assertEqual(scene["view_names"], [f"view_{i:02d}" for i in range(10)])
            self.assertEqual(tuple(scene["views"]), tuple(map(str, range(10))))

    def test_packed_media_reads_are_exact(self):
        self.assertEqual(_packed_frames(_packed(b"r0", b"r1")), (b"r0", b"r1"))
        self.assertEqual(_packed_frames(_packed(b"D0", b"D1")), (b"D0", b"D1"))


if __name__ == "__main__":
    unittest.main()
