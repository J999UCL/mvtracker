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


class KubricIndexedLoaderTests(unittest.TestCase):
    def test_catalog_reads_only_catalog_and_maps_selected_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "train"
            split.mkdir()
            (split / "record-locator.npz").touch()
            (split / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "mvtracker-kubric-webdataset",
                        "record_locator": "record-locator.npz",
                        "scenes": {
                            "1001": {
                                "metadata_index": 7,
                                "view_names": [f"view_{i:02d}" for i in range(10)],
                                "views": {str(i): {"media_index": 20 + i} for i in range(10)},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            catalog = KubricWebDatasetCatalog(split / "manifest.json")
            self.assertEqual(catalog.scene("1001")[0]["metadata_index"], 7)
            self.assertEqual(catalog.scene("1001")[0]["views"]["4"]["media_index"], 24)

    def test_packed_media_reads_are_exact(self):
        record = {
            ".rgb.npz": _packed(b"r0", b"r1"),
            ".depth.npz": _packed(b"D0", b"D1"),
        }
        self.assertEqual(_packed_frames(record, "rgb.npz"), (b"r0", b"r1"))
        self.assertEqual(_packed_frames(record, "depth.npz"), (b"D0", b"D1"))


if __name__ == "__main__":
    unittest.main()
