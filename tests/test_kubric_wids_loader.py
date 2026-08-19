import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mvtracker.datasets.kubric_dali_dataset import (
    KubricWebDatasetCatalog,
    _WidsRecordStore,
    _packed_frames,
)


class _Reader:
    def __init__(self, records, calls):
        self.records = records
        self.calls = calls

    def __getitem__(self, index):
        self.calls.append(int(index))
        return self.records[int(index)]


def _packed(*frames):
    offsets = [0]
    payload = b""
    for frame in frames:
        payload += frame
        offsets.append(len(payload))
    stream = io.BytesIO()
    np.savez(stream, bytes=np.frombuffer(payload, dtype=np.uint8), offsets=np.asarray(offsets))
    return stream.getvalue()


class KubricWidsLoaderTests(unittest.TestCase):
    def test_catalog_reads_only_catalog_and_maps_selected_views(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = root / "train"
            split.mkdir()
            (split / "shards.json").write_text("{}", encoding="utf-8")
            (split / "manifest.json").write_text(
                json.dumps(
                    {
                        "format": "mvtracker-kubric-webdataset",
                        "wids_descriptor": "shards.json",
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

    def test_wids_reader_is_lazy_and_media_reads_are_exact(self):
        calls = []
        records = {
            7: {".meta.npz": b"metadata"},
            20: {".rgb.npz": _packed(b"r0", b"r1"), ".depth.npz": _packed(b"d0", b"d1")},
            24: {".rgb.npz": _packed(b"R0", b"R1"), ".depth.npz": _packed(b"D0", b"D1")},
        }
        store = _WidsRecordStore(
            "/tmp/descriptor.json",
            reader_factory=lambda _: _Reader(records, calls),
        )
        self.assertEqual(calls, [])
        self.assertEqual(_packed_frames(store.get(20), "rgb.npz"), (b"r0", b"r1"))
        self.assertEqual(_packed_frames(store.get(24), "depth.npz"), (b"D0", b"D1"))
        self.assertEqual(calls, [20, 24])


if __name__ == "__main__":
    unittest.main()
