import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from mvtracker.datasets.kubric_dali_dataset import (
    IndexedReadStats,
    KubricWebDatasetCatalog,
    DaliKubricMultiViewDataset,
    _IndexedRecordStore,
    _packed_frames,
)
from mvtracker.datasets.tapvid3d_multiview_dataset import SamplePlan


def _packed(*frames):
    offsets = [0]
    payload = b""
    for frame in frames:
        payload += frame
        offsets.append(len(payload))
    stream = io.BytesIO()
    np.savez(stream, bytes=np.frombuffer(payload, dtype=np.uint8), offsets=np.asarray(offsets))
    return stream.getvalue()


def _plan(scene, views, media_indices=(), metadata=None):
    frames = np.arange(24, dtype=np.int64)
    return SamplePlan(
        dataset="kubric-dali",
        virtual_index=7,
        scene_index=0,
        sequence=scene,
        seed=7,
        frame_indices=frames,
        views=views,
        preselected_track_indices=np.asarray([0]),
        selected_track_indices=np.asarray([0]),
        selected_global_track_indices=np.asarray([0]),
        track_count=1,
        query_points_3d=np.zeros((1, 4), dtype=np.float32),
        trajectory=np.zeros((len(views), 24, 1, 3), dtype=np.float32),
        trajectory_3d=np.zeros((24, 1, 3), dtype=np.float32),
        visibility=np.ones((len(views), 24, 1), dtype=np.bool_),
        intrinsics=np.zeros((len(views), 24, 3, 3), dtype=np.float32),
        extrinsics=np.zeros((len(views), 24, 3, 4), dtype=np.float32),
        theta=np.eye(3, dtype=np.float32),
        source_size=(2, 3),
        output_size=(2, 3),
        image_codec="dali",
        depth_source="gt",
        rgb_sources=(),
        depth_sources=(),
        apply_rgb_aug=False,
        rgb_augmentation=None,
        apply_depth_aug=False,
        depth_patch_operations=(),
        augmentation_seed=7,
        depth_scale=1.0,
        max_depth=100.0,
        depth_sensor_widths=tuple(1.0 for _ in views),
        depth_focal_lengths=tuple(1.0 for _ in views),
        metadata={} if metadata is None else metadata,
        media_record_indices=media_indices,
    )


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

    def test_indexed_store_reads_requested_components_by_global_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "shard.tar"
            archive.write_bytes(b"unusedMETAxxRGBDEPTH")
            locator = root / "record-locator.npz"
            np.savez(
                locator,
                format=np.asarray("mvtracker-record-locator-v1"),
                shards=np.asarray([archive.name]),
                keys=np.asarray(["scene-a", "scene-a-view-03"]),
                record_shards=np.asarray([0, 0], dtype=np.int32),
                component_names=np.asarray(["meta.npz", "rgb.npz", "depth.npz"]),
                offsets=np.asarray([[6, -1, -1], [-1, 12, 15]], dtype=np.int64),
                sizes=np.asarray([[4, 0, 0], [0, 3, 5]], dtype=np.int64),
            )

            records, stats = _IndexedRecordStore(locator).read_many((1, 0))

            self.assertEqual(records[0]["__key__"], "scene-a-view-03")
            self.assertEqual(records[0][".rgb.npz"], b"RGB")
            self.assertEqual(records[0][".depth.npz"], b"DEPTH")
            self.assertEqual(records[1][".meta.npz"], b"META")
            self.assertEqual(stats.record_count, 2)
            self.assertEqual(stats.read_bytes, 12)

    def test_indexed_plan_honors_expected_scene_and_selected_views(self):
        metadata_stream = io.BytesIO()
        np.savez(
            metadata_stream,
            scene_name=np.asarray("scene-b"),
            tracks_3d=np.zeros((24, 1, 3), dtype=np.float32),
            visibility=np.ones((10, 24, 1), dtype=np.bool_),
            intrinsics=np.zeros((10, 3, 3), dtype=np.float32),
            extrinsics=np.zeros((10, 24, 3, 4), dtype=np.float32),
            sensor_widths=np.ones(10, dtype=np.float32),
            focal_lengths=np.ones(10, dtype=np.float32),
            resolution_hw=np.asarray((2, 3)),
        )
        dataset = DaliKubricMultiViewDataset.__new__(DaliKubricMultiViewDataset)
        dataset._sequential_stream = False
        dataset.seq_names = ["scene-a", "scene-b"]
        dataset.real_len = 2
        dataset.catalog = SimpleNamespace(
            scene=lambda name: (
                {
                    "metadata_index": 11,
                    "views": {"3": {"media_index": 19}},
                },
                {},
            )
        )
        dataset._records = SimpleNamespace(
            read=lambda index: (
                {"__key__": "scene-scene-b", ".meta.npz": metadata_stream.getvalue()},
                IndexedReadStats(10, 10, 0.1, 1),
            )
        )
        dataset._plan_scene_metadata = lambda request, scene: _plan(scene.name, (3,))

        plan = dataset.plan_sample(
            SimpleNamespace(
                virtual_index=7,
                expected_scene="scene-b",
                scene_index=1,
            )
        )

        self.assertEqual(plan.sequence, "scene-b")
        self.assertEqual(plan.views, (3,))
        self.assertEqual(plan.media_record_indices, (19,))

    def test_materialize_reads_only_planned_view_records(self):
        calls = []
        records = (
            {
                "__key__": "scene-scene-a-view-01",
                ".rgb.npz": _packed(b"r1-0", b"r1-1"),
                ".depth.npz": _packed(b"d1-0", b"d1-1"),
            },
            {
                "__key__": "scene-scene-a-view-03",
                ".rgb.npz": _packed(b"r3-0", b"r3-1"),
                ".depth.npz": _packed(b"d3-0", b"d3-1"),
            },
        )

        def read_many(indices):
            calls.append(tuple(indices))
            return records, IndexedReadStats(40, 40, 0.2, 2)

        dataset = DaliKubricMultiViewDataset.__new__(DaliKubricMultiViewDataset)
        dataset.depth_provider = "gt"
        dataset._records = SimpleNamespace(read_many=read_many)
        plan = _plan(
            "scene-a",
            (1, 3),
            media_indices=(2, 4),
            metadata={
                "indexed_metadata_requested_bytes": 10,
                "indexed_metadata_read_bytes": 10,
                "indexed_metadata_read_seconds": 0.1,
            },
        )

        sample, valid = dataset.materialize_sample(plan)

        self.assertTrue(valid)
        self.assertEqual(calls, [(2, 4)])
        self.assertEqual(sample.jpeg_bytes, (b"r1-0", b"r1-1", b"r3-0", b"r3-1"))
        self.assertEqual(sample.depth_bytes, (b"d1-0", b"d1-1", b"d3-0", b"d3-1"))


if __name__ == "__main__":
    unittest.main()
