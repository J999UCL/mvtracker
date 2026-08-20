import inspect
import io
import json
import sys
import tempfile
import types
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from mvtracker.datasets import kubric_dali_dataset
from mvtracker.datasets.kubric_dali_dataset import DaliKubricMultiViewDataset
from mvtracker.datasets.kubric_dali_stream import (
    KubricDaliSceneBundle,
    KubricDaliSceneGroup,
    KubricDaliSceneStream,
)


ROOT = Path(__file__).resolve().parents[1]


def _metadata(scene_name: str) -> bytes:
    output = io.BytesIO()
    np.savez(output, scene_name=np.asarray(scene_name))
    return output.getvalue()


def _packed_frames(prefix: str) -> bytes:
    frames = tuple(f"{prefix}:{frame}".encode() for frame in range(24))
    encoded = b"".join(frames)
    offsets = np.cumsum((0, *(len(frame) for frame in frames)), dtype=np.int64)
    output = io.BytesIO()
    np.savez(
        output,
        bytes=np.frombuffer(encoded, dtype=np.uint8),
        offsets=offsets,
    )
    return output.getvalue()


def _training_bundle(scene_name: str) -> KubricDaliSceneBundle:
    frames = 24
    tracks = 8
    views = 10
    tracks_3d = np.zeros((frames, tracks, 3), dtype=np.float32)
    tracks_3d[..., 0] = np.linspace(0.1, 0.8, tracks)
    tracks_3d[..., 1] = 0.2
    tracks_3d[..., 2] = 3.0
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], views, axis=0)
    intrinsics[:, 0, 0] = 10.0
    intrinsics[:, 1, 1] = 10.0
    intrinsics[:, 0, 2] = 16.0
    intrinsics[:, 1, 2] = 16.0
    extrinsics = np.zeros((views, frames, 3, 4), dtype=np.float32)
    extrinsics[..., :3, :3] = np.eye(3, dtype=np.float32)
    metadata = io.BytesIO()
    np.savez(
        metadata,
        scene_name=np.asarray(scene_name),
        tracks_3d=tracks_3d,
        visibility=np.ones((views, frames, tracks), dtype=np.bool_),
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        sensor_widths=np.ones(views, dtype=np.float32),
        focal_lengths=np.ones(views, dtype=np.float32),
        invalid_frame_indices=np.empty(0, dtype=np.int64),
        resolution_hw=np.asarray((32, 32), dtype=np.int64),
    )
    return KubricDaliSceneBundle(
        scene_name=scene_name,
        metadata_npz=metadata.getvalue(),
        rgb_npz=tuple(_packed_frames(f"rgb:{view}") for view in range(views)),
        depth_npz=tuple(_packed_frames(f"depth:{view}") for view in range(views)),
    )


class _TensorList:
    def __init__(self, values):
        self.values = tuple(
            np.frombuffer(value, dtype=np.uint8) for value in values
        )

    def __len__(self):
        return len(self.values)

    def at(self, index):
        return self.values[index]


class _Pipeline:
    def __init__(self, outputs):
        self.outputs = outputs

    def run(self):
        return self.outputs


def _scene_outputs(scene_names):
    metadata = []
    rgb = []
    depth = []
    for scene_name in scene_names:
        metadata.append(_metadata(scene_name))
        rgb.append(b"")
        depth.append(b"")
        for view in range(10):
            metadata.append(b"")
            rgb.append(f"rgb:{scene_name}:{view}".encode())
            depth.append(f"depth:{scene_name}:{view}".encode())
    return tuple(_TensorList(values) for values in (metadata, rgb, depth))


class DaliStreamSmokeContractTests(unittest.TestCase):
    def test_native_reader_groups_exactly_one_metadata_and_ten_views_per_scene(self):
        scene_names = ("scene-a", "scene-b", "scene-c", "scene-d")
        stream = KubricDaliSceneStream.__new__(KubricDaliSceneStream)
        stream.rank = 0
        stream.heartbeat_seconds = 60.0
        stream._expected_scenes = scene_names
        stream._scene_cursor = 0
        stream._batch_index = 0
        stream._pipeline = _Pipeline(_scene_outputs(scene_names))

        group = stream.next_scene_group()

        self.assertEqual(tuple(scene.scene_name for scene in group.scenes), scene_names)
        self.assertEqual(group.batch_index, 1)
        for scene in group.scenes:
            self.assertEqual(len(scene.rgb_npz), 10)
            self.assertEqual(len(scene.depth_npz), 10)
            self.assertEqual(
                scene.rgb_npz,
                tuple(
                    f"rgb:{scene.scene_name}:{view}".encode()
                    for view in range(10)
                ),
            )
            self.assertEqual(
                scene.depth_npz,
                tuple(
                    f"depth:{scene.scene_name}:{view}".encode()
                    for view in range(10)
                ),
            )

    def test_ranks_receive_disjoint_shards_from_the_same_shuffle(self):
        reader_calls = []

        def webdataset(**kwargs):
            reader_calls.append(tuple(kwargs["paths"]))
            return (object(), object(), object())

        def pipeline_def(function):
            def build_pipeline(**_kwargs):
                outputs = function()
                return types.SimpleNamespace(outputs=outputs, build=lambda: None)

            return build_pipeline

        nvidia = types.ModuleType("nvidia")
        dali = types.ModuleType("nvidia.dali")
        dali.pipeline_def = pipeline_def
        fn = types.ModuleType("nvidia.dali.fn")
        fn.readers = types.SimpleNamespace(webdataset=webdataset)
        nvidia.dali = dali
        dali.fn = fn

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scene_counts = (4, 4, 4, 2, 2)
            manifest = {
                "shards": [
                    {
                        "tar": f"shard-{shard:02d}.tar",
                        "nsamples": scene_count * 11,
                        "scene_ids": [
                            f"scene-{shard:02d}-{position}"
                            for position in range(scene_count)
                        ],
                    }
                    for shard, scene_count in enumerate(scene_counts)
                ]
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with patch.dict(
                sys.modules,
                {
                    "nvidia": nvidia,
                    "nvidia.dali": dali,
                    "nvidia.dali.fn": fn,
                },
            ):
                rank_zero = KubricDaliSceneStream(
                    manifest_path, rank=0, world_size=2, seed=72
                )
                rank_one = KubricDaliSceneStream(
                    manifest_path, rank=1, world_size=2, seed=72
                )

        zero = set(rank_zero.assigned_shards)
        one = set(rank_one.assigned_shards)
        expected = {
            str(root / f"shard-{shard:02d}.tar")
            for shard in range(len(scene_counts))
        }
        self.assertTrue(zero.isdisjoint(one))
        self.assertEqual(zero | one, expected)
        self.assertEqual(set(reader_calls[0]), zero)
        self.assertEqual(set(reader_calls[1]), one)
        assigned_scene_counts = [
            sum(
                scene_count
                for shard, scene_count in enumerate(scene_counts)
                if str(root / f"shard-{shard:02d}.tar") in assigned
            )
            for assigned in (zero, one)
        ]
        self.assertEqual(assigned_scene_counts, [8, 8])

    def test_one_scene_group_is_consumed_in_two_ordered_passes(self):
        scenes = tuple(
            KubricDaliSceneBundle(name, b"meta", (b"rgb",) * 10, (b"depth",) * 10)
            for name in ("a", "b", "c", "d")
        )
        group = KubricDaliSceneGroup(
            scenes=scenes,
            batch_index=1,
            read_seconds=0.25,
            payload_bytes=123,
        )
        stream = types.SimpleNamespace(next_scene_group=lambda: group)
        dataset = DaliKubricMultiViewDataset.__new__(DaliKubricMultiViewDataset)
        dataset.stream = stream
        dataset._streamed_scenes = deque()

        consumed = [dataset._next_scene() for _ in range(8)]

        self.assertEqual(
            [bundle.scene_name for bundle, *_ in consumed],
            ["a", "b", "c", "d", "a", "b", "c", "d"],
        )
        self.assertEqual([position for _, _, position, _ in consumed], [0, 1, 2, 3] * 2)
        self.assertEqual([reuse for *_, reuse in consumed], [0] * 4 + [1] * 4)
        for start in range(0, 8, 2):
            pair = [consumed[index][0].scene_name for index in (start, start + 1)]
            self.assertEqual(len(set(pair)), 2)

    def test_reused_scene_uses_each_requests_independent_virtual_index(self):
        bundle = _training_bundle("scene-a")
        group = KubricDaliSceneGroup(
            scenes=(bundle,),
            batch_index=7,
            read_seconds=0.5,
            payload_bytes=456,
        )
        dataset = DaliKubricMultiViewDataset.__new__(DaliKubricMultiViewDataset)
        dataset._streamed_scenes = deque(
            [(bundle, group, 0, 0), (bundle, group, 0, 1)]
        )
        dataset.seq_names = ["scene-a"]
        dataset.seed = 72
        dataset.enable_variable_num_views_augs = True
        dataset.enable_variable_num_views_augs__n_views_probability = {
            views: 0.1 for views in range(1, 11)
        }
        dataset.num_views = -1
        dataset.ratio_dynamic = 0.0
        dataset.ratio_very_dynamic = 0.0
        dataset.max_tracks_to_preload = None
        dataset.augmentation_probability = 1.0
        dataset.enable_rgb_augs = False
        dataset.enable_depth_augs = False
        dataset.enable_cropping_augs = True
        dataset.crop_size = (16, 16)
        dataset.traj_per_sample = 4
        dataset.enable_variable_num_views_augs__trajpersample_adjustment_factor = {
            views: 1.0 for views in range(1, 11)
        }
        dataset.enable_variable_trajpersample_augs = False
        dataset.enable_scene_transform_augs = False
        dataset.enable_camera_params_noise_augs = False
        dataset.max_depth = 24.0

        sample_indices = []
        def sample_tracks(tracks, *_args, sample_index, **_kwargs):
            sample_indices.append(sample_index)
            selected = np.arange(4, dtype=np.int64)
            query_points = np.concatenate(
                (
                    np.zeros((len(selected), 1), dtype=np.float32),
                    tracks[0, selected],
                ),
                axis=1,
            )
            return selected, query_points

        request = lambda virtual_index: types.SimpleNamespace(
            virtual_index=virtual_index,
            scene_index=999,
            view_count=2,
        )
        with patch.object(kubric_dali_dataset, "_sample_tracks", sample_tracks):
            first = dataset.plan_sample(request(100))
            second = dataset.plan_sample(request(101))

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertEqual(sample_indices, [100, 101])
        self.assertEqual((first.sequence, second.sequence), ("scene-a", "scene-a"))
        self.assertEqual((first.virtual_index, second.virtual_index), (100, 101))
        self.assertEqual((first.seed, second.seed), (172, 173))
        self.assertEqual(
            (first.metadata["dali_reuse_pass"], second.metadata["dali_reuse_pass"]),
            (0, 1),
        )
        self.assertFalse(np.array_equal(first.theta, second.theta))

    def test_mixed_training_uses_the_native_dali_scene_stream(self):
        config = yaml.safe_load(
            (ROOT / "configs/experiment/diegesis_mvkubric_gt_ddp.yaml").read_text(
                encoding="utf-8"
            )
        )

        self.assertEqual(
            config["datasets"]["train"]["mvkubric_storage"], "dali_stream"
        )

    def test_every_mixed_training_smoke_explicitly_disables_validation(self):
        paths = sorted(
            (ROOT / "configs/experiment").glob(
                "diegesis_mvkubric_gt_ddp*smoke*.yaml"
            )
        )
        self.assertGreater(len(paths), 0)

        for path in paths:
            with self.subTest(config=path.name):
                config = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.assertEqual(config["datasets"]["eval"]["names"], [])
                self.assertFalse(config["modes"]["validate_at_start"])

    def test_native_dali_stream_has_no_python_tar_reading_hot_path(self):
        from mvtracker.datasets import kubric_dali_stream

        source = inspect.getsource(kubric_dali_stream)
        self.assertIn("fn.readers.webdataset", source)
        self.assertNotIn("ShardListDataset", source)
        self.assertNotIn("_WidsRecordStore", source)
        self.assertNotIn("os.pread", source)
        self.assertNotIn("shutil.copy", source)


if __name__ == "__main__":
    unittest.main()
