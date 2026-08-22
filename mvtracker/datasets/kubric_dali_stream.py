"""Native DALI streaming for MV-Kubric WebDataset shards."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import random
import threading
import time

import numpy as np


SCENES_PER_BATCH = 4
VIEWS_PER_SCENE = 10
RECORDS_PER_SCENE = 1 + VIEWS_PER_SCENE
RECORDS_PER_BATCH = SCENES_PER_BATCH * RECORDS_PER_SCENE


@dataclass(frozen=True)
class KubricDaliSceneBundle:
    scene_name: str
    metadata_npz: bytes
    rgb_npz: tuple[bytes, ...]
    depth_npz: tuple[bytes, ...]


@dataclass(frozen=True)
class KubricDaliSceneGroup:
    scenes: tuple[KubricDaliSceneBundle, ...]
    batch_index: int
    read_seconds: float
    payload_bytes: int


def _tensor_bytes(tensor) -> bytes:
    return np.asarray(tensor, dtype=np.uint8).reshape(-1).tobytes()


def _scene_name(metadata_npz: bytes) -> str:
    with np.load(BytesIO(metadata_npz), allow_pickle=False) as payload:
        return str(np.asarray(payload["scene_name"]).item())


class KubricDaliSceneStream:
    """Stream complete four-scene shards directly through DALI WebDataset."""

    repeat = True
    scenes_per_batch = SCENES_PER_BATCH
    records_per_batch = RECORDS_PER_BATCH
    _selected_scene_ids = None

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        rank: int,
        world_size: int,
        seed: int,
        scenes_per_batch: int = SCENES_PER_BATCH,
        num_threads: int = 8,
        prefetch_queue_depth: int = 2,
        heartbeat_seconds: float = 10.0,
        repeat: bool = True,
        shuffle_shards: bool = True,
        include_scene_ids: tuple[str, ...] | None = None,
        allow_empty: bool = False,
        start_group_index: int = 0,
    ):
        if scenes_per_batch < 1:
            raise ValueError("scenes_per_batch must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if start_group_index < 0:
            raise ValueError("start_group_index must be non-negative")

        manifest_path = Path(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected_scene_ids = (
            None if include_scene_ids is None else frozenset(map(str, include_scene_ids))
        )
        shard_pairs = []
        for shard in manifest["shards"]:
            scene_names = tuple(map(str, shard["scene_ids"]))
            if int(shard["nsamples"]) != len(scene_names) * RECORDS_PER_SCENE:
                raise ValueError(
                    f"{shard['tar']}: record count does not match its scenes"
                )
            archive = manifest_path.parent / str(shard["tar"])
            selected_names = (
                scene_names
                if selected_scene_ids is None
                else tuple(name for name in scene_names if name in selected_scene_ids)
            )
            if selected_names:
                shard_pairs.append(
                    (archive, archive.with_suffix(".idx"), scene_names, selected_names)
                )
        if shuffle_shards:
            random.Random(int(seed)).shuffle(shard_pairs)
        shard_pairs.sort(key=lambda pair: len(pair[3]), reverse=True)
        partitions: list[
            list[tuple[Path, Path, tuple[str, ...], tuple[str, ...]]]
        ] = [
            [] for _ in range(world_size)
        ]
        scene_counts = [0] * world_size
        for pair in shard_pairs:
            destination = min(range(world_size), key=lambda value: scene_counts[value])
            partitions[destination].append(pair)
            scene_counts[destination] += len(pair[3])
        assigned = tuple(partitions[int(rank)])
        if not assigned and not allow_empty:
            raise ValueError(f"rank {rank} was assigned no WebDataset shards")
        local_scene_count = int(scene_counts[int(rank)])
        assigned_scene_count = sum(
            len(scene_names) for _, _, scene_names, _ in assigned
        )
        if assigned_scene_count % int(scenes_per_batch):
            raise ValueError("rank-local scene count must form complete DALI batches")
        groups_per_epoch = assigned_scene_count // int(scenes_per_batch)
        start_epoch, start_group_in_epoch = (
            divmod(int(start_group_index), groups_per_epoch)
            if groups_per_epoch
            else (0, 0)
        )

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.heartbeat_seconds = float(heartbeat_seconds)
        self.repeat = bool(repeat)
        self._shuffle_shards = bool(shuffle_shards)
        self._shuffle_scene_groups = bool(shuffle_shards)
        self.scenes_per_batch = int(scenes_per_batch)
        self.records_per_batch = self.scenes_per_batch * RECORDS_PER_SCENE
        self._assigned = assigned
        self.assigned_shards = tuple(str(archive) for archive, _, _, _ in assigned)
        self._selected_scene_ids = selected_scene_ids
        self.local_scene_names = tuple(
            scene_name
            for _, _, _, selected_names in assigned
            for scene_name in selected_names
        )
        self.local_scene_count = local_scene_count
        self._scene_cursor = 0
        self._batch_index = int(start_group_index)
        self._epoch = int(start_epoch)
        self._group_in_epoch = 0

        if not assigned:
            self.build_seconds = 0.0
            self._pipeline = None
            return

        import nvidia.dali.fn as fn
        from nvidia.dali import pipeline_def

        def build_pipeline(shards):
            paths = [str(archive) for archive, _, _, _ in shards]
            index_paths = [str(index) for _, index, _, _ in shards]

            @pipeline_def
            def scene_pipeline():
                return tuple(
                    fn.readers.webdataset(
                        paths=paths,
                        index_paths=index_paths,
                        ext=["meta.npz", "rgb.npz", "depth.npz"],
                        dont_use_mmap=True,
                        missing_component_behavior="empty",
                        random_shuffle=False,
                        prefetch_queue_depth=int(prefetch_queue_depth),
                        name="MVKubricReader",
                    )
                )
            pipeline = scene_pipeline(
                batch_size=self.records_per_batch,
                num_threads=int(num_threads),
                device_id=None,
                prefetch_queue_depth=int(prefetch_queue_depth),
            )
            pipeline.build()
            return pipeline

        self._build_pipeline = build_pipeline
        build_started = time.perf_counter()
        self._start_epoch(self._epoch)
        for _ in range(start_group_in_epoch):
            self._pipeline.run()
        self._scene_cursor = start_group_in_epoch * self.scenes_per_batch
        self._group_in_epoch = start_group_in_epoch
        self.build_seconds = time.perf_counter() - build_started

    def _epoch_assignment(self, epoch: int):
        assigned = list(self._assigned)
        if self._shuffle_shards:
            random.Random(f"{self.seed}:{self.rank}:{int(epoch)}").shuffle(assigned)
        return tuple(assigned)

    def _start_epoch(self, epoch: int) -> None:
        assigned = self._epoch_assignment(epoch)
        self._pipeline = self._build_pipeline(assigned)
        self._expected_scenes = tuple(
            scene_name
            for _, _, scene_names, _ in assigned
            for scene_name in scene_names
        )
        self._scene_cursor = 0
        self._group_in_epoch = 0
        self._epoch = int(epoch)

    def _heartbeat(self, stop: threading.Event, batch_index: int, started: float) -> None:
        while not stop.wait(self.heartbeat_seconds):
            print(
                "DALI_STREAM event=waiting "
                f"rank={self.rank} batch={batch_index} "
                f"elapsed_seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )

    def next_scene_group(self) -> KubricDaliSceneGroup:
        if self._pipeline is None:
            raise StopIteration
        while True:
            group = self._next_scene_group()
            if self._selected_scene_ids is None:
                return group
            selected = tuple(
                scene for scene in group.scenes if scene.scene_name in self._selected_scene_ids
            )
            if selected:
                return KubricDaliSceneGroup(
                    scenes=selected,
                    batch_index=group.batch_index,
                    read_seconds=group.read_seconds,
                    payload_bytes=group.payload_bytes,
                )

    def _next_scene_group(self) -> KubricDaliSceneGroup:
        self._batch_index += 1
        started = time.perf_counter()
        stop = threading.Event()
        watcher = threading.Thread(
            target=self._heartbeat,
            args=(stop, self._batch_index, started),
            daemon=True,
        )
        watcher.start()
        try:
            try:
                outputs = self._pipeline.run()
            except StopIteration:
                if not self.repeat:
                    raise
                if self._shuffle_shards:
                    self._start_epoch(self._epoch + 1)
                    print(
                        f"DALI_STREAM event=epoch_reshuffle "
                        f"rank={self.rank} epoch={self._epoch}",
                        flush=True,
                    )
                else:
                    self._pipeline.reset()
                    self._epoch += 1
                    self._scene_cursor = 0
                    self._group_in_epoch = 0
                    print(
                        f"DALI_STREAM event=epoch_reset rank={self.rank}",
                        flush=True,
                    )
                outputs = self._pipeline.run()
        finally:
            stop.set()
            watcher.join()
        read_seconds = time.perf_counter() - started

        if len(outputs) != 3 or len(outputs[0]) != self.records_per_batch:
            raise RuntimeError("DALI WebDataset reader did not return one complete shard")
        components = [
            tuple(
                _tensor_bytes(output.at(position))
                for position in range(self.records_per_batch)
            )
            for output in outputs
        ]
        metadata, rgb, depth = components
        scenes = []
        payload_bytes = 0
        for scene_position in range(self.scenes_per_batch):
            start = scene_position * RECORDS_PER_SCENE
            if not metadata[start] or rgb[start] or depth[start]:
                raise RuntimeError(f"record {start} is not an MV-Kubric metadata record")
            media = slice(start + 1, start + RECORDS_PER_SCENE)
            if any(metadata[media]) or not all(rgb[media]) or not all(depth[media]):
                raise RuntimeError(f"records {start + 1}:{start + RECORDS_PER_SCENE} are malformed")
            scene_name = _scene_name(metadata[start])
            expected = self._expected_scenes[
                self._scene_cursor % len(self._expected_scenes)
            ]
            if scene_name != expected:
                raise RuntimeError(
                    f"DALI scene order diverged: expected {expected!r}, got {scene_name!r}"
                )
            self._scene_cursor += 1
            scene_rgb = tuple(rgb[media])
            scene_depth = tuple(depth[media])
            scenes.append(
                KubricDaliSceneBundle(
                    scene_name=scene_name,
                    metadata_npz=metadata[start],
                    rgb_npz=scene_rgb,
                    depth_npz=scene_depth,
                )
            )
            payload_bytes += (
                len(metadata[start])
                + sum(map(len, scene_rgb))
                + sum(map(len, scene_depth))
            )

        if self._shuffle_scene_groups and len(scenes) > 1:
            random.Random(
                f"{self.seed}:{self.rank}:{self._epoch}:{self._group_in_epoch}"
            ).shuffle(scenes)
        self._group_in_epoch += 1

        return KubricDaliSceneGroup(
            scenes=tuple(scenes),
            batch_index=self._batch_index,
            read_seconds=read_seconds,
            payload_bytes=payload_bytes,
        )


__all__ = [
    "KubricDaliSceneBundle",
    "KubricDaliSceneGroup",
    "KubricDaliSceneStream",
]
