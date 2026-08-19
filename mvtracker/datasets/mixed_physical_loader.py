"""CPU lookahead and just-in-time CUDA decoding for mixed training."""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, fields
from typing import Mapping, Sequence

import numpy as np
import torch

from mvtracker.datasets.physical_batch_scheduler import (
    BatchCapacity,
    H100_BATCH_CAPACITY,
    PhysicalBatchGroup,
    SceneSummary,
    schedule_rank_local_batch,
    schedule_physical_batch,
)
from mvtracker.datasets.tapvid3d_multiview_dataset import (
    DaliEncodedImageDecoder,
    EncodedTapVid3DBatch,
    EncodedTapVid3DSample,
    SamplePlan,
    decode_tapvid3d_batch,
)
from mvtracker.datasets.utils import Datapoint


@dataclass(frozen=True)
class PlannedScene:
    source: str
    plan: SamplePlan

    @property
    def identity(self) -> tuple[str, str, int]:
        return self.source, self.plan.sequence, self.plan.virtual_index


@dataclass(frozen=True)
class PreparedPhysicalGroup:
    scenes: tuple[PlannedScene, ...]
    samples: tuple[EncodedTapVid3DSample, ...]

    def __post_init__(self) -> None:
        if len(self.scenes) != len(self.samples):
            raise ValueError("prepared scenes and samples must align")
        if not 1 <= len(self.samples) <= 2:
            raise ValueError("physical groups contain one or two scenes")

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(scene.source for scene in self.scenes)


@dataclass(frozen=True)
class PreparedMixedStep:
    start_cursors: dict[str, int]
    end_cursors: dict[str, int]
    groups: tuple[PreparedPhysicalGroup, ...]
    fingerprint: str
    retry_count: int
    encoded_bytes: int
    planning_seconds: float
    materialization_seconds: float
    pair_count: int
    padding_tracks: int
    materialization_error: str | None = None

    @property
    def logical_scene_count(self) -> int:
        return sum(len(group.samples) for group in self.groups)


def _plan_summary(scene: PlannedScene) -> SceneSummary:
    plan = scene.plan
    return SceneSummary(
        source=scene.source,
        scene=plan.sequence,
        cursor=plan.virtual_index,
        view_count=len(plan.views),
        frame_count=len(plan.frame_indices),
        resolution=tuple(plan.output_size),
        track_count=plan.track_count,
    )


def _sample_nbytes(sample: EncodedTapVid3DSample) -> int:
    total = 0
    for field in fields(sample):
        value = getattr(sample, field.name)
        values = value if isinstance(value, tuple) else (value,)
        for item in values:
            if isinstance(item, torch.Tensor):
                total += item.numel() * item.element_size()
            elif isinstance(item, (bytes, bytearray, memoryview)):
                total += len(item)
    return total


def _pin_sample(sample: EncodedTapVid3DSample) -> EncodedTapVid3DSample:
    for field in fields(sample):
        value = getattr(sample, field.name)
        if isinstance(value, torch.Tensor) and value.device.type == "cpu":
            setattr(sample, field.name, value.pin_memory())
        elif isinstance(value, tuple) and any(
            isinstance(item, torch.Tensor) for item in value
        ):
            setattr(
                sample,
                field.name,
                tuple(
                    item.pin_memory()
                    if isinstance(item, torch.Tensor) and item.device.type == "cpu"
                    else item
                    for item in value
                ),
            )
    return sample


def _step_fingerprint(
    scenes: Sequence[PlannedScene],
    physical_groups: Sequence[PhysicalBatchGroup],
    start_cursors: Mapping[str, int],
    end_cursors: Mapping[str, int],
) -> str:
    payload = {
        "start_cursors": dict(sorted(start_cursors.items())),
        "end_cursors": dict(sorted(end_cursors.items())),
        "scenes": [
            {
                "source": scene.source,
                "sequence": scene.plan.sequence,
                "virtual_index": scene.plan.virtual_index,
                "seed": scene.plan.seed,
                "frames": scene.plan.frame_indices.tolist(),
                "views": list(scene.plan.views),
                "tracks": hashlib.sha256(
                    np.asarray(
                        scene.plan.selected_global_track_indices, dtype=np.int64
                    ).tobytes()
                ).hexdigest(),
            }
            for scene in scenes
        ],
        "groups": [
            [
                (scene.source, scene.scene, scene.cursor)
                for scene in group.scenes
            ]
            for group in physical_groups
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class _ByteBoundedQueue:
    def __init__(self, max_steps: int, max_bytes: int):
        if max_steps < 1 or max_bytes < 1:
            raise ValueError("lookahead limits must be positive")
        self._queue = queue.Queue(maxsize=max_steps)
        self._max_bytes = int(max_bytes)
        self._bytes = 0
        self._condition = threading.Condition()

    def put(self, item: PreparedMixedStep | BaseException | object) -> None:
        size = item.encoded_bytes if isinstance(item, PreparedMixedStep) else 0
        if size > self._max_bytes:
            raise MemoryError(
                f"one prepared step uses {size} bytes, exceeding the "
                f"{self._max_bytes}-byte CPU cache"
            )
        with self._condition:
            while self._bytes + size > self._max_bytes:
                self._condition.wait()
            self._bytes += size
        self._queue.put(item)

    def get(self):
        item = self._queue.get()
        if isinstance(item, PreparedMixedStep):
            with self._condition:
                self._bytes -= item.encoded_bytes
                self._condition.notify_all()
        return item


class MixedStepLookahead:
    """Plan globally and materialize only one rank's encoded scenes ahead."""

    def __init__(
        self,
        *,
        datasets: Mapping[str, object],
        schedule,
        source_cursors: Mapping[str, int],
        rank: int,
        remaining_steps: int,
        worker_count: int,
        lookahead_steps: int = 4,
        max_cache_bytes: int = 12 * 1024**3,
        capacity: BatchCapacity = H100_BATCH_CAPACITY,
        rank_local: bool = False,
    ):
        if remaining_steps < 0:
            raise ValueError("remaining_steps must be non-negative")
        if worker_count < 1:
            raise ValueError("worker_count must be positive")
        if not 0 <= rank < capacity.rank_count:
            raise ValueError("rank is outside the physical capacity")
        self.datasets = dict(datasets)
        self.schedule = schedule
        self.rank = int(rank)
        self.remaining_steps = int(remaining_steps)
        self.worker_count = int(worker_count)
        self.capacity = capacity
        self.rank_local = bool(rank_local)
        self._next_cursors = {name: int(value) for name, value in source_cursors.items()}
        self._queue = _ByteBoundedQueue(lookahead_steps, max_cache_bytes)
        self._finished = object()
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()

    def _plan_step(self):
        start_cursors = dict(self._next_cursors)
        scenes = []
        retries = 0
        if self.rank_local:
            for source in self.schedule.source_pattern:
                cursor = self._next_cursors[source]
                while True:
                    request = self.schedule.sample_source(source, cursor, self.rank).request
                    plan = self.datasets[source].plan_sample(request)
                    cursor += 1
                    self._next_cursors[source] = cursor
                    if plan is None:
                        retries += 1
                        continue
                    scenes.append(PlannedScene(source, plan))
                    break
            local_groups = schedule_rank_local_batch(
                tuple(_plan_summary(scene) for scene in scenes),
                capacity=self.capacity,
            )
            planned_by_identity = {scene.identity: scene for scene in scenes}
            local_physical_groups = tuple(local_groups)
            physical = None
            physical_groups = local_physical_groups
            local_scenes = tuple(
                planned_by_identity[(summary.source, summary.scene, summary.cursor)]
                for group in local_physical_groups
                for summary in group.scenes
            )
        else:
            local_physical_groups = None
        for source in self.schedule.source_pattern:
            if self.rank_local:
                break
            cursor = self._next_cursors[source]
            while True:
                candidates = []
                for rank in range(self.schedule.world_size):
                    request = self.schedule.sample_source(source, cursor, rank).request
                    plan = self.datasets[source].plan_sample(request)
                    if plan is None:
                        break
                    candidates.append(PlannedScene(source, plan))
                if len(candidates) == self.schedule.world_size:
                    scenes.extend(candidates)
                    self._next_cursors[source] = cursor + 1
                    break
                cursor += 1
                retries += 1

        if not self.rank_local:
            physical = schedule_physical_batch(
                tuple(_plan_summary(scene) for scene in scenes),
                capacity=self.capacity,
            )
            planned_by_identity = {scene.identity: scene for scene in scenes}
            local_physical_groups = physical.ranks[self.rank].groups
            local_scenes = [
                planned_by_identity[(summary.source, summary.scene, summary.cursor)]
                for group in local_physical_groups
                for summary in group.scenes
            ]
        return (
            start_cursors,
            dict(self._next_cursors),
            tuple(scenes),
            physical,
            tuple(local_physical_groups),
            tuple(local_scenes),
            retries,
        )

    def _prepare_step(self, executor: ThreadPoolExecutor) -> PreparedMixedStep:
        planning_started = time.perf_counter()
        (
            start_cursors,
            end_cursors,
            scenes,
            physical,
            physical_groups,
            local_scenes,
            retries,
        ) = self._plan_step()
        planning_seconds = time.perf_counter() - planning_started
        materialization_started = time.perf_counter()
        futures = {
            scene.identity: executor.submit(
                self.datasets[scene.source].materialize_sample, scene.plan
            )
            for scene in local_scenes
        }
        prepared = {}
        materialization_error = None
        for scene in local_scenes:
            try:
                sample, gotit = futures[scene.identity].result()
            except BaseException as error:
                materialization_error = f"{scene.identity}: {error}"
                break
            if not gotit or sample is None:
                materialization_error = f"{scene.identity}: invalid materialized sample"
                break
            sample.metadata["source"] = scene.source
            prepared[scene.identity] = _pin_sample(sample)

        groups = []
        if materialization_error is None:
            for physical_group in physical_groups:
                group_scenes = tuple(
                    next(
                        scene
                        for scene in local_scenes
                        if scene.identity
                        == (summary.source, summary.scene, summary.cursor)
                    )
                    for summary in physical_group.scenes
                )
                groups.append(
                    PreparedPhysicalGroup(
                        scenes=group_scenes,
                        samples=tuple(
                            prepared[scene.identity] for scene in group_scenes
                        ),
                    )
                )
        encoded_bytes = sum(
            _sample_nbytes(sample) for group in groups for sample in group.samples
        )
        materialization_seconds = time.perf_counter() - materialization_started
        if physical is None:
            pair_count = sum(len(group.scenes) - 1 for group in physical_groups)
            padding_tracks = sum(group.padded_track_count for group in physical_groups)
        else:
            pair_count = physical.pair_count
            padding_tracks = physical.total_padding_tracks
        return PreparedMixedStep(
            start_cursors=start_cursors,
            end_cursors=end_cursors,
            groups=tuple(groups),
            fingerprint=(
                _step_fingerprint(
                    scenes,
                    tuple(
                        group
                        for rank_wave in physical.ranks
                        for group in rank_wave.groups
                    ),
                    start_cursors,
                    end_cursors,
                )
                if physical is not None
                else _step_fingerprint(
                    scenes,
                    tuple(physical_groups),
                    start_cursors,
                    end_cursors,
                )
            ),
            retry_count=retries,
            encoded_bytes=encoded_bytes,
            planning_seconds=planning_seconds,
            materialization_seconds=materialization_seconds,
            pair_count=pair_count,
            padding_tracks=padding_tracks,
            materialization_error=materialization_error,
        )

    def _produce(self) -> None:
        try:
            with ThreadPoolExecutor(max_workers=self.worker_count) as executor:
                for _ in range(self.remaining_steps):
                    self._queue.put(self._prepare_step(executor))
        except BaseException as error:
            self._queue.put(error)
        else:
            self._queue.put(self._finished)

    def __iter__(self):
        return self

    def __next__(self) -> PreparedMixedStep:
        item = self._queue.get()
        if item is self._finished:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        return item


def _pad_tensor(
    value: torch.Tensor,
    axis: int,
    target: int,
    *,
    fill_value: bool | float = 0,
) -> torch.Tensor:
    if value.shape[axis] == target:
        return value
    shape = list(value.shape)
    shape[axis] = target - value.shape[axis]
    padding = value.new_full(shape, fill_value)
    return torch.cat((value, padding), dim=axis)


def merge_decoded_datapoints(datapoints: Sequence[Datapoint]) -> Datapoint:
    """Merge same-view decoded batches while padding only trajectory axes."""
    if len(datapoints) == 1:
        return datapoints[0]
    track_axes = {
        "trajectory": -2,
        "trajectory_3d": -2,
        "visibility": -1,
        "valid": -1,
        "query_points_3d": -2,
        "track_padding_mask": -1,
    }
    target_tracks = max(point.trajectory.shape[-2] for point in datapoints)
    values = {}
    for field in fields(Datapoint):
        items = [getattr(point, field.name) for point in datapoints]
        if all(item is None for item in items):
            values[field.name] = None
        elif all(isinstance(item, torch.Tensor) for item in items):
            axis = track_axes.get(field.name)
            if axis is not None:
                items = [
                    _pad_tensor(
                        item,
                        axis,
                        target_tracks,
                        fill_value=field.name == "track_padding_mask",
                    )
                    for item in items
                ]
            values[field.name] = torch.cat(items, dim=0)
        elif all(isinstance(item, list) for item in items):
            values[field.name] = [value for item in items for value in item]
        else:
            first = items[0]
            if any(item != first for item in items[1:]):
                raise ValueError(f"cannot merge Datapoint field {field.name}")
            values[field.name] = first
    return Datapoint(**values)


def _record_stream(datapoint: Datapoint, stream: torch.cuda.Stream) -> None:
    for value in vars(datapoint).values():
        if isinstance(value, torch.Tensor) and value.is_cuda:
            value.record_stream(stream)


class PhysicalBatchDecoder:
    """Decode one scheduled physical group on persistent CUDA streams."""

    def __init__(
        self,
        device: torch.device,
        *,
        decode_image_chunk_size: int = 64,
        dali_num_threads: int = 4,
        dali_prefetch_queue_depth: int = 2,
        dali_max_encoded_images: int = 288,
    ):
        self.device = device
        self.decode_image_chunk_size = int(decode_image_chunk_size)
        if self.decode_image_chunk_size < 1:
            raise ValueError("decode_image_chunk_size must be positive")
        self.rgb_stream = torch.cuda.Stream(device=device)
        self.depth_stream = torch.cuda.Stream(device=device)
        self.prepare_stream = torch.cuda.Stream(device=device)
        self.rgb_decoder = None
        self.depth_decoder = None
        self.dali_decoder = None
        self.dali_num_threads = int(dali_num_threads)
        self.dali_prefetch_queue_depth = int(dali_prefetch_queue_depth)
        self.dali_max_encoded_images = int(dali_max_encoded_images)
        if (
            self.dali_num_threads < 1
            or self.dali_prefetch_queue_depth < 1
            or self.dali_max_encoded_images < 1
        ):
            raise ValueError("DALI decoder settings must be positive")

    def _ensure_nvimagecodec(self) -> None:
        if self.rgb_decoder is not None:
            return
        from nvidia import nvimgcodec

        device_id = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        self.rgb_decoder = nvimgcodec.Decoder(device_id=device_id)
        self.depth_decoder = nvimgcodec.Decoder(device_id=device_id)

    def _ensure_dali(self) -> None:
        if self.dali_decoder is None:
            self.dali_decoder = DaliEncodedImageDecoder(
                self.device,
                num_threads=self.dali_num_threads,
                prefetch_queue_depth=self.dali_prefetch_queue_depth,
                max_encoded_images=self.dali_max_encoded_images,
            )

    def decode_async(self, group: PreparedPhysicalGroup):
        codec_groups = {}
        for position, sample in enumerate(group.samples):
            codec_groups.setdefault(sample.image_codec, []).append((position, sample))
        if "nvimagecodec" in codec_groups:
            self._ensure_nvimagecodec()
        if "dali" in codec_groups:
            self._ensure_dali()
        decoded = [None] * len(group.samples)
        with torch.cuda.stream(self.prepare_stream):
            for codec_items in codec_groups.values():
                datapoint = decode_tapvid3d_batch(
                    EncodedTapVid3DBatch([sample for _, sample in codec_items]),
                    self.device,
                    nvimagecodec_rgb_decoder=self.rgb_decoder,
                    nvimagecodec_depth_decoder=self.depth_decoder,
                    dali_decoder=self.dali_decoder,
                    rgb_stream=self.rgb_stream,
                    depth_stream=self.depth_stream,
                    prepare_stream=self.prepare_stream,
                    decode_image_chunk_size=self.decode_image_chunk_size,
                )
                if len(codec_items) == len(group.samples):
                    for offset, (position, _) in enumerate(codec_items):
                        decoded[position] = _slice_datapoint(datapoint, offset, offset + 1)
                else:
                    for offset, (position, _) in enumerate(codec_items):
                        decoded[position] = _slice_datapoint(datapoint, offset, offset + 1)
            merged = merge_decoded_datapoints(decoded)
            ready_event = torch.cuda.Event()
            ready_event.record(self.prepare_stream)
        return merged, ready_event


def _slice_datapoint(datapoint: Datapoint, start: int, end: int) -> Datapoint:
    values = {}
    batch_size = len(datapoint.seq_name)
    for field in fields(Datapoint):
        value = getattr(datapoint, field.name)
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == batch_size:
            value = value[start:end]
        elif isinstance(value, list) and len(value) == batch_size:
            value = value[start:end]
        values[field.name] = value
    return Datapoint(**values)


class PhysicalGroupPrefetchIterator:
    """Keep exactly one decoded physical group ready ahead of training."""

    def __init__(self, groups: Sequence[PreparedPhysicalGroup], decoder):
        self.groups = tuple(groups)
        self.decoder = decoder
        self.ready = queue.Queue(maxsize=1)
        self.finished = object()
        self.thread = threading.Thread(target=self._produce, daemon=True)
        self.thread.start()

    def _produce(self):
        try:
            torch.cuda.set_device(self.decoder.device)
            for group in self.groups:
                datapoint, event = self.decoder.decode_async(group)
                self.ready.put((group, datapoint, event))
        except BaseException as error:
            self.ready.put(error)
        else:
            self.ready.put(self.finished)

    def __iter__(self):
        return self

    def __next__(self):
        item = self.ready.get()
        if item is self.finished:
            raise StopIteration
        if isinstance(item, BaseException):
            raise item
        group, datapoint, event = item
        current = torch.cuda.current_stream(self.decoder.device)
        current.wait_event(event)
        _record_stream(datapoint, current)
        return group, datapoint


__all__ = [
    "MixedStepLookahead",
    "PhysicalBatchDecoder",
    "PhysicalGroupPrefetchIterator",
    "PreparedMixedStep",
    "PreparedPhysicalGroup",
    "merge_decoded_datapoints",
]
