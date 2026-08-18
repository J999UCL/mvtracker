#!/usr/bin/env python3
"""Dopey-only profiler for the mixed physical loader.

The normal profile is deliberately bounded and uses physical Titan devices 1
and 2 only.  It plans one deterministic mixed stream, writes that stream to a
JSON sidecar, then replays the same plans on each selected device.  CPU
planning/materialisation and CUDA decode are measured separately; no model or
training step is run.

Examples (on Dopey):

    python tools/profile_mixed_physical_loader.py \
      --output-dir /media/data3/jthakwani/mvtracker-runs/mixed-physical-loader \
      --steps 4

    python tools/profile_mixed_physical_loader.py --mode parity \
      --parity-device cuda:0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


SOURCE_PATTERN = ("diegesis", "mvkubric", "diegesis", "mvkubric")
DEFAULT_DIEGESIS_ROOT = Path("/media/data3/jthakwani/datasets/diegesis-mvtracker")
DEFAULT_MVKUBRIC_ROOT = Path("/media/data3/jthakwani/datasets/mv3dpt-train-micro")
DEFAULT_MVKUBRIC_INDEX = (
    DEFAULT_MVKUBRIC_ROOT / "kubric-multiview/train/MVTracker_index"
)
DEFAULT_CONFIG = Path("configs/experiment/diegesis_mvkubric_gt_ddp.yaml")


def parse_device_ids(value: str) -> tuple[int, ...]:
    ids = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if ids != (1, 2):
        raise ValueError("loader profiling is pinned to Titan physical devices 1,2")
    return ids


@dataclass
class FixedStep:
    start_cursors: dict[str, int]
    end_cursors: dict[str, int]
    scenes: tuple[Any, ...]
    physical: Any
    retry_count: int


def _json_scene(scene) -> dict[str, Any]:
    plan = scene.plan
    return {
        "source": scene.source,
        "virtual_index": int(plan.virtual_index),
        "scene_index": int(plan.scene_index),
        "sequence": plan.sequence,
        "views": list(plan.views),
        "frames": [int(value) for value in plan.frame_indices],
        "track_count": int(plan.track_count),
    }


def _build_capacity(settings):
    from mvtracker.datasets.physical_batch_scheduler import BatchCapacity

    return BatchCapacity(
        name=str(settings.capacity_name),
        rank_count=2,
        logical_scenes_per_rank=4,
        max_group_size=int(settings.max_scenes),
        pair_track_capacity_by_views=tuple(
            sorted(
                (int(view), int(tracks))
                for view, tracks in settings.pair_track_capacity_by_views.items()
            )
        ),
        singleton_only_views=frozenset(
            int(view) for view in settings.singleton_only_views
        ),
    )


def _build_datasets(config_path: Path, diegesis_root: Path, mvkubric_root: Path, index_root: Path):
    from omegaconf import OmegaConf
    from mvtracker.datasets import KubricMultiViewDataset, TapVid3DMultiViewDataset
    from mvtracker.datasets.kubric_gpu_dataset import GpuDecodedKubricMultiViewDataset

    cfg = OmegaConf.load(config_path)
    cfg.datasets.root = str(mvkubric_root)
    cfg.datasets.train.kubric_metadata_index_root = str(index_root)
    cfg.datasets.train.sources.diegesis.root = str(diegesis_root)
    cfg.datasets.train.sources.mvkubric.root = str(mvkubric_root)
    fabric = SimpleNamespace(world_size=2)

    diegesis_cfg = cfg.datasets.train.sources.diegesis
    diegesis_kwargs = TapVid3DMultiViewDataset.from_name(
        diegesis_cfg.name,
        str(diegesis_root),
        cfg,
        fabric=fabric,
        just_return_kwargs=True,
        include_scene_ids=list(diegesis_cfg.include_scene_ids),
    )
    diegesis_kwargs["view_count_probabilities"] = tuple(
        diegesis_cfg.view_count_probabilities
    )
    mvkubric_cfg = cfg.datasets.train.sources.mvkubric
    mvkubric_kwargs = KubricMultiViewDataset.from_name(
        mvkubric_cfg.name,
        str(mvkubric_root),
        cfg,
        fabric=fabric,
        just_return_kwargs=True,
        include_scene_ids=list(mvkubric_cfg.include_scene_ids),
    )
    mvkubric_kwargs["metadata_index_root"] = str(index_root)
    return (
        cfg,
        {
            "diegesis": TapVid3DMultiViewDataset(**diegesis_kwargs),
            "mvkubric": GpuDecodedKubricMultiViewDataset(**mvkubric_kwargs),
        },
    )


def _plan_step(datasets, schedule, cursors, capacity):
    from mvtracker.datasets.mixed_physical_loader import PlannedScene, _plan_summary
    from mvtracker.datasets.physical_batch_scheduler import schedule_physical_batch

    start = dict(cursors)
    scenes = []
    retries = 0
    for source in SOURCE_PATTERN:
        cursor = cursors[source]
        while True:
            candidates = []
            for rank in range(schedule.world_size):
                request = schedule.sample_source(source, cursor, rank).request
                plan = datasets[source].plan_sample(request)
                if plan is None:
                    break
                candidates.append(PlannedScene(source, plan))
            if len(candidates) == schedule.world_size:
                scenes.extend(candidates)
                cursors[source] = cursor + 1
                break
            cursor += 1
            retries += 1
    physical = schedule_physical_batch(
        tuple(_plan_summary(scene) for scene in scenes), capacity=capacity
    )
    return FixedStep(start, dict(cursors), tuple(scenes), physical, retries)


def _plan_stream(datasets, cfg, steps: int, seed: int, capacity):
    from mvtracker.datasets.mixed_source_schedule import BalancedMixedSourceSchedule

    schedule = BalancedMixedSourceSchedule(
        {source: dataset.real_len for source, dataset in datasets.items()},
        SOURCE_PATTERN,
        world_size=2,
        master_seed=int(seed),
    )
    cursors = {source: 0 for source in datasets}
    fixed = []
    planning_seconds = []
    for _ in range(steps):
        started = time.perf_counter()
        fixed.append(_plan_step(datasets, schedule, cursors, capacity))
        planning_seconds.append(time.perf_counter() - started)
    return schedule, fixed, planning_seconds


def _save_plan(path: Path, schedule, fixed: Sequence[FixedStep], planning_seconds):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "mixed-physical-loader-plan-v1",
        "source_pattern": list(SOURCE_PATTERN),
        "master_seed": int(schedule.master_seed),
        "scene_counts": dict(schedule.scene_counts),
        "planning_seconds": planning_seconds,
        "steps": [
            {
                "start_cursors": step.start_cursors,
                "end_cursors": step.end_cursors,
                "retry_count": step.retry_count,
                "scenes": [_json_scene(scene) for scene in step.scenes],
            }
            for step in fixed
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_plan(path: Path, datasets, capacity):
    from mvtracker.datasets.mixed_source_schedule import BalancedMixedSourceSchedule
    from mvtracker.datasets.mixed_source_schedule import ScheduledSampleRequest
    from mvtracker.datasets.mixed_physical_loader import PlannedScene, _plan_summary
    from mvtracker.datasets.physical_batch_scheduler import schedule_physical_batch

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "mixed-physical-loader-plan-v1":
        raise ValueError(f"unsupported plan schema: {path}")
    schedule = BalancedMixedSourceSchedule(
        payload["scene_counts"],
        tuple(payload["source_pattern"]),
        world_size=2,
        master_seed=int(payload["master_seed"]),
    )
    fixed = []
    replay_seconds = []
    for step_payload in payload["steps"]:
        started = time.perf_counter()
        scenes = []
        for item in step_payload["scenes"]:
            request = ScheduledSampleRequest(
                virtual_index=int(item["virtual_index"]),
                scene_index=int(item["scene_index"]),
            )
            plan = datasets[item["source"]].plan_sample(request)
            if plan is None:
                raise RuntimeError(f"fixed plan became invalid: {item}")
            observed = _json_scene(PlannedScene(item["source"], plan))
            for key in ("source", "virtual_index", "scene_index", "sequence", "views", "frames", "track_count"):
                if observed[key] != item[key]:
                    raise RuntimeError(
                        f"fixed plan mismatch for {key}: {observed[key]!r} != {item[key]!r}"
                    )
            scenes.append(PlannedScene(item["source"], plan))
        physical = schedule_physical_batch(
            tuple(_plan_summary(scene) for scene in scenes), capacity=capacity
        )
        fixed.append(
            FixedStep(
                dict(step_payload["start_cursors"]),
                dict(step_payload["end_cursors"]),
                tuple(scenes),
                physical,
                int(step_payload["retry_count"]),
            )
        )
        replay_seconds.append(time.perf_counter() - started)
    return schedule, fixed, replay_seconds


def _materialize_groups(step: FixedStep, datasets, rank: int, workers: int):
    from mvtracker.datasets.mixed_physical_loader import (
        PreparedPhysicalGroup,
        _pin_sample,
        _sample_nbytes,
    )

    local_groups = step.physical.ranks[rank].groups
    by_identity = {scene.identity: scene for scene in step.scenes}
    local_scenes = [
        by_identity[(summary.source, summary.scene, summary.cursor)]
        for group in local_groups
        for summary in group.scenes
    ]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            scene.identity: executor.submit(
                datasets[scene.source].materialize_sample, scene.plan
            )
            for scene in local_scenes
        }
        prepared = {}
        for scene in local_scenes:
            sample, valid = futures[scene.identity].result()
            if not valid or sample is None:
                raise RuntimeError(f"materialization failed: {scene.identity}")
            sample.metadata["source"] = scene.source
            prepared[scene.identity] = _pin_sample(sample)
    groups = []
    for physical_group in local_groups:
        scenes = tuple(
            by_identity[(summary.source, summary.scene, summary.cursor)]
            for summary in physical_group.scenes
        )
        groups.append(
            PreparedPhysicalGroup(
                scenes=scenes,
                samples=tuple(prepared[scene.identity] for scene in scenes),
            )
        )
    return tuple(groups), time.perf_counter() - started, sum(
        _sample_nbytes(sample) for group in groups for sample in group.samples
    )


class _TimedDecoder:
    def __init__(self, device):
        from mvtracker.datasets.mixed_physical_loader import PhysicalBatchDecoder

        self.device = device
        self.decoder = PhysicalBatchDecoder(device)
        self.timings = []

    def decode_async(self, group):
        import torch

        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        started.record(self.decoder.prepare_stream)
        result = self.decoder.decode_async(group)
        finished.record(self.decoder.prepare_stream)
        self.timings.append((started, finished))
        return result


def _decode_groups(groups, device, gpu_handles):
    import torch
    from mvtracker.datasets.mixed_physical_loader import PhysicalGroupPrefetchIterator

    timed = _TimedDecoder(device)
    records = []
    for group in groups:
        exposed_started = time.perf_counter()
        iterator = PhysicalGroupPrefetchIterator((group,), timed)
        _, datapoint = next(iterator)
        exposed_wait = time.perf_counter() - exposed_started
        started_event, finished_event = timed.timings[-1]
        finished_event.synchronize()
        decode_ms = started_event.elapsed_time(finished_event)
        validation_started = time.perf_counter()
        if not datapoint.video.is_cuda or not datapoint.videodepth.is_cuda:
            raise RuntimeError("CUDA decode returned a CPU tensor")
        finite = bool(
            torch.isfinite(datapoint.video).all().item()
            and torch.isfinite(datapoint.videodepth).all().item()
        )
        if not finite:
            raise RuntimeError("CUDA decode returned non-finite values")
        validation_seconds = time.perf_counter() - validation_started
        util = []
        for handle in gpu_handles:
            rates = handle[0].get_utilization_rates(handle[1])
            memory = handle[0].get_memory_info(handle[1])
            util.append({
                "gpu_utilization_percent": float(rates.gpu),
                "memory_used_gib": memory.used / 1024**3,
                "memory_total_gib": memory.total / 1024**3,
            })
        records.append({
            "decode_ms": decode_ms,
            "exposed_wait_seconds": exposed_wait,
            "validation_seconds": validation_seconds,
            "cuda_finite": finite,
            "gpu": util,
        })
    return records


def _resource_sample(gpu_handles):
    import psutil

    process = psutil.Process()
    sample = {
        "cpu_percent": float(process.cpu_percent(None)),
        "rss_gib": process.memory_info().rss / 1024**3,
    }
    gpu = []
    for handle in gpu_handles:
        rates = handle[0].get_utilization_rates(handle[1])
        memory = handle[0].get_memory_info(handle[1])
        gpu.append({
            "gpu_utilization_percent": float(rates.gpu),
            "memory_used_gib": memory.used / 1024**3,
            "memory_total_gib": memory.total / 1024**3,
        })
    sample["gpu"] = gpu
    return sample


def _run_lane(device_index: int, fixed, datasets, workers: int, passes: int):
    import torch
    import pynvml

    torch.cuda.set_device(device_index)
    pynvml.nvmlInit()
    nvml_handles = [
        (pynvml, pynvml.nvmlDeviceGetHandleByIndex(1)),
        (pynvml, pynvml.nvmlDeviceGetHandleByIndex(2)),
    ]
    results = {"device": device_index, "passes": []}
    for pass_index in range(passes):
        pass_name = "cold" if pass_index == 0 else "warm"
        samples = 0
        trajectories = 0
        materialization_seconds = 0.0
        encoded_bytes = 0
        decode_records = []
        resources = []
        step_records = []
        pass_started = time.perf_counter()
        for step in fixed:
            groups, materialization_time, encoded = _materialize_groups(
                step, datasets, device_index, workers
            )
            materialization_seconds += materialization_time
            encoded_bytes += encoded
            samples += sum(len(group.samples) for group in groups)
            trajectories += sum(
                sample.trajectory.shape[-2]
                for group in groups
                for sample in group.samples
            )
            resources.append(_resource_sample(nvml_handles))
            step_decode = _decode_groups(
                groups, torch.device(f"cuda:{device_index}"), nvml_handles
            )
            decode_records.extend(step_decode)
            step_records.append({
                "materialization_seconds": materialization_time,
                "encoded_bytes": encoded,
                "samples": sum(len(group.samples) for group in groups),
                "trajectories": sum(
                    sample.trajectory.shape[-2]
                    for group in groups
                    for sample in group.samples
                ),
                "decode": step_decode,
            })
        wall = time.perf_counter() - pass_started
        decode_seconds = sum(item["decode_ms"] for item in decode_records) / 1000.0
        exposed_wait = sum(item["exposed_wait_seconds"] for item in decode_records)
        results["passes"].append({
            "name": pass_name,
            "steps": len(fixed),
            "samples": samples,
            "trajectories": trajectories,
            "encoded_bytes": encoded_bytes,
            "wall_seconds": wall,
            "materialization_seconds": materialization_seconds,
            "decode_seconds_cuda_events": decode_seconds,
            "exposed_wait_seconds": exposed_wait,
            "samples_per_second": samples / wall if wall else 0.0,
            "trajectories_per_second": trajectories / wall if wall else 0.0,
            "resources": resources,
            "decode": decode_records,
            "steps_detail": step_records,
        })
    pynvml.nvmlShutdown()
    return results


def _run_parity(device_name: str):
    import torch
    from mvtracker.datasets.mixed_physical_loader import merge_decoded_datapoints

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    from mvtracker.datasets.utils import Datapoint

    def datapoint(batch, tracks):
        return Datapoint(
            video=torch.arange(batch * 1 * 2 * 3 * 4 * 5, device=device, dtype=torch.float32).reshape(batch, 1, 2, 3, 4, 5),
            segmentation=None,
            videodepth=torch.ones(batch, 1, 2, 1, 4, 5, device=device),
            valid=torch.ones(batch, 2, tracks, device=device),
            seq_name=[f"scene-{i}" for i in range(batch)],
            intrs=torch.eye(3, device=device).reshape(1, 1, 3, 3).repeat(batch, 1, 1, 1),
            query_points_3d=torch.zeros(batch, tracks, 4, device=device),
            trajectory=torch.zeros(batch, 1, 2, tracks, 3, device=device),
            trajectory_3d=torch.zeros(batch, 2, tracks, 3, device=device),
            visibility=torch.ones(batch, 1, 2, tracks, device=device),
            extrs=torch.zeros(batch, 1, 2, 3, 4, device=device),
            track_padding_mask=torch.zeros(batch, tracks, dtype=torch.bool, device=device),
            sample_metadata=[{"source": "diegesis"}] * batch,
        )

    first, second = datapoint(1, 3), datapoint(1, 5)
    merged = merge_decoded_datapoints((first, second))
    expected = torch.cat((first.video, second.video), dim=0)
    torch.testing.assert_close(merged.video, expected)
    torch.testing.assert_close(merged.trajectory[0, ..., :3, :], first.trajectory[0])
    torch.testing.assert_close(merged.trajectory[1], second.trajectory[0])
    for batch_size in (1, 2):
        scaled = torch.tensor(8.0, device=device) * (batch_size / 4.0)
        torch.testing.assert_close(scaled, torch.tensor(2.0 * batch_size, device=device))
    return {"requested_device": device_name, "merge": "ok", "loss_scaling": "ok"}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("loader", "parity"), default="loader")
    parser.add_argument("--device-ids", default="1,2")
    parser.add_argument("--parity-device", default="cuda:0")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--diegesis-root", type=Path, default=DEFAULT_DIEGESIS_ROOT)
    parser.add_argument("--mvkubric-root", type=Path, default=DEFAULT_MVKUBRIC_ROOT)
    parser.add_argument("--mvkubric-index-root", type=Path, default=DEFAULT_MVKUBRIC_INDEX)
    parser.add_argument("--output-dir", type=Path, default=Path("/media/data3/jthakwani/mvtracker-runs/mixed-physical-loader"))
    parser.add_argument("--plan-file", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=72)
    args = parser.parse_args(argv)
    if args.mode == "parity":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.parity_device.rsplit(":", 1)[-1])
        result = _run_parity(args.parity_device)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.steps < 1 or args.workers < 1 or args.passes < 1:
        raise ValueError("steps, workers, and passes must be positive")
    physical_ids = parse_device_ids(args.device_ids)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(value) for value in physical_ids)

    import torch
    import pynvml

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("loader profile requires two visible CUDA devices")
    cfg, datasets = _build_datasets(
        args.config,
        args.diegesis_root,
        args.mvkubric_root,
        args.mvkubric_index_root,
    )
    capacity = _build_capacity(cfg.datasets.train.physical_batching)
    plan_file = args.plan_file or args.output_dir / "plan.json"
    if plan_file.exists():
        schedule, fixed, replay_seconds = _load_plan(plan_file, datasets, capacity)
        plan_mode = "replay"
    else:
        schedule, fixed, planning_seconds = _plan_stream(
            datasets, cfg, args.steps, args.seed, capacity
        )
        _save_plan(plan_file, schedule, fixed, planning_seconds)
        replay_seconds = []
        plan_mode = "generated"

    result = {
        "mode": "loader",
        "plan_mode": plan_mode,
        "plan_file": str(plan_file),
        "steps": len(fixed),
        "source_pattern": list(SOURCE_PATTERN),
        "physical_devices": list(physical_ids),
        "planning_seconds": replay_seconds if plan_mode == "replay" else planning_seconds,
        "steps_retry_count": [step.retry_count for step in fixed],
        "lanes": [],
    }
    for local_device in range(2):
        print(f"[loader-profile] starting physical Titan {physical_ids[local_device]}", flush=True)
        result["lanes"].append(
            _run_lane(local_device, fixed, datasets, args.workers, args.passes)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = args.output_dir / "report.json"
    report.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"report": str(report), "plan": str(plan_file)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
