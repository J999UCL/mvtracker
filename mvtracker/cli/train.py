# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch

torch.set_float32_matmul_precision('high')

from lightning.fabric.wrappers import _unwrap_objects
from mvtracker.datasets.generic_scene_dataset import GenericSceneDataset

from torch.utils.tensorboard import SummaryWriter
import contextlib
import dataclasses
import gpustat
import json
import statistics
import threading
import traceback
import warnings
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch.optim as optim
import wandb
from lightning.fabric import Fabric
from lightning.fabric.utilities import AttributeDict
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import signal, sys

from mvtracker.datasets import (
    KubricMultiViewDataset,
    PointOdysseyMultiViewDataset,
    TapVid3DMultiViewDataset,
)
from mvtracker.datasets.tapvid3d_multiview_dataset import CudaPrefetchLoader
from mvtracker.datasets.kubric_dali_dataset import DaliKubricValidationDataset
from mvtracker.datasets.mixed_source_schedule import BalancedMixedSourceSchedule
from mvtracker.datasets.mixed_physical_loader import (
    MixedStepLookahead,
    PhysicalBatchDecoder,
    PhysicalGroupPrefetchIterator,
)
from mvtracker.datasets.physical_batch_scheduler import BatchCapacity
from mvtracker.datasets import TapVidDataset
from mvtracker.datasets import kubric_multiview_dataset
from mvtracker.datasets.dexycb_multiview_dataset import DexYCBMultiViewDataset
from mvtracker.datasets.panoptic_studio_multiview_dataset import PanopticStudioMultiViewDataset
from mvtracker.datasets.utils import (
    HomogeneousViewBatchSampler,
    collate_fn,
    dataclass_to_cuda_,
)
from mvtracker.models.core.losses import balanced_ce_loss, sequence_loss_3d
from mvtracker.models.core.model_utils import world_space_to_pixel_xy_and_camera_z, pixel_xy_and_camera_z_to_world_space
from mvtracker.models.evaluation_predictor_3dpt import EvaluationPredictor as EvaluationPredictor3D
from mvtracker.utils.visualizer_mp4 import MultiViewVisualizer, Visualizer
from mvtracker.cli.utils import extras
from mvtracker.cli.utils.helpers import maybe_close_wandb

import logging
import os

import torch
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from torchdata.stateful_dataloader import StatefulDataLoader


LATEST_CHECKPOINT_MANIFEST = "latest_checkpoint.json"
WANDB_RUN_ID_FILE = "wandb_run_id.txt"


class _ContainerHardwareMonitor:
    """Read CPU and memory usage for the complete Linux container cgroup."""

    def __init__(self, cgroup_root=Path("/sys/fs/cgroup")):
        self.cgroup_root = Path(cgroup_root)
        self._last_cpu_seconds = self._cpu_usage_seconds()
        self._last_sample_time = time.monotonic()

    def _cpu_usage_seconds(self):
        usage_ns = int(
            (self.cgroup_root / "cpuacct/cpuacct.usage").read_text()
        )
        return usage_ns / 1_000_000_000.0

    def _cpu_limit(self):
        quota = int((self.cgroup_root / "cpu/cpu.cfs_quota_us").read_text())
        period = int((self.cgroup_root / "cpu/cpu.cfs_period_us").read_text())
        if quota < 0:
            return float(len(os.sched_getaffinity(0)))
        return quota / period

    def sample(self):
        sample_time = time.monotonic()
        cpu_seconds = self._cpu_usage_seconds()
        elapsed = sample_time - self._last_sample_time
        cpu_cores = (cpu_seconds - self._last_cpu_seconds) / elapsed
        self._last_cpu_seconds = cpu_seconds
        self._last_sample_time = sample_time

        memory_used = int(
            (self.cgroup_root / "memory/memory.usage_in_bytes").read_text()
        )
        memory_limit = int(
            (self.cgroup_root / "memory/memory.limit_in_bytes").read_text()
        )
        available_cpus = self._cpu_limit()
        return {
            "hardware/container/cpu_cores_used": cpu_cores,
            "hardware/container/cpu_utilization_percent": (
                100.0 * cpu_cores / available_cpus
            ),
            "hardware/container/memory_used_gib": memory_used / (1024 ** 3),
            "hardware/container/memory_limit_gib": memory_limit / (1024 ** 3),
            "hardware/container/memory_utilization_percent": (
                100.0 * memory_used / memory_limit
            ),
        }


class _RankGpuMonitor:
    """Sample the GPU assigned to one DDP rank through NVML."""

    def __init__(self, device_index):
        import pynvml

        pynvml.nvmlInit()
        self._pynvml = pynvml
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(int(device_index))

    def sample(self):
        utilization = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
        memory = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        return {
            "utilization_percent": float(utilization.gpu),
            "memory_used_gib": memory.used / (1024 ** 3),
            "memory_total_gib": memory.total / (1024 ** 3),
            "memory_utilization_percent": 100.0 * memory.used / memory.total,
        }

    def close(self):
        self._pynvml.nvmlShutdown()


def _scale_microbatch_loss(loss, gradient_accumulation_steps):
    """Scale one microbatch loss so accumulated gradients form a batch mean."""
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    return loss / gradient_accumulation_steps


def _scale_physical_batch_loss(loss, physical_batch_size, accumulation_steps):
    """Weight a physical scene batch as its share of one rank's logical batch."""
    if physical_batch_size < 1 or accumulation_steps < 1:
        raise ValueError("physical and logical batch sizes must be positive")
    if physical_batch_size > accumulation_steps:
        raise ValueError("physical batch cannot exceed the logical rank batch")
    return loss * (physical_batch_size / accumulation_steps)


def _global_gradient_l2_norm(parameters):
    """Return the global FP32 L2 norm without changing gradients."""
    per_parameter_norms = [
        parameter.grad.detach().float().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not per_parameter_norms:
        return torch.tensor(0.0)
    return torch.stack(per_parameter_norms).norm(2)


def _gradient_value_clip_stats(parameters, clip_value):
    """Summarize the elementwise clipping performed by Fabric ``clip_val``."""
    max_abs_terms = []
    clipped_count_terms = []
    total_elements = 0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach()
        max_abs_terms.append(gradient.abs().max().float())
        clipped_count_terms.append((gradient.abs() > clip_value).sum())
        total_elements += gradient.numel()
    if not max_abs_terms:
        return 0.0, 0.0
    max_abs = float(torch.stack(max_abs_terms).max().item())
    clipped_elements = int(torch.stack(clipped_count_terms).sum().item())
    return max_abs, clipped_elements / float(total_elements)


class _MicrobatchGradientDiagnostics:
    """Measure individual microbatch gradients without storing full copies.

    Leaf hooks see the incoming gradient before PyTorch adds it to an existing
    accumulated ``parameter.grad``. This lets us compute the exact dot product
    between the current microbatch and the running accumulator using scalar
    reductions instead of another model-sized gradient buffer.
    """

    def __init__(self, parameters, enabled=True):
        self.parameters = [parameter for parameter in parameters if parameter.requires_grad]
        self.enabled = bool(enabled)
        self.active = False
        self.current_squared_norm_terms = []
        self.accumulator_dot_terms = []
        self.previous_accumulator_norm = None
        self.handles = []
        if self.enabled:
            for parameter in self.parameters:
                self.handles.append(parameter.register_hook(self._hook_for(parameter)))

    def _hook_for(self, parameter):
        def hook(gradient):
            if not self.active:
                return gradient
            gradient_fp32 = gradient.detach().float()
            self.current_squared_norm_terms.append(gradient_fp32.square().sum())
            if parameter.grad is not None:
                previous = parameter.grad.detach().float()
                self.accumulator_dot_terms.append((previous * gradient_fp32).sum())
            return gradient

        return hook

    def begin(self):
        if not self.enabled:
            return
        self.current_squared_norm_terms = []
        self.accumulator_dot_terms = []
        self.previous_accumulator_norm = _global_gradient_l2_norm(self.parameters).detach()
        self.active = True

    def finish(self, unscale_factor=1.0):
        if not self.enabled:
            return None
        self.active = False
        if not self.current_squared_norm_terms:
            return None

        current_squared_norm = torch.stack(self.current_squared_norm_terms).sum()
        current_norm = current_squared_norm.sqrt()
        cosine = None
        previous_norm = self.previous_accumulator_norm
        if (
            previous_norm is not None
            and previous_norm.item() > 0.0
            and self.accumulator_dot_terms
            and current_norm.item() > 0.0
        ):
            dot = torch.stack(self.accumulator_dot_terms).sum()
            cosine = float((dot / (previous_norm * current_norm)).clamp(-1, 1).item())

        return {
            # Backward receives loss / accumulation_steps. Undo that scale so
            # this is the norm of the microbatch's own mean loss gradient.
            "norm": float(current_norm.item() * unscale_factor),
            "cosine_to_running_accumulator": cosine,
        }

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _create_torch_profiler(cfg, experiment_path, global_rank):
    profiler_cfg = cfg.trainer.get("profiler", {})
    if not profiler_cfg or not bool(profiler_cfg.get("enabled", False)):
        return None

    trace_dir = Path(experiment_path) / "profiler" / f"rank_{global_rank}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(str(trace_dir))

    def trace_ready(profiler):
        tensorboard_trace_handler(profiler)
        averages = profiler.key_averages()
        sort_key = "self_device_time_total" if torch.cuda.is_available() else "self_cpu_time_total"
        summary = averages.table(sort_by=sort_key, row_limit=100)
        summary_path = trace_dir / f"summary_step_{profiler.step_num:06d}.txt"
        summary_path.write_text(summary + "\n", encoding="utf-8")
        flops = {
            "profile_step": int(profiler.step_num),
            "supported_operator_flops": int(
                sum(int(event.flops or 0) for event in averages)
            ),
            "summed_self_cuda_time_us": float(
                sum(float(event.self_device_time_total) for event in averages)
            ),
        }
        (trace_dir / f"flops_step_{profiler.step_num:06d}.json").write_text(
            json.dumps(flops, indent=2) + "\n", encoding="utf-8"
        )
        if torch.cuda.is_available() and bool(profiler_cfg.get("profile_memory", True)):
            memory_summary = averages.table(
                sort_by="self_device_memory_usage", row_limit=200
            )
            (trace_dir / f"memory_summary_step_{profiler.step_num:06d}.txt").write_text(
                memory_summary + "\n", encoding="utf-8"
            )
            profiler.export_memory_timeline(
                str(trace_dir / f"memory_timeline_step_{profiler.step_num:06d}.html"),
                device=torch.device("cuda", torch.cuda.current_device()),
            )
            profiler.export_memory_timeline(
                str(trace_dir / f"memory_timeline_step_{profiler.step_num:06d}.raw.json.gz"),
                device=torch.device("cuda", torch.cuda.current_device()),
            )
        logging.info("Wrote torch.profiler trace and summary to %s", trace_dir)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(
            wait=int(profiler_cfg.get("wait", 0)),
            warmup=int(profiler_cfg.get("warmup", 1)),
            active=int(profiler_cfg.get("active", 2)),
            repeat=int(profiler_cfg.get("repeat", 1)),
        ),
        on_trace_ready=trace_ready,
        record_shapes=bool(profiler_cfg.get("record_shapes", True)),
        profile_memory=bool(profiler_cfg.get("profile_memory", True)),
        with_stack=bool(profiler_cfg.get("with_stack", False)),
        with_flops=bool(profiler_cfg.get("with_flops", True)),
    )


def _cuda_storage_bytes(value) -> int:
    seen = set()

    def visit(item):
        if isinstance(item, torch.Tensor):
            if item.device.type != "cuda" or item.numel() == 0:
                return 0
            storage = item.untyped_storage()
            key = (item.device.index, storage._cdata)
            if key in seen:
                return 0
            seen.add(key)
            return int(storage.nbytes())
        if dataclasses.is_dataclass(item):
            return sum(visit(getattr(item, field.name)) for field in dataclasses.fields(item))
        if isinstance(item, dict):
            return sum(visit(entry) for entry in item.values())
        if isinstance(item, (list, tuple, set)):
            return sum(visit(entry) for entry in item)
        return 0

    return visit(value)


class _CudaMemoryRecorder:
    def __init__(self, cfg, experiment_path, global_rank, model, optimizer):
        profile_cfg = cfg.trainer.get("memory_profile", {})
        self.enabled = bool(profile_cfg.get("enabled", False))
        self.start_step = int(profile_cfg.get("start_step", 1))
        self.output_dir = Path(experiment_path) / "memory_profile" / f"rank_{global_rank}"
        self.global_rank = int(global_rank)
        self.model = model
        self.optimizer = optimizer
        self.records = []
        self.groups = []
        self.current_group = None
        self.history_started = False
        self.saved_refs = {}
        self.saved_live_bytes = 0
        self.saved_peak_bytes = 0
        self.saved_unique_bytes = 0
        self.saved_reference_bytes = 0
        self.saved_sites = {}
        self.saved_shapes = {}

    def active(self, completed_steps) -> bool:
        return self.enabled and int(completed_steps) == self.start_step

    def begin_group(self, completed_steps, group_index, batch, loader_wait_seconds):
        if not self.active(completed_steps):
            self.current_group = None
            return
        if not self.history_started:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            torch.cuda.memory._record_memory_history(max_entries=100000)
            self.history_started = True
        self.current_group = {
            "optimizer_step": int(completed_steps),
            "group_index": int(group_index),
            "scenes": list(batch.seq_name),
            "batch_size": int(batch.video.shape[0]),
            "views": int(batch.video.shape[1]),
            "frames": int(batch.video.shape[2]),
            "padded_trajectories": int(batch.trajectory.shape[-2]),
            "real_trajectories": [
                int((~mask.bool()).sum().item()) for mask in batch.track_padding_mask
            ],
            "loader_wait_seconds": float(loader_wait_seconds),
            "host_to_device_seconds": None,
            "forward_seconds": None,
            "backward_seconds": None,
            "optimizer_seconds": None,
        }
        self.group_started_at = time.perf_counter()
        self.saved_refs = {}
        self.saved_live_bytes = 0
        self.saved_peak_bytes = 0
        self.saved_unique_bytes = 0
        self.saved_reference_bytes = 0
        self.saved_sites = {}
        self.saved_shapes = {}
        torch.cuda.reset_peak_memory_stats()

    def _saved_callsite(self):
        for frame in reversed(traceback.extract_stack(limit=32)):
            if "/mvtracker/" in frame.filename and not frame.filename.endswith("cli/train.py"):
                return f"{Path(frame.filename).name}:{frame.lineno}:{frame.name}"
        return "unknown"

    def _pack_saved_tensor(self, tensor):
        if tensor.device.type != "cuda" or tensor.numel() == 0:
            return tensor
        storage = tensor.untyped_storage()
        key = (tensor.device.index, storage._cdata)
        size = int(storage.nbytes())
        entry = self.saved_refs.get(key)
        if entry is None:
            site = self._saved_callsite()
            self.saved_refs[key] = [1, size]
            self.saved_live_bytes += size
            self.saved_unique_bytes += size
            self.saved_sites[site] = self.saved_sites.get(site, 0) + size
        else:
            entry[0] += 1
        self.saved_reference_bytes += int(tensor.numel() * tensor.element_size())
        shape_key = f"{str(tensor.dtype).removeprefix('torch.')}:{tuple(tensor.shape)}"
        self.saved_shapes[shape_key] = self.saved_shapes.get(shape_key, 0) + int(
            tensor.numel() * tensor.element_size()
        )
        self.saved_peak_bytes = max(self.saved_peak_bytes, self.saved_live_bytes)
        return tensor

    def _unpack_saved_tensor(self, tensor):
        if tensor.device.type != "cuda" or tensor.numel() == 0:
            return tensor
        storage = tensor.untyped_storage()
        key = (tensor.device.index, storage._cdata)
        entry = self.saved_refs[key]
        entry[0] -= 1
        if entry[0] == 0:
            self.saved_live_bytes -= entry[1]
            del self.saved_refs[key]
        return tensor

    def saved_tensors(self):
        if self.current_group is None:
            return contextlib.nullcontext()
        return torch.autograd.graph.saved_tensors_hooks(
            self._pack_saved_tensor, self._unpack_saved_tensor
        )

    def capture(self, stage, batch=None):
        if self.current_group is None:
            return
        torch.cuda.synchronize()
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        parameter_bytes = _cuda_storage_bytes(tuple(self.model.parameters()))
        gradient_bytes = _cuda_storage_bytes(
            tuple(parameter.grad for parameter in self.model.parameters())
        )
        optimizer_bytes = _cuda_storage_bytes(self.optimizer.state)
        batch_bytes = _cuda_storage_bytes(batch) if batch is not None else 0
        allocated_bytes = int(torch.cuda.memory_allocated())
        reserved_bytes = int(torch.cuda.memory_reserved())
        known_bytes = parameter_bytes + gradient_bytes + optimizer_bytes + batch_bytes
        record = {
            **self.current_group,
            "stage": str(stage),
            "seconds_since_group_start": time.perf_counter() - self.group_started_at,
            "allocated_bytes": allocated_bytes,
            "reserved_bytes": reserved_bytes,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "device_used_bytes": int(total_bytes - free_bytes),
            "parameter_bytes": parameter_bytes,
            "gradient_bytes": gradient_bytes,
            "optimizer_state_bytes": optimizer_bytes,
            "batch_bytes": batch_bytes,
            "unattributed_allocated_bytes": max(0, allocated_bytes - known_bytes),
            "allocator_cache_bytes": max(0, reserved_bytes - allocated_bytes),
            "non_pytorch_estimate_bytes": max(0, int(total_bytes - free_bytes) - reserved_bytes),
            "saved_tensor_live_bytes": self.saved_live_bytes,
            "saved_tensor_peak_bytes": self.saved_peak_bytes,
        }
        self.records.append(record)
        logging.info("MEMORY_PROFILE %s", json.dumps(record, sort_keys=True))

    def set_host_to_device_seconds(self, seconds):
        if self.current_group is not None:
            self.current_group["host_to_device_seconds"] = float(seconds)

    def end_group(self, forward_seconds, backward_seconds, optimizer_seconds=None):
        if self.current_group is None:
            return
        self.current_group["forward_seconds"] = float(forward_seconds)
        self.current_group["backward_seconds"] = float(backward_seconds)
        if optimizer_seconds is not None:
            self.current_group["optimizer_seconds"] = float(optimizer_seconds)
        self.current_group["group_total_seconds"] = (
            time.perf_counter() - self.group_started_at
        )
        self.groups.append(
            {
                **self.current_group,
                "saved_tensor_peak_bytes": self.saved_peak_bytes,
                "saved_tensor_unique_bytes": self.saved_unique_bytes,
                "saved_tensor_reference_bytes": self.saved_reference_bytes,
                "saved_tensor_live_after_backward_bytes": self.saved_live_bytes,
                "top_saved_sites": sorted(
                    self.saved_sites.items(), key=lambda item: item[1], reverse=True
                )[:50],
                "top_saved_shapes": sorted(
                    self.saved_shapes.items(), key=lambda item: item[1], reverse=True
                )[:50],
            }
        )
        self.current_group = None

    def finish(self):
        if not self.history_started:
            return
        torch.cuda.synchronize()
        torch.cuda.memory._dump_snapshot(
            str(self.output_dir / "allocator_snapshot.pickle")
        )
        torch.cuda.memory._record_memory_history(enabled=None)
        payload = {
            "rank": self.global_rank,
            "start_step": self.start_step,
            "records": self.records,
            "groups": self.groups,
        }
        (self.output_dir / "memory_accounting.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        self.history_started = False


def fetch_optimizer(trainer_cfg, model):
    """Create the optimizer and learning rate scheduler"""
    optimizer = optim.AdamW(model.parameters(), lr=trainer_cfg.lr, weight_decay=trainer_cfg.wdecay)
    if trainer_cfg.anneal_strategy in ["linear", "cos"]:
        schedule_steps = int(trainer_cfg.lr_schedule_steps)
        if schedule_steps < int(trainer_cfg.num_steps):
            raise ValueError("trainer.lr_schedule_steps must be at least trainer.num_steps")
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            trainer_cfg.lr,
            schedule_steps,
            pct_start=0.05,
            cycle_momentum=False,
            anneal_strategy=trainer_cfg.anneal_strategy,
            div_factor=25.0,
            final_div_factor=1.0e4,
        )
    elif trainer_cfg.anneal_strategy == "restarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5000,
            T_mult=1,
            eta_min=trainer_cfg.lr / 1000,
        )

    return optimizer, scheduler


def _reduce_scalar(fabric, value, reduce_op="mean"):
    """Return the mean scalar across ranks on every rank."""
    tensor = torch.as_tensor(value, device=fabric.device, dtype=torch.float64)
    return float(fabric.all_reduce(tensor, reduce_op=reduce_op).item())


def _reduce_scalar_dict(fabric, values, reduce_op="mean"):
    """Reduce a same-key scalar mapping, including future per-source metrics."""
    return {
        name: _reduce_scalar(fabric, value, reduce_op=reduce_op)
        for name, value in values.items()
    }


def _gather_rank_metrics(fabric, values):
    """Gather one scalar mapping from every rank without averaging ranks away."""
    names = tuple(sorted(values))
    local = torch.tensor(
        [values[name] for name in names],
        device=fabric.device,
        dtype=torch.float64,
    )
    gathered = fabric.all_gather(local).reshape(fabric.world_size, len(names))
    return {
        f"hardware/gpu_{rank}/{name}": float(gathered[rank, index].item())
        for rank in range(fabric.world_size)
        for index, name in enumerate(names)
    }


def _throughput_metrics(step_seconds, global_samples, global_trajectories):
    return {
        "performance/global_samples": global_samples,
        "performance/global_trajectories": global_trajectories,
        "performance/samples_per_second": global_samples / step_seconds,
        "performance/trajectories_per_second": (
            global_trajectories / step_seconds
        ),
    }


def _all_ranks_succeeded(fabric, succeeded):
    value = torch.tensor(int(bool(succeeded)), device=fabric.device)
    return bool(fabric.all_reduce(value, reduce_op="min").item())


def _latest_checkpoint_path(experiment_path):
    manifest_path = Path(experiment_path) / LATEST_CHECKPOINT_MANIFEST
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(experiment_path) / manifest["checkpoint"]
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"latest checkpoint manifest points to missing file: {checkpoint_path}"
        )
    return checkpoint_path


def _write_latest_checkpoint_manifest(experiment_path, checkpoint_path, completed_steps):
    manifest_path = Path(experiment_path) / LATEST_CHECKPOINT_MANIFEST
    temporary_path = manifest_path.with_suffix(".tmp")
    temporary_path.write_text(
        json.dumps(
            {
                "checkpoint": Path(checkpoint_path).name,
                "completed_steps": int(completed_steps),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)


def _get_or_create_wandb_run_id(experiment_path, configured_run_id=None):
    run_id_path = Path(experiment_path) / WANDB_RUN_ID_FILE
    if run_id_path.is_file():
        persisted_run_id = run_id_path.read_text(encoding="utf-8").strip()
        if configured_run_id and configured_run_id != persisted_run_id:
            raise ValueError("configured W&B run ID does not match wandb_run_id.txt")
        return persisted_run_id
    run_id = configured_run_id or wandb.util.generate_id()
    run_id_path.write_text(run_id + "\n", encoding="utf-8")
    return run_id


def _save_training_checkpoint(
    fabric,
    experiment_path,
    checkpoint_path,
    model,
    optimizer,
    scheduler,
    completed_steps,
    master_seed,
    wandb_run_id,
    mixed_schedule_state=None,
    source_cursors=None,
):
    state = AttributeDict(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        total_steps=int(completed_steps),
        master_seed=int(master_seed),
        wandb_run_id=wandb_run_id or "",
    )
    if mixed_schedule_state is not None:
        state.mixed_schedule_state = mixed_schedule_state
        state.source_cursors = dict(source_cursors)
    fabric.save(checkpoint_path, state)
    fabric.barrier()
    if fabric.global_rank == 0:
        _write_latest_checkpoint_manifest(
            experiment_path,
            checkpoint_path,
            completed_steps,
        )
    fabric.barrier()


class _ScheduledSourceSampler(torch.utils.data.Sampler):
    """Feed one source loader deterministic rank-local scene requests."""

    def __init__(self, schedule, source, rank, request_count):
        self.schedule = schedule
        self.source = source
        self.rank = int(rank)
        self.request_count = int(request_count)
        self.start_cursor = 0

    def __len__(self):
        return self.request_count

    def set_start_cursor(self, cursor):
        self.start_cursor = int(cursor)

    def __iter__(self):
        for offset in range(self.request_count):
            yield self.schedule.sample_source(
                self.source,
                self.start_cursor + offset,
                self.rank,
            ).request


def _mixed_source_name(cfg):
    return cfg.datasets.train.name == "mixed-diegesis-mvkubric-training"


def _eval_dataset_names_for_step(cfg, step):
    """Return the configured evaluation datasets for one completed step."""
    schedule = cfg.datasets.eval.get("schedule")
    if schedule is None:
        return tuple(cfg.datasets.eval.names)
    for entry in schedule:
        if int(step) in {int(value) for value in entry.steps}:
            return tuple(entry.names)
    raise ValueError(f"no evaluation dataset schedule entry configured for step {step}")


def _planned_physical_batching(cfg):
    settings = cfg.datasets.train.get("physical_batching")
    return settings is not None and bool(settings.get("enabled", False))


def _mvkubric_dali_stream(cfg):
    return str(cfg.datasets.train.get("mvkubric_storage", "native")) == "dali_stream"


def _physical_batch_capacity(cfg):
    settings = cfg.datasets.train.physical_batching
    return BatchCapacity(
        name=str(settings.capacity_name),
        rank_count=2,
        logical_scenes_per_rank=int(cfg.trainer.gradient_accumulation_steps),
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


def _assert_matching_step_fingerprint(fabric, fingerprint):
    digest = torch.tensor(
        list(bytes.fromhex(fingerprint)),
        dtype=torch.uint8,
        device=fabric.device,
    )
    gathered = fabric.all_gather(digest)
    if gathered.ndim == 1:
        gathered = gathered[None]
    if not torch.equal(gathered, gathered[0:1].expand_as(gathered)):
        raise RuntimeError("planned physical step differs between DDP ranks")


def _build_training_dataset(dataset_name, dataset_root, cfg, fabric, source_cfg=None):
    include_scene_ids = (
        source_cfg.get("include_scene_ids") if source_cfg is not None else None
    )
    exclude_scene_ids = (
        source_cfg.get("exclude_scene_ids", ()) if source_cfg is not None else ()
    )
    if dataset_name.startswith("kubric-multiview-v3"):
        dataset = KubricMultiViewDataset.from_name(
            dataset_name,
            dataset_root,
            cfg,
            fabric,
            include_scene_ids=include_scene_ids,
            exclude_scene_ids=exclude_scene_ids,
        )
        return dataset
    if dataset_name.startswith("pointodyssey-multiview-"):
        return PointOdysseyMultiViewDataset.from_name(
            dataset_name, dataset_root, cfg, fabric
        )
    if dataset_name.startswith("tapvid3d-multiview-"):
        kwargs = TapVid3DMultiViewDataset.from_name(
            dataset_name,
            dataset_root,
            cfg,
            fabric,
            just_return_kwargs=True,
            include_scene_ids=include_scene_ids,
            exclude_scene_ids=exclude_scene_ids,
        )
        if source_cfg is not None and "view_count_probabilities" in source_cfg:
            kwargs["view_count_probabilities"] = tuple(
                source_cfg.view_count_probabilities
            )
        return TapVid3DMultiViewDataset(**kwargs)
    raise ValueError(f"Dataset {dataset_name} not supported for training")


def _source_batch_shape_metrics(batch):
    view_count = int(batch.video.shape[1])
    if batch.track_padding_mask is not None:
        padding_mask = batch.track_padding_mask.to(dtype=torch.bool)
        track_count = float((~padding_mask).sum(dim=1).float().mean().item())
    else:
        track_count = float(batch.trajectory.shape[-2])
    return view_count, track_count


def _build_source_train_loader(dataset, sampler, cfg, fabric):
    requires_cuda_prefetch = getattr(dataset, "requires_cuda_prefetch", False)
    if requires_cuda_prefetch:
        torch.multiprocessing.set_sharing_strategy("file_system")
    loader_type = torch.utils.data.DataLoader if requires_cuda_prefetch else StatefulDataLoader
    loader_kwargs = {}
    if not requires_cuda_prefetch:
        loader_kwargs["in_order"] = cfg.reproducibility.deterministic
    elif cfg.datasets.train.num_workers > 0:
        loader_kwargs["multiprocessing_context"] = "spawn"
    loader = loader_type(
        dataset,
        batch_size=int(cfg.datasets.train.batch_size),
        sampler=sampler,
        num_workers=cfg.datasets.train.num_workers,
        pin_memory=True,
        collate_fn=getattr(dataset, "collate_fn", collate_fn),
        prefetch_factor=(
            cfg.datasets.train.get("prefetch_factor", 2)
            if cfg.datasets.train.num_workers > 0 else None
        ),
        persistent_workers=cfg.datasets.train.num_workers > 0,
        **loader_kwargs,
    )
    loader = fabric.setup_dataloaders(
        loader,
        move_to_device=not requires_cuda_prefetch,
        use_distributed_sampler=False,
    )
    if requires_cuda_prefetch:
        loader = CudaPrefetchLoader(
            loader,
            device=fabric.device,
            timing_interval=int(cfg.trainer.expensive_diagnostics_interval),
            queue_depth=int(cfg.datasets.train.cuda_prefetch_queue_depth),
            decode_batch_size=int(cfg.datasets.train.cuda_decode_batch_size),
        )
    return loader


def _start_mixed_source_iterators(train_loaders):
    raw_iterators = {
        source: iter(loader.loader) for source, loader in train_loaders.items()
    }
    return {
        source: loader.iter_from(raw_iterators[source])
        for source, loader in train_loaders.items()
    }


def _load_mixed_step(
    fabric,
    source_pattern,
    data_iters,
    source_samplers,
    train_loaders,
    source_cursors,
):
    """Materialize one rank's complete mixed optimizer step concurrently."""
    sources = tuple(dict.fromkeys(source_pattern))
    counts = {source: source_pattern.count(source) for source in sources}

    def load_source(source, count):
        batches = []
        for _ in range(count):
            try:
                batches.append(next(data_iters[source]))
            except StopIteration:
                source_samplers[source].set_start_cursor(
                    source_cursors[source] + len(batches)
                )
                data_iters[source] = iter(train_loaders[source])
                batches.append(next(data_iters[source]))
        return batches

    with ThreadPoolExecutor(max_workers=len(sources)) as executor:
        futures = {
            source: executor.submit(load_source, source, counts[source])
            for source in sources
        }
        pending = {source: list(futures[source].result()) for source in sources}

    accepted = {}
    loaded_count = 0
    failed_count = 0
    for source in sources:
        accepted[source] = []
        attempt = 0
        while len(accepted[source]) < counts[source]:
            if pending[source]:
                batch, gotit = pending[source].pop(0)
            else:
                batch, gotit = load_source(source, 1)[0]
            source_cursors[source] += 1
            loaded_count += 1
            if _all_ranks_succeeded(fabric, all(gotit)):
                if batch.sample_metadata:
                    for metadata in batch.sample_metadata:
                        metadata["paired_retry_attempt"] = attempt
                accepted[source].append(batch)
                attempt = 0
            else:
                failed_count += 1
                attempt += 1

    positions = {source: 0 for source in sources}
    ordered = []
    for source in source_pattern:
        ordered.append((source, accepted[source][positions[source]]))
        positions[source] += 1
    return ordered, loaded_count, failed_count


def _run_eval(fabric, cfg, evaluator, model, dataloaders, writer, step):
    """Run validation across ranks and write the combined metrics once."""
    fabric.barrier()
    scheduled_names = tuple(_eval_dataset_names_for_step(cfg, step))
    dataloaders = [
        (name, dataloader)
        for name, dataloader in dataloaders
        if name in scheduled_names
    ]
    if _mixed_source_name(cfg) and fabric.world_size > 1:
        local_metrics = run_test_eval(
            cfg,
            evaluator,
            _unwrap_objects(model),
            dataloaders,
            None,
            step,
            write_results=False,
        )
        gathered_metrics = [None] * fabric.world_size
        torch.distributed.all_gather_object(gathered_metrics, local_metrics)
        if fabric.global_rank == 0:
            combined_metrics = {}
            for dataset_name in scheduled_names:
                scenes = [
                    scene_metrics
                    for rank_metrics in gathered_metrics
                    for scene_metrics in rank_metrics.get(dataset_name, {}).values()
                ]
                combined_metrics[dataset_name] = dict(enumerate(scenes))
            _write_eval_metrics(cfg, writer, step, combined_metrics)
    elif fabric.global_rank == 0:
        run_test_eval(
            cfg,
            evaluator,
            _unwrap_objects(model),
            dataloaders,
            writer,
            step,
        )
    fabric.barrier()


def _scene_scale(track_upscaling_factor, scene_index, batch_size, device, dtype):
    """Return one scene's world-space loss scale as a device tensor."""
    if torch.is_tensor(track_upscaling_factor):
        if track_upscaling_factor.ndim == 0:
            value = track_upscaling_factor
        elif track_upscaling_factor.shape[0] == batch_size:
            value = track_upscaling_factor[scene_index]
        else:
            raise ValueError(
                "track_upscaling_factor must be scalar or have one value per scene"
            )
    elif isinstance(track_upscaling_factor, (list, tuple)):
        if len(track_upscaling_factor) != batch_size:
            raise ValueError(
                "track_upscaling_factor must be scalar or have one value per scene"
            )
        value = track_upscaling_factor[scene_index]
    else:
        value = track_upscaling_factor
    return torch.as_tensor(value, device=device, dtype=dtype)


def _scene_index_tensor(value, scene_index, batch_size, device):
    """Select a scene's track permutation while accepting list/tensor records."""
    if torch.is_tensor(value):
        if value.ndim > 1 and value.shape[0] == batch_size:
            value = value[scene_index]
        return value.to(device=device, dtype=torch.long)
    if isinstance(value, (list, tuple)):
        if len(value) == batch_size and not all(
            isinstance(item, (int, np.integer)) for item in value
        ):
            value = value[scene_index]
        return torch.as_tensor(value, device=device, dtype=torch.long)
    return torch.as_tensor(value, device=device, dtype=torch.long)


def _scene_prediction_tree(value, scene_index, batch_size):
    """Keep a prediction tree's leading scene dimension at one."""
    if torch.is_tensor(value):
        if value.ndim > 0 and value.shape[0] == batch_size:
            return value[scene_index:scene_index + 1]
        if value.ndim > 0 and value.shape[0] == 1:
            return value
        return value.unsqueeze(0)
    if isinstance(value, (list, tuple)):
        return type(value)(
            _scene_prediction_tree(item, scene_index, batch_size)
            for item in value
        )
    raise TypeError(f"Unsupported prediction value: {type(value)!r}")


def _scene_records(train_data, query_points_3d, batch_size, window_len):
    """Normalize new per-scene records and retain the legacy B=1 contract."""
    records = train_data.get("scenes")
    if records is not None:
        if len(records) != batch_size:
            raise ValueError(
                f"model returned {len(records)} scene records for batch size {batch_size}"
            )
        return records

    if batch_size != 1:
        raise ValueError(
            "batched training requires model train_data['scenes']; "
            "the legacy window metadata only supports batch size 1"
        )
    query_times = query_points_3d[0, :, 0].long()
    start = int(query_times.min().item())
    p_idx_end_list = train_data["p_idx_end_list"]
    return [{
        "sort_inds": train_data["sort_inds"],
        "window_starts": [
            start + i * (window_len // 2)
            for i in range(len(p_idx_end_list))
        ],
        "p_idx_end_list": p_idx_end_list,
        "coord_predictions": train_data["coord_predictions"],
        "vis_predictions": train_data["vis_predictions"],
    }]


def _window_targets(
    scene_record,
    scene_index,
    batch_size,
    gt_visibility,
    gt_trajectory,
    valid_tracks,
    query_points_3d,
    window_len,
):
    """Build one scene's loss windows from the model's schedule metadata."""
    device = query_points_3d.device
    sort_inds = _scene_index_tensor(
        scene_record["sort_inds"], scene_index, batch_size, device
    )
    gt_visibility = gt_visibility[scene_index:scene_index + 1].index_select(2, sort_inds)
    gt_trajectory = gt_trajectory[scene_index:scene_index + 1].index_select(2, sort_inds)
    valid_tracks = valid_tracks[scene_index:scene_index + 1].index_select(2, sort_inds)

    starts = [int(value) for value in scene_record["window_starts"]]
    p_idx_end_list = [int(value) for value in scene_record["p_idx_end_list"]]
    if len(starts) != len(p_idx_end_list):
        raise ValueError("window_starts and p_idx_end_list must have equal lengths")

    vis_gts, traj_gts, valids_gts = [], [], []
    for start, p_idx_end in zip(starts, p_idx_end_list):
        vis_gts.append(gt_visibility[:, start:start + window_len, :p_idx_end])
        traj_gts.append(gt_trajectory[:, start:start + window_len, :p_idx_end])
        valids_gts.append(valid_tracks[:, start:start + window_len, :p_idx_end])
    return sort_inds, vis_gts, traj_gts, valids_gts


def _project_prediction_batch(pred_trajectories, intrs, extrs):
    """Project B world-space predictions while retaining the batch dimension."""
    predictions = []
    for scene_index in range(pred_trajectories.shape[0]):
        predictions.append(torch.stack([
            torch.cat(world_space_to_pixel_xy_and_camera_z(
                world_xyz=pred_trajectories[scene_index],
                intrs=intrs[scene_index, view_index],
                extrs=extrs[scene_index, view_index],
            ), dim=-1)
            for view_index in range(intrs.shape[1])
        ], dim=0))
    return torch.stack(predictions, dim=0)


def _assert_real_tracks_visible(
    visible_any_view: torch.Tensor,
    track_padding_mask: torch.Tensor | None = None,
) -> None:
    """Check visibility only for real tracks, ignoring padded batch slots."""
    visible = visible_any_view.any(dim=1)
    if track_padding_mask is not None:
        visible = visible | track_padding_mask.bool()
    assert visible.all(), "All real points should be visible in at least one frame."


def build_model_execution_schedule(batch):
    query_points = batch.query_points_3d.detach().cpu()
    padding = getattr(batch, "track_padding_mask", None)
    if padding is None:
        padding = torch.zeros(query_points.shape[:2], dtype=torch.bool)
    else:
        padding = padding.detach().cpu().bool()
    query_times = []
    real_track_counts = []
    schedule_starts = []
    for scene_index in range(query_points.shape[0]):
        real_times = query_points[scene_index, ~padding[scene_index], 0].long()
        sorted_times = torch.sort(real_times).values.tolist()
        query_times.append(sorted_times)
        real_track_counts.append(len(sorted_times))
        schedule_starts.append(sorted_times[0])
    active_counts = []
    window_start = schedule_starts[0]
    frame_count = int(batch.video.shape[2])
    window_length = 12
    while window_start < frame_count - window_length // 2:
        active_counts.append([
            int(np.searchsorted(times, window_start + window_length))
            for times in query_times
        ])
        window_start += window_length // 2
    return {
        "schedule_starts": schedule_starts,
        "real_track_counts": real_track_counts,
        "query_times": query_times,
        "active_counts": active_counts,
    }


def build_camera_inverses(batch):
    intrs_inv = torch.inverse(batch.intrs.float()).type(batch.intrs.dtype)
    batch_size, views, frames = batch.extrs.shape[:3]
    extrs_square = torch.eye(
        4, device=batch.extrs.device, dtype=batch.extrs.dtype
    )[None, None, None].repeat(batch_size, views, frames, 1, 1)
    extrs_square[:, :, :, :3, :] = batch.extrs
    extrs_inv = torch.inverse(extrs_square.float()).type(batch.extrs.dtype)
    return intrs_inv, extrs_inv


def build_pointcloud_grids(batch, stride, levels):
    height = int(batch.video.shape[-2]) // int(stride)
    width = int(batch.video.shape[-1]) // int(stride)
    grids = []
    for level in range(int(levels)):
        level_height = height // 2**level
        level_width = width // 2**level
        level_stride = int(stride) * 2**level
        y, x = torch.meshgrid(
            (torch.arange(level_height, device=batch.video.device) + 0.5)
            * level_stride
            - 0.5,
            (torch.arange(level_width, device=batch.video.device) + 0.5)
            * level_stride
            - 0.5,
            indexing="ij",
        )
        grids.append(torch.stack((x, y, torch.ones_like(x)), dim=-1).float())
    return tuple(grids)


def forward_batch_multi_view(
        batch,
        model,
        cfg,
        step,
        train_iters,
        gamma,
        save_debug_logs=False,
        debug_logs_path='',
        run_expensive_diagnostics=True,
        capture_training_trace=False,
        execution_schedule=None,
        graph_capture=False,
        camera_inverses=None,
        pointcloud_grids=None,
):
    # Per view data
    rgbs = batch.video
    depths = batch.videodepth
    image_features = batch.feats
    intrs = batch.intrs
    extrs = batch.extrs
    gt_trajectories_2d_pixelspace_w_z_cameraspace = batch.trajectory
    gt_visibilities_per_view = batch.visibility
    query_points_3d = batch.query_points_3d

    # Non-per-view data
    gt_trajectories_3d_worldspace = batch.trajectory_3d
    valid_tracks_per_frame = batch.valid
    track_upscaling_factor = batch.track_upscaling_factor
    track_padding_mask = getattr(batch, "track_padding_mask", None)
    if track_padding_mask is not None:
        track_padding_mask = track_padding_mask.bool()

    batch_size, num_views, num_frames, _, height, width = rgbs.shape
    num_points = gt_trajectories_2d_pixelspace_w_z_cameraspace.shape[3]

    # Assert shapes of per-view data
    assert rgbs.shape == (batch_size, num_views, num_frames, 3, height, width)
    assert depths.shape == (batch_size, num_views, num_frames, 1, height, width)
    assert intrs.shape == (batch_size, num_views, num_frames, 3, 3)
    assert extrs.shape == (batch_size, num_views, num_frames, 3, 4)
    assert gt_trajectories_2d_pixelspace_w_z_cameraspace.shape == (batch_size, num_views, num_frames, num_points, 3)
    assert gt_visibilities_per_view.shape == (batch_size, num_views, num_frames, num_points)

    # Assert shapes of non-per-view data
    assert query_points_3d.shape == (batch_size, num_points, 4)
    assert gt_trajectories_3d_worldspace.shape == (batch_size, num_frames, num_points, 3)
    assert valid_tracks_per_frame.shape == (batch_size, num_frames, num_points)

    gt_visibilities_any_view = gt_visibilities_per_view.any(dim=1)
    if save_debug_logs:
        _assert_real_tracks_visible(gt_visibilities_any_view, track_padding_mask)

    frame_indices = torch.arange(num_frames, device=query_points_3d.device)[None, :, None]
    query_frames = query_points_3d[:, :, 0].long()[:, None, :]
    valid_tracks_per_frame = valid_tracks_per_frame * (frame_indices >= query_frames)

    # Run the model
    results = model(
        rgbs=rgbs,
        depths=depths,
        image_features=image_features,
        query_points=query_points_3d,
        iters=train_iters,
        is_train=True,
        intrs=intrs,
        extrs=extrs,
        save_debug_logs=save_debug_logs,
        debug_logs_path=debug_logs_path,
        track_padding_mask=track_padding_mask,
        execution_schedule=execution_schedule,
        camera_inverses=camera_inverses,
        pointcloud_grids=pointcloud_grids,
        capture_safe=graph_capture,
    )
    pred_trajectories = results["traj_e"]
    pred_visibilities = results["vis_e"]
    train_data = results["train_data"]
    scene_records = _scene_records(
        train_data,
        query_points_3d,
        batch_size,
        cfg.model.sliding_window_len,
    )
    if track_padding_mask is not None:
        valid_tracks_per_frame = valid_tracks_per_frame * (
            ~track_padding_mask[:, None, :]
        )

    scene_losses = []
    scene_vis_losses = []
    scene_windows = []
    for scene_index, scene_record in enumerate(scene_records):
        sort_inds, vis_gts, traj_gts, valids_gts = _window_targets(
            scene_record,
            scene_index,
            batch_size,
            gt_visibilities_any_view,
            gt_trajectories_3d_worldspace,
            valid_tracks_per_frame,
            query_points_3d,
            cfg.model.sliding_window_len,
        )
        scene_coord_predictions = _scene_prediction_tree(
            scene_record["coord_predictions"], scene_index, batch_size
        )
        scene_vis_predictions = _scene_prediction_tree(
            scene_record["vis_predictions"], scene_index, batch_size
        )
        scene_windows.append((sort_inds, vis_gts, traj_gts, valids_gts,
                              scene_coord_predictions, scene_vis_predictions))
        scene_losses.append(
            sequence_loss_3d(
                scene_coord_predictions,
                traj_gts,
                vis_gts,
                valids_gts,
                gamma,
            ) * _scene_scale(
                track_upscaling_factor,
                scene_index,
                batch_size,
                query_points_3d.device,
                gt_trajectories_3d_worldspace.dtype,
            )
        )
        scene_vis_losses.append(
            balanced_ce_loss(scene_vis_predictions, vis_gts, valids_gts)
        )

    xyz_loss = torch.stack(scene_losses).mean()
    vis_loss = torch.stack(scene_vis_losses).mean()
    if save_debug_logs:
        first_scene = scene_windows[0]
        logging.info(
            f"[DEBUG] {step=} {track_upscaling_factor=} "
            f"{first_scene[4][0][0][0, 0, 0]=} "
            f"{first_scene[4][-1][0][0, 0, 0]=} "
            f"{first_scene[5][0][0, 0, 0]=} "
            f"{first_scene[5][-1][0, 0, 0]=}"
        )

    # Directly comparable no-motion baseline: keep every track at its queried
    # world-space coordinate for the full clip, then evaluate it with the exact
    # same sliding windows, valid masks, refinement weights, and Z scaling as
    # the model trajectory loss.
    diagnostic_metrics = {}
    if not graph_capture:
        if track_padding_mask is None:
            real_track_counts = torch.full(
                (batch_size,), num_points, device=query_points_3d.device
            )
        else:
            real_track_counts = (~track_padding_mask.bool()).sum(dim=1)
        real_track_mean, padded_track_mean, fill_fraction = torch.stack((
            real_track_counts.float().mean(),
            num_points - real_track_counts.float().mean(),
            real_track_counts.sum() / float(batch_size * num_points),
        )).detach().cpu().tolist()
        diagnostic_metrics.update({
            "batching/physical_scenes": float(batch_size),
            "batching/real_tracks_mean": real_track_mean,
            "batching/padded_tracks_per_scene": padded_track_mean,
            "batching/track_fill_fraction": fill_fraction,
            "batching/view_count": float(num_views),
        })
    if run_expensive_diagnostics:
        with torch.no_grad():
            stationary_losses = []
            for scene_index, (sort_inds, vis_gts, traj_gts, valids_gts,
                              scene_coord_predictions, _) in enumerate(scene_windows):
                query_xyz_sorted = query_points_3d[scene_index:scene_index + 1].index_select(
                    1, sort_inds
                )[..., 1:]
                stationary_predictions = []
                for window_predictions, window_gt in zip(scene_coord_predictions, traj_gts):
                    stationary = query_xyz_sorted[:, None, :window_gt.shape[2]].expand(
                        -1, window_gt.shape[1], -1, -1
                    )
                    stationary_predictions.append([
                        stationary.clone() for _ in window_predictions
                    ])
                stationary_losses.append(
                    sequence_loss_3d(
                        stationary_predictions,
                        traj_gts,
                        vis_gts,
                        valids_gts,
                        gamma,
                    ) * _scene_scale(
                        track_upscaling_factor,
                        scene_index,
                        batch_size,
                        query_points_3d.device,
                        gt_trajectories_3d_worldspace.dtype,
                    )
                )
            stationary_xyz_loss = torch.stack(stationary_losses).mean()
            model_to_stationary_ratio = xyz_loss.detach() / stationary_xyz_loss.clamp_min(1e-12)
            diagnostic_metrics = {
                "baseline/stationary_trajectory_loss": stationary_xyz_loss.item(),
                "baseline/model_to_stationary_ratio": model_to_stationary_ratio.item(),
            }

    # Compute 3DPT metrics
    # eval_3dpt_results_dict = evaluate_3dpt(
    #     gt_tracks=gt_trajectories_3d_worldspace[0].cpu().numpy(),
    #     gt_visibilities=gt_visibilities_any_view[0].cpu().numpy(),
    #     pred_tracks=pred_trajectories[0].detach().cpu().numpy(),
    #     pred_visibilities=(pred_visibilities[0] > 0.5).detach().cpu().numpy(),
    #     evaluation_setting="kubric-multiview",
    #     track_upscaling_factor=track_upscaling_factor,
    #     prefix="train_3dpt",
    #     verbose=False,
    #     query_points=query_points_3d[0].cpu().numpy(),
    # )

    # Project the predictions to pixel space
    pred_trajectories = pred_trajectories.detach()
    if pred_trajectories.ndim == 3:
        pred_trajectories = pred_trajectories.unsqueeze(0)
    if pred_visibilities.ndim == 2:
        pred_visibilities = pred_visibilities.unsqueeze(0)
    pred_trajectories_pixel_xy_camera_z_per_view = _project_prediction_batch(
        pred_trajectories,
        intrs,
        extrs,
    )
    if run_expensive_diagnostics:
        intrs_inv = torch.inverse(intrs.float())
        extrs_square = torch.eye(4, device=extrs.device)[None, None, None].expand(
            batch_size, num_views, num_frames, -1, -1
        ).clone()
        extrs_square[:, :, :, :3, :] = extrs
        extrs_inv = torch.inverse(extrs_square.float())
        for scene_index in range(batch_size):
            for view_idx in range(num_views):
                pred_trajectories_reproduced = pixel_xy_and_camera_z_to_world_space(
                    pixel_xy=pred_trajectories_pixel_xy_camera_z_per_view[
                        scene_index, view_idx, :, :, :2
                    ],
                    camera_z=pred_trajectories_pixel_xy_camera_z_per_view[
                        scene_index, view_idx, :, :, 2:
                    ],
                    intrs_inv=intrs_inv[scene_index, view_idx],
                    extrs_inv=extrs_inv[scene_index, view_idx],
                )
                if not torch.allclose(
                    pred_trajectories_reproduced,
                    pred_trajectories[scene_index],
                    atol=1,
                ):
                    warnings.warn(
                        "Reprojection of the predicted trajectories failed: "
                        f"scene_index={scene_index}, view_idx={view_idx}, "
                        f"max_diff={torch.max(torch.abs(pred_trajectories_reproduced - pred_trajectories[scene_index]))}"
                    )

    if save_debug_logs:
        logging.info(
            f"{step=}, "
            f"seq={batch.seq_name}, "
            f"{xyz_loss.item()=}, "
            f"{vis_loss.item()=}, "
        )

    output = {
        "flow": {
            "loss": xyz_loss * 1.0,
            "predictions": pred_trajectories_pixel_xy_camera_z_per_view,
            "predictions_worldspace": pred_trajectories,
        },
        "visibility": {
            "loss": vis_loss * cfg.trainer.visibility_loss_weight,
            "predictions": pred_visibilities.detach(),
        },
        "metrics": diagnostic_metrics,
        "scene_losses": {
            "flow": torch.stack(scene_losses).detach(),
            "visibility": (
                torch.stack(scene_vis_losses).detach()
                * cfg.trainer.visibility_loss_weight
            ),
        },
        # "metrics": {
        #     k: v
        #     for k, v in eval_3dpt_results_dict.items()
        #     if "per_track" not in k
        # },
    }
    if capture_training_trace:
        output["training_trace"] = {
            "coordinates": [
                prediction.detach()
                for scene in scene_records
                for window in scene["coord_predictions"]
                for prediction in window
            ],
            "visibility_logits": [
                prediction.detach()
                for scene in scene_records
                for prediction in scene["vis_predictions"]
            ],
        }
    return output


def _forward_backward_microbatch(
    *,
    fabric,
    model,
    batch,
    cfg,
    completed_steps,
    train_iters,
    gradient_accumulation_steps,
    is_final_microbatch,
    run_expensive_diagnostics,
    gradient_diagnostics,
    memory_recorder,
):
    """Run one microbatch, suppressing DDP synchronization until the last one."""
    with memory_recorder.saved_tensors():
        with fabric.no_backward_sync(model, enabled=not is_final_microbatch):
            forward_started_at = time.time()
            with torch.profiler.record_function("train/model_and_loss_forward"):
                output = forward_batch_multi_view(
                    batch=batch,
                    model=model,
                    cfg=cfg,
                    step=completed_steps,
                    train_iters=train_iters,
                    gamma=cfg.trainer.gamma,
                    save_debug_logs=(
                        is_final_microbatch
                        and (
                            completed_steps % cfg.trainer.viz_freq
                            == cfg.trainer.viz_freq - 1
                            or completed_steps in (0, 10, 100, cfg.trainer.num_steps - 1)
                        )
                    ),
                    debug_logs_path=os.path.join(
                        cfg.experiment_path,
                        "forward_pass__train_step-"
                        f"{completed_steps}_global_rank-{fabric.global_rank}",
                    ),
                    run_expensive_diagnostics=run_expensive_diagnostics,
                )

            loss = torch.zeros((), device=fabric.device)
            component_losses = {}
            metrics = {}
            scene_losses = None
            for name, value in output.items():
                if name == "metrics":
                    metrics.update({key: float(item) for key, item in value.items()})
                elif name == "scene_losses":
                    scene_losses = value
                elif "loss" in value:
                    loss = loss + value["loss"]
                    component_losses[name] = value["loss"].detach()
                else:
                    raise ValueError(f"Unknown key {name} in output")
            forward_duration = time.time() - forward_started_at
            memory_recorder.capture("after_forward", batch=batch)

            if run_expensive_diagnostics:
                gradient_diagnostics.begin()
            backward_started_at = time.time()
            with torch.profiler.record_function("train/backward"):
                fabric.backward(
                    _scale_microbatch_loss(loss, gradient_accumulation_steps)
                )
            microbatch_gradient = (
                gradient_diagnostics.finish(
                    unscale_factor=gradient_accumulation_steps,
                )
                if run_expensive_diagnostics
                else None
            )
            backward_duration = time.time() - backward_started_at
            memory_recorder.capture("after_backward", batch=batch)

    return (
        output,
        loss.detach(),
        component_losses,
        metrics,
        scene_losses,
        microbatch_gradient,
        forward_duration,
        backward_duration,
    )


def _write_eval_metrics(cfg, writer, step, metrics_by_dataset):
    for ds_name, metrics in metrics_by_dataset.items():
        if not metrics:
            continue
        first = next(iter(metrics.values()))
        metrics_to_log = {
            key: np.nanmean(
                [value[key] for value in metrics.values() if key in value]
            ).round(2)
            for key in first
        }
        if writer is not None:
            for key, value in metrics_to_log.items():
                writer.add_scalar(f"eval/{ds_name}/{key}", value, step)

        logging.info(f"Per-sequence Metrics for {ds_name}: {pd.DataFrame(metrics)}")
        logging.info(
            f"Average metrics for {ds_name}: {json.dumps(metrics_to_log, indent=4)}"
        )
        log_dir_ds = os.path.join(cfg.experiment_path, f"eval_{ds_name}")
        os.makedirs(log_dir_ds, exist_ok=True)
        frame = pd.DataFrame(metrics).T
        frame = frame.map(
            lambda value: value[0]
            if isinstance(value, np.ndarray) or isinstance(value, list)
            else value
        )
        frame.to_csv(f"{log_dir_ds}/step-{step}_metrics.csv")
        pd.DataFrame(metrics_to_log, index=["score"]).T.to_csv(
            f"{log_dir_ds}/step-{step}_metrics_avg.csv"
        )
        logging.info(f"Saved metrics to {log_dir_ds}/step-{step}_metrics_avg.csv")


def run_test_eval(cfg, evaluator, model, dataloaders, writer, step, *, write_results=True):
    if len(dataloaders) == 0:
        return {}

    logging.info(f"Eval – GPU usage A: {gpustat.new_query()}")

    log_dir = cfg.experiment_path
    model.eval()
    metrics_by_dataset = {}
    for ds_name, dataloader in dataloaders:
        if ds_name.startswith("kubric"):
            predictor_settings = cfg.evaluation.predictor_settings["kubric"]
        elif ds_name.startswith("dex-ycb"):
            predictor_settings = cfg.evaluation.predictor_settings["dex_ycb"]
        elif ds_name.startswith("panoptic"):
            predictor_settings = cfg.evaluation.predictor_settings["panoptic"]
        elif ds_name.startswith("tapvid2d-davis"):
            predictor_settings = cfg.evaluation.predictor_settings["tapvid2d-davis"]
        else:
            predictor_settings = cfg.evaluation.predictor_settings["generic"]
            logging.info(f"Using generic predictor settings for dataset with name {ds_name}")

        predictor = EvaluationPredictor3D(
            multiview_model=model,
            interp_shape=cfg.evaluation.interp_shape,
            single_point="single" in ds_name,
            n_iters=cfg.evaluation.eval_iters,
            **predictor_settings
        )

        log_dir_ds = os.path.join(log_dir, f"eval_{ds_name}")
        os.makedirs(log_dir_ds, exist_ok=True)

        if cfg.evaluation.consume_model_stats and hasattr(model, "init_stats"):
            model.init_stats()
        metrics = evaluator.evaluate_sequence(
            model=predictor,
            test_dataloader=dataloader,
            dataset_name=ds_name,
            writer=writer,
            step=step,
            log_dir=log_dir_ds,
        )
        metrics_by_dataset[ds_name] = metrics
        if cfg.evaluation.consume_model_stats and hasattr(model, "consume_stats"):
            model.consume_stats()

    if write_results:
        _write_eval_metrics(cfg, writer, step, metrics_by_dataset)

    # logging.info(f"Eval – GPU usage B: {gpustat.new_query()}")
    del predictor
    del metrics
    # logging.info(f"Eval – GPU usage C: {gpustat.new_query()}")
    torch.cuda.empty_cache()
    # logging.info(f"Eval – GPU usage D: {gpustat.new_query()}")

    model.train()
    return metrics_by_dataset


def augment_train_iters(train_iters: int, current_step: int, warmup_steps: int = 1000) -> int:
    """
    Adaptive iteration scheduler with warmup:
    - During warmup_steps: always return 1
    - After warmup:
        - 10% chance: return 1
        - 15% chance: return random int in [2, train_iters - 1]
        - 75% chance: return train_iters
    """
    if current_step < warmup_steps or train_iters <= 1:
        return 1

    rng = torch.Generator().manual_seed(current_step)
    p = torch.rand(1, generator=rng).item()

    if p < 0.10:
        return 1
    elif p < 0.25 and train_iters > 2:
        mid_candidates = list(range(2, train_iters))
        idx = torch.randint(len(mid_candidates), (1,), generator=rng).item()
        return mid_candidates[idx]
    else:
        return train_iters


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
@maybe_close_wandb
def main(cfg: DictConfig):
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    extras(cfg)
    Path(cfg.experiment_path).mkdir(exist_ok=True, parents=True)

    gradient_accumulation_steps = int(cfg.trainer.gradient_accumulation_steps)
    if gradient_accumulation_steps < 1:
        raise ValueError("trainer.gradient_accumulation_steps must be at least 1")
    logging.info(
        "Serial gradient accumulation: %d microbatches per optimizer step",
        gradient_accumulation_steps,
    )

    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", 1))
    devices = int(os.environ.get("SLURM_GPUS_PER_NODE", torch.cuda.device_count()))
    logging.info(f"SLURM job num nodes: {num_nodes}")
    logging.info(f"SLURM tasks per node (devices): {devices}")

    strategy = "auto"
    if num_nodes * devices > 1:
        from lightning.fabric.strategies import DDPStrategy
        strategy = DDPStrategy(find_unused_parameters=True)
    fabric = Fabric(
        num_nodes=num_nodes,
        devices=devices,
        precision=cfg.trainer.precision,
        strategy=strategy,
    )
    fabric.launch()
    fabric.seed_everything(cfg.reproducibility.seed, workers=True)
    if cfg.reproducibility.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.autograd.set_detect_anomaly(True)

    wandb_run_id = None
    if cfg.logging.get("log_wandb", False) and fabric.global_rank == 0:
        wandb_run_id = _get_or_create_wandb_run_id(
            cfg.experiment_path,
            cfg.logging.get("wandb_run_id"),
        )
        exp_name = cfg.logging.get("wandb_run_name") or cfg.experiment_path.replace(
            "./logs/", ""
        ).replace("/", "_").replace("\\", "_")
        wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.get("wandb_entity"),
            group=cfg.logging.get("wandb_group"),
            name=exp_name,
            tags=cfg.logging.get("tags", []),
            config=OmegaConf.to_container(cfg, resolve=True),
            sync_tensorboard=True,
            id=wandb_run_id,
            resume="allow",
        )
    if cfg.logging.get("log_wandb", False):
        wandb_run_id = fabric.broadcast(wandb_run_id, src=0)

    original_numpy = torch.Tensor.numpy

    def patched_numpy(self, *args, **kwargs):
        if self.dtype == torch.bfloat16:
            return original_numpy(self.float(), *args, **kwargs)
        return original_numpy(self, *args, **kwargs)

    torch.Tensor.numpy = patched_numpy

    eval_dataloaders = []
    for dataset_name in cfg.datasets.eval.names:
        if _mixed_source_name(cfg) and dataset_name == "tapvid3d-multiview-validation":
            source_cfg = cfg.datasets.eval.sources.diegesis
            include_scene_ids = list(source_cfg.include_scene_ids)
            if fabric.world_size > 1:
                include_scene_ids = include_scene_ids[
                    fabric.global_rank::fabric.world_size
                ]
            eval_dataset = TapVid3DMultiViewDataset.from_name(
                dataset_name,
                source_cfg.root,
                cfg,
                include_scene_ids=include_scene_ids,
            )
            expected_views = list(source_cfg.views)
            if any(
                list(manifest["views"]) != expected_views
                for manifest in eval_dataset._manifests.values()
            ):
                raise ValueError("DIEGESIS validation scenes do not match fixed views")
        elif _mixed_source_name(cfg) and dataset_name in {
            "kubric-multiview-v3-validation-subset",
            "kubric-multiview-v3-validation-full",
        }:
            source_cfg = cfg.datasets.eval.sources.mvkubric
            include_scene_ids = list(
                source_cfg.subset_scene_ids
                if dataset_name.endswith("-subset")
                else source_cfg.full_scene_ids
            )
            stream_rank = fabric.global_rank
            stream_world_size = fabric.world_size
            if dataset_name.endswith("-subset") and fabric.world_size > 1:
                include_scene_ids = include_scene_ids[
                    fabric.global_rank::fabric.world_size
                ]
                stream_rank = 0
                stream_world_size = 1
            kubric_kwargs = KubricMultiViewDataset.from_name(
                "kubric-multiview-v3",
                source_cfg.root,
                cfg,
                subset="train",
                include_scene_ids=include_scene_ids,
                just_return_kwargs=True,
            )
            kubric_kwargs.pop("include_scene_ids", None)
            eval_dataset = DaliKubricValidationDataset(
                **kubric_kwargs,
                webdataset_root=cfg.datasets.train.mvkubric_webdataset_root,
                include_scene_ids=include_scene_ids,
                views=tuple(source_cfg.views),
                stream_rank=stream_rank,
                stream_world_size=stream_world_size,
                stream_seed=int(cfg.reproducibility.seed),
            )
        elif dataset_name.startswith("tapvid2d-davis-"):
            eval_dataset = TapVidDataset.from_name(dataset_name, cfg.datasets.root)
        elif dataset_name.startswith("kubric-multiview-v3-25views"):
            kubric_kwargs = {
                "data_root": os.path.join(cfg.datasets.root, "kubric_multiview_003", "kubric_25_view"),
                "seq_len": 24,
                "traj_per_sample": 200,
                "seed": 72,
                "sample_vis_1st_frame": True,
                "tune_per_scene": False,
                "max_videos": 30,
                "use_duster_depths": False,
                "duster_views": None,
                "clean_duster_depths": False,
                "views_to_return": list(range(20)),
                "novel_views": list(range(20, 25)),
                "num_views": -1,
                "depth_noise_std": 0,
            }
            eval_dataset = KubricMultiViewDataset(**kubric_kwargs)
        elif dataset_name.startswith("kubric-multiview-v3"):
            eval_dataset = KubricMultiViewDataset.from_name(dataset_name, cfg.datasets.root, cfg)
        elif dataset_name.startswith("pointodyssey-multiview-"):
            eval_dataset = PointOdysseyMultiViewDataset.from_name(
                dataset_name, cfg.datasets.root, cfg
            )
        elif dataset_name.startswith("tapvid3d-multiview-"):
            eval_dataset = TapVid3DMultiViewDataset.from_name(
                dataset_name, cfg.datasets.root, cfg
            )
        elif dataset_name.startswith("panoptic-multiview"):
            eval_dataset = PanopticStudioMultiViewDataset.from_name(dataset_name, cfg.datasets.root)
        elif dataset_name.startswith("dex-ycb-multiview"):
            eval_dataset = DexYCBMultiViewDataset.from_name(dataset_name, cfg.datasets.root)
        elif dataset_name == "egoexo4d":
            eval_dataset = GenericSceneDataset(
                dataset_dir="datasets/egoexo4d-processed/maxframes-300_downsample-1_downscale-512/",
                drop_first_n_frames=44,
            )
        elif dataset_name == "4d-dress":
            eval_dataset = GenericSceneDataset(
                dataset_dir="datasets/4d-dress-processed-resized-512-selection",
                use_duster_depths=False,
            )
        elif dataset_name == "hi4d":
            eval_dataset = GenericSceneDataset(
                dataset_dir="datasets/hi4d-processed-resized-512",
                use_duster_depths=False,
                use_vggt_depths_with_aligned_cameras=True,
            )
        elif dataset_name == "selfcap-v1":
            eval_dataset = GenericSceneDataset(
                dataset_dir="datasets/selfcap-processed/numcams-8-seq-False_startframe-90_maxframes-256_downsample-10_downscale-512/",
                drop_first_n_frames=72,
            )
        elif dataset_name == "selfcap-v2":
            eval_dataset = GenericSceneDataset(
                dataset_dir="datasets/selfcap-processed/numcams-8-seq-True_startframe-90_maxframes-256_downsample-10_downscale-512/",
                drop_first_n_frames=72,
            )
        elif dataset_name == "selfcap-v3":
            eval_dataset = GenericSceneDataset(
                dataset_dir="datasets/selfcap-processed/numcams-8-seq-False_startframe-90_maxframes-256_downsample-20_downscale-512/",
                drop_first_n_frames=36,
            )
        elif dataset_name == "selfcap-v4":
            eval_dataset = GenericSceneDataset(
                dataset_dir="datasets/selfcap-processed/numcams-8-seq-False_startframe-90_maxframes-256_downsample-30_downscale-512/",
                drop_first_n_frames=24,
            )
        elif dataset_name == "selfcap-v5":
            eval_dataset = GenericSceneDataset(
                dataset_dir="datasets/selfcap-processed/numcams-8-seq-False_startframe-90_maxframes-256_downsample-5_downscale-512/",
                drop_first_n_frames=144,
            )
        elif dataset_name == "selfcap-v6":
            eval_dataset = GenericSceneDataset(
                dataset_dir="datasets/selfcap-processed/numcams-8-seq-False_startframe-90_maxframes-2560_downsample-10_downscale-512/",
                drop_first_n_frames=44,
            )
        elif dataset_name == "selfcap-v7":
            eval_dataset = GenericSceneDataset(
                dataset_dir="datasets/selfcap-processed/numcams-4-seq-False_startframe-90_maxframes-256_downsample-10_downscale-512/",
                drop_first_n_frames=72,
            )
        else:
            raise ValueError(f"Dataset {dataset_name} not supported for evaluation.")
        dali_validation = isinstance(eval_dataset, DaliKubricValidationDataset)
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=2 if dali_validation else 1,
            shuffle=False,
            num_workers=0 if dali_validation else cfg.datasets.eval.num_workers,
            collate_fn=getattr(eval_dataset, "collate_fn", collate_fn),
            pin_memory=True,
            persistent_workers=(
                not dali_validation and cfg.datasets.eval.num_workers > 0
            ),
            prefetch_factor=(
                2
                if not dali_validation and cfg.datasets.eval.num_workers > 0
                else None
            ),
            multiprocessing_context=(
                "spawn"
                if not dali_validation
                and getattr(eval_dataset, "requires_cuda_prefetch", False)
                and cfg.datasets.eval.num_workers > 0
                else None
            ),
        )
        if getattr(eval_dataset, "requires_cuda_prefetch", False):
            eval_dataloader = CudaPrefetchLoader(
                eval_dataloader,
                device=fabric.device,
                queue_depth=int(cfg.datasets.train.cuda_prefetch_queue_depth),
                decode_batch_size=(
                    1
                    if dali_validation
                    else int(cfg.datasets.train.cuda_decode_batch_size)
                ),
            )
        eval_dataloaders.append((dataset_name, eval_dataloader))

    # # Let each rank handle a subset of the evaluation dataloaders
    # eval_dataloaders_for_rank = []
    # for idx, (dset_name, dset_loader) in enumerate(eval_dataloaders):
    #     if (idx % fabric.world_size) == fabric.global_rank:
    #         eval_dataloaders_for_rank.append((dset_name, fabric.setup_dataloaders(dset_loader)))
    # eval_dataloaders = eval_dataloaders_for_rank

    train_viz_save_dir = os.path.join(cfg.experiment_path, f"train_{cfg.datasets.train.name}")
    os.makedirs(train_viz_save_dir, exist_ok=True)
    visualizer = MultiViewVisualizer(
        save_dir=train_viz_save_dir,
        pad_value=16,
        fps=12,
        show_first_frame=0,
        tracks_leave_trace=0,
    )

    evaluator = hydra.utils.instantiate(cfg.evaluation.evaluator)

    if cfg.modes.do_initial_static_pretrain and not cfg.modes.eval_only:
        pretraining_datasets = [
            kubric_multiview_dataset.KubricMultiViewDataset(
                data_root=os.path.join(cfg.datasets.root, "kubric_multiview_003", "train"),
                traj_per_sample=cfg.datasets.train.traj_per_sample,
                ratio_dynamic=0.1,
                ratio_very_dynamic=0.0,
                num_views=4,
                enable_cropping_augs=cfg.augmentations.cropping,

                seq_len=seq_len,
                static_cropping=static_cropping,
                max_videos=max_videos,
            )
            for seq_len, static_cropping, max_videos in [
                (12, True, 500),
                (18, True, 500),
                (24, True, 1000),
                (24, False, 2000),
            ]
        ]
        pretraining_dataset = torch.utils.data.ConcatDataset(pretraining_datasets)
        pretraining_dataloader = StatefulDataLoader(
            pretraining_dataset,
            batch_size=cfg.datasets.train.batch_size,
            shuffle=False,
            num_workers=cfg.datasets.train.num_workers,
            pin_memory=True,
            pin_memory_device="cuda",
            collate_fn=collate_fn,
            drop_last=True,
            in_order=cfg.reproducibility.deterministic,
        )
        pretraining_dataloader = fabric.setup_dataloaders(pretraining_dataloader)
    else:
        pretraining_dataloader = None

    mixed_training = not cfg.modes.eval_only and _mixed_source_name(cfg)
    mixed_schedule = None
    planned_physical_batching = False
    source_cursors = None
    source_samplers = None
    train_loaders = None
    train_datasets = None
    if cfg.modes.eval_only:
        train_dataset = None
    elif mixed_training:
        source_pattern = tuple(cfg.datasets.train.source_schedule)
        if source_pattern != ("diegesis", "mvkubric", "diegesis", "mvkubric"):
            raise ValueError("mixed training requires source_schedule D/K/D/K")
        if gradient_accumulation_steps != len(source_pattern):
            raise ValueError(
                "gradient accumulation must match the mixed source schedule length"
            )
        train_datasets = {
            source: _build_training_dataset(
                source_cfg.name,
                source_cfg.root,
                cfg,
                fabric,
                source_cfg,
            )
            for source, source_cfg in cfg.datasets.train.sources.items()
        }
        if _mvkubric_dali_stream(cfg):
            local_stream_build_seconds = float(
                train_datasets["mvkubric"].stream.build_seconds
            )
            logging.info(
                "MV-Kubric DALI stream built in %.3fs (rank=%d)",
                local_stream_build_seconds,
                fabric.global_rank,
            )
            stream_build_seconds_max = _reduce_scalar(
                fabric, local_stream_build_seconds, reduce_op="max"
            )
            stream_build_seconds_mean = _reduce_scalar(
                fabric, local_stream_build_seconds, reduce_op="mean"
            )
            if wandb_run_id is not None and fabric.global_rank == 0:
                wandb.log(
                    {
                        "startup/dali_stream_build_seconds_max": (
                            stream_build_seconds_max
                        ),
                        "startup/dali_stream_build_seconds_mean": (
                            stream_build_seconds_mean
                        ),
                    },
                    step=0,
                )
        mixed_schedule = BalancedMixedSourceSchedule(
            {source: dataset.real_len for source, dataset in train_datasets.items()},
            source_pattern,
            world_size=fabric.world_size,
            master_seed=int(cfg.reproducibility.seed),
        )
        source_cursors = {source: 0 for source in train_datasets}
        planned_physical_batching = _planned_physical_batching(cfg)
        if _mvkubric_dali_stream(cfg):
            if not planned_physical_batching:
                raise ValueError(
                    "mvkubric_storage=dali_stream requires physical batching"
                )
            if not bool(
                cfg.datasets.train.physical_batching.get("rank_local", False)
            ):
                raise ValueError(
                    "mvkubric_storage=dali_stream requires rank-local batching"
                )
        if planned_physical_batching:
            capacity = _physical_batch_capacity(cfg)
            rank_local = bool(
                cfg.datasets.train.physical_batching.get("rank_local", False)
            )
            if not rank_local and fabric.world_size != capacity.rank_count:
                raise ValueError(
                    "global physical batching requires exactly "
                    f"{capacity.rank_count} DDP ranks"
                )
        else:
            source_samplers = {
                source: _ScheduledSourceSampler(
                    mixed_schedule,
                    source,
                    fabric.global_rank,
                    len(dataset),
                )
                for source, dataset in train_datasets.items()
            }
            train_loaders = {
                source: _build_source_train_loader(
                    dataset, source_samplers[source], cfg, fabric
                )
                for source, dataset in train_datasets.items()
            }
        train_dataset = None
    else:
        train_dataset = _build_training_dataset(
            cfg.datasets.train.name, cfg.datasets.root, cfg, fabric
        )

    train_batch_sampler = None
    if mixed_training:
        train_loader = None
        optimizer_steps_per_epoch = max(1, int(cfg.trainer.num_steps))
        num_epochs = 1
    elif not cfg.modes.eval_only:
        requires_cuda_prefetch = getattr(train_dataset, "requires_cuda_prefetch", False)
        if requires_cuda_prefetch:
            torch.multiprocessing.set_sharing_strategy("file_system")
        loader_type = torch.utils.data.DataLoader if requires_cuda_prefetch else StatefulDataLoader
        loader_kwargs = {}
        if not requires_cuda_prefetch:
            loader_kwargs["in_order"] = cfg.reproducibility.deterministic
        elif cfg.datasets.train.num_workers > 0:
            loader_kwargs["multiprocessing_context"] = "spawn"
        common_loader_kwargs = {
            "num_workers": cfg.datasets.train.num_workers,
            "pin_memory": True,
            "collate_fn": getattr(train_dataset, "collate_fn", collate_fn),
            "prefetch_factor": (
                cfg.datasets.train.get("prefetch_factor", 2)
                if cfg.datasets.train.num_workers > 0 else None
            ),
            "persistent_workers": cfg.datasets.train.num_workers > 0,
            **loader_kwargs,
        }
        if (
            isinstance(train_dataset, TapVid3DMultiViewDataset)
            and cfg.augmentations.variable_num_views
        ):
            train_batch_sampler = HomogeneousViewBatchSampler(
                train_dataset,
                int(cfg.datasets.train.batch_size),
                world_size=fabric.world_size,
                rank=fabric.global_rank,
                seed=int(cfg.reproducibility.seed),
                view_count_probabilities=cfg.datasets.get(
                    "tapvid3d_view_count_probabilities", (0.25,) * 4
                ),
            )
            train_loader = loader_type(
                train_dataset,
                batch_sampler=train_batch_sampler,
                **common_loader_kwargs,
            )
        else:
            train_loader = loader_type(
                train_dataset,
                batch_size=cfg.datasets.train.batch_size,
                shuffle=True,
                drop_last=True,
                **common_loader_kwargs,
            )
        # eval_dataloaders += [("kubric-multiview-v3-training", train_loader)]
        train_loader = fabric.setup_dataloaders(
            train_loader,
            move_to_device=not requires_cuda_prefetch,
            use_distributed_sampler=train_batch_sampler is None,
        )
        if requires_cuda_prefetch:
            train_loader = CudaPrefetchLoader(
                train_loader,
                device=fabric.device,
                timing_interval=int(cfg.trainer.expensive_diagnostics_interval),
                queue_depth=int(cfg.datasets.train.cuda_prefetch_queue_depth),
                decode_batch_size=int(cfg.datasets.train.cuda_decode_batch_size),
            )
        logging.info(f"LEN TRAIN LOADER={len(train_loader)}")
        optimizer_steps_per_epoch = max(
            1,
            len(train_loader) // gradient_accumulation_steps,
        )
        num_epochs = (
            cfg.trainer.num_steps // optimizer_steps_per_epoch
            + 1
            + (1 if cfg.modes.do_initial_static_pretrain else 0)
        )
        if cfg.modes.do_initial_static_pretrain:
            cfg.trainer.num_steps += (
                len(pretraining_dataloader) // gradient_accumulation_steps
            )
    else:
        train_loader = None
        num_epochs = None

    epoch = -1
    total_steps = 0

    model: nn.Module = hydra.utils.instantiate(cfg.model)
    model.cuda()
    optimizer, scheduler = fetch_optimizer(cfg.trainer, model)
    model, optimizer = fabric.setup(model, optimizer)
    gradient_diagnostics = _MicrobatchGradientDiagnostics(
        model.parameters(),
        enabled=bool(cfg.trainer.get("gradient_diagnostics", True)),
    )
    expensive_diagnostics_interval = int(
        cfg.trainer.get("expensive_diagnostics_interval", 50)
    )
    if expensive_diagnostics_interval < 1:
        raise ValueError("trainer.expensive_diagnostics_interval must be at least 1")

    latest_checkpoint = _latest_checkpoint_path(cfg.experiment_path)
    if latest_checkpoint is not None:
        state = AttributeDict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            total_steps=total_steps,
            master_seed=int(cfg.reproducibility.seed),
            wandb_run_id=wandb_run_id or "",
        )
        if mixed_training:
            state.mixed_schedule_state = mixed_schedule.state_dict()
            state.source_cursors = dict(source_cursors)
        fabric.load(latest_checkpoint, state)
        total_steps = int(state.total_steps)
        if int(state.master_seed) != int(cfg.reproducibility.seed):
            raise ValueError(
                "resume master seed does not match reproducibility.seed"
            )
        if state.wandb_run_id and state.wandb_run_id != (wandb_run_id or ""):
            raise ValueError("resume W&B run ID does not match wandb_run_id.txt")
        if mixed_training:
            mixed_schedule.load_state_dict(state.mixed_schedule_state)
            source_cursors = {
                source: int(cursor)
                for source, cursor in state.source_cursors.items()
            }
            if source_samplers is not None:
                for source, sampler in source_samplers.items():
                    sampler.set_start_cursor(source_cursors[source])
        elif train_loader is not None:
            epoch = total_steps // optimizer_steps_per_epoch - 1
        logging.info(
            "Resumed canonical checkpoint %s at %d completed steps",
            latest_checkpoint,
            total_steps,
        )

    elif cfg.restore_ckpt_path is not None:
        restore_ckpt_path = cfg.restore_ckpt_path
        assert restore_ckpt_path.endswith(".pth")
        logging.info(f"Restoring pre-trained weights from {os.path.abspath(restore_ckpt_path)}")
        training_ckpt = "total_steps" in torch.load(restore_ckpt_path)
        if training_ckpt:
            state = AttributeDict(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                total_steps=total_steps,
            )
            if mixed_training:
                state.mixed_schedule_state = mixed_schedule.state_dict()
                state.source_cursors = dict(source_cursors)
            fabric.load(restore_ckpt_path, state, strict=True)
            total_steps = int(state.total_steps)
            if mixed_training:
                mixed_schedule.load_state_dict(state.mixed_schedule_state)
                source_cursors = {
                    source: int(cursor)
                    for source, cursor in state.source_cursors.items()
                }
                if source_samplers is not None:
                    for source, sampler in source_samplers.items():
                        sampler.set_start_cursor(source_cursors[source])
            logging.info(
                "Resumed explicit training checkpoint %s at %d completed steps",
                restore_ckpt_path,
                total_steps,
            )
        else:
            fabric.load_raw(restore_ckpt_path, model)

    # Compile the indexed-correlation forward, Triton target backward and
    # prebuilt CUDA source backward before training timings begin. Every DDP
    # rank warms its own CUDA context; the barrier keeps the first measured
    # step from waiting on a slower rank's compilation.
    from mvtracker.models.core.mvtracker.indexed_correlation import (
        warmup_indexed_correlation,
    )
    indexed_correlation_warmup_seconds = warmup_indexed_correlation(fabric.device)
    logging.info(
        "Indexed-correlation startup warmup completed in %.3fs (rank=%d)",
        indexed_correlation_warmup_seconds,
        fabric.global_rank,
    )
    fabric.barrier()
    if wandb_run_id is not None and fabric.global_rank == 0:
        wandb.log(
            {"startup/indexed_correlation_warmup_seconds": indexed_correlation_warmup_seconds},
            step=total_steps,
        )

    tb_writer = (
        SummaryWriter(log_dir=os.path.join(cfg.experiment_path, "runs"), flush_secs=10)
        if fabric.global_rank == 0
        else None
    )
    last_eval_step = None
    if cfg.modes.eval_only or cfg.modes.validate_at_start:
        _run_eval(
            fabric,
            cfg,
            evaluator,
            model,
            eval_dataloaders,
            tb_writer,
            total_steps,
        )
        last_eval_step = total_steps
        if cfg.modes.eval_only:
            gradient_diagnostics.close()
            if tb_writer is not None:
                tb_writer.close()
            return

    physical_lookahead = None
    physical_decoder = None
    if planned_physical_batching:
        settings = cfg.datasets.train.physical_batching
        physical_lookahead = MixedStepLookahead(
            datasets=train_datasets,
            schedule=mixed_schedule,
            source_cursors=source_cursors,
            rank=fabric.global_rank,
            remaining_steps=int(cfg.trainer.num_steps) - total_steps,
            worker_count=int(cfg.datasets.train.num_workers),
            lookahead_steps=int(settings.lookahead_steps),
            max_cache_bytes=int(float(settings.cpu_cache_gib_per_rank) * 1024**3),
            capacity=_physical_batch_capacity(cfg),
            rank_local=bool(settings.get("rank_local", False)),
        )
        physical_decoder = PhysicalBatchDecoder(
            fabric.device,
            decode_image_chunk_size=int(settings.decode_image_chunk_size),
            dali_num_threads=int(cfg.datasets.train.dali.num_threads),
            dali_prefetch_queue_depth=int(
                cfg.datasets.train.dali.prefetch_queue_depth
            ),
        )

    dali_stream_batches_seen = set()

    profiler_start_step = int(
        cfg.trainer.get("profiler", {}).get("start_step", 0)
    )
    torch_profiler = None
    if total_steps >= profiler_start_step:
        torch_profiler = _create_torch_profiler(
            cfg,
            cfg.experiment_path,
            fabric.global_rank,
        )
    if torch_profiler is not None:
        logging.info("Starting torch.profiler for successful training microbatches")
        torch_profiler.start()
    memory_recorder = _CudaMemoryRecorder(
        cfg, cfg.experiment_path, fabric.global_rank, model, optimizer
    )

    total_durations = deque()
    dataloader_durations = deque()
    fwd_durations = deque()
    sync_durations = deque()
    bwd_durations = deque()
    timing_log_freq = 100
    checkpoint_source_cursors = (
        dict(source_cursors) if mixed_training else None
    )

    def handle_sigterm(signum, frame):
        logging.error(f"Signal {signum} received, saving checkpoint and exiting...")
        save_path = Path(cfg.experiment_path) / f"model_{total_steps:06d}.pth"
        _save_training_checkpoint(
            fabric,
            cfg.experiment_path,
            save_path,
            model,
            optimizer,
            scheduler,
            total_steps,
            cfg.reproducibility.seed,
            wandb_run_id,
            mixed_schedule.state_dict() if mixed_training else None,
            checkpoint_source_cursors,
        )
        logging.info(f"Saved checkpoint to {save_path}")
        logging.info(f"Calling sys.exit(0) now.")
        sys.exit(0)

    signal.signal(signal.SIGUSR1, handle_sigterm)
    signal.signal(signal.SIGTERM, handle_sigterm)
    logging.info(f"Registered signal handlers for SIGUSR1 and SIGTERM.")

    model.train()
    hardware_metrics_interval = int(cfg.trainer.get("hardware_metrics_interval", 10))
    if hardware_metrics_interval < 1:
        raise ValueError("trainer.hardware_metrics_interval must be at least 1")
    gpu_monitor = _RankGpuMonitor(torch.cuda.current_device())
    container_monitor = (
        _ContainerHardwareMonitor() if fabric.global_rank == 0 else None
    )
    should_keep_training = total_steps < cfg.trainer.num_steps
    total_batches_loaded = 0
    total_batches_failed = 0
    clipped_optimizer_steps = 0
    diagnostic_optimizer_steps = 0
    if fabric.global_rank == 0:
        tqdm_total_steps = tqdm(
            total=cfg.trainer.num_steps,
            desc=f"Total Training Progress (rank={fabric.global_rank})",
            unit="optimizer step",
            initial=total_steps,
            position=0,
        )
    threads = []
    pretraining_optimizer_steps = (
        len(pretraining_dataloader) // gradient_accumulation_steps
        if pretraining_dataloader is not None
        else 0
    )
    had_run_pretraining_epoch = (
        cfg.modes.do_initial_static_pretrain
        and total_steps >= pretraining_optimizer_steps
    )
    logging.info(f"{total_steps=}, {epoch=}/{num_epochs}, {had_run_pretraining_epoch=}")
    while should_keep_training:
        epoch += 1
        i_batch = 0
        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(epoch)

        if mixed_training:
            if not planned_physical_batching:
                for source, sampler in source_samplers.items():
                    sampler.set_start_cursor(source_cursors[source])
                data_iters = _start_mixed_source_iterators(train_loaders)
            n_batches = (
                int(cfg.trainer.num_steps) - total_steps
            ) * gradient_accumulation_steps
        elif cfg.modes.do_initial_static_pretrain and not had_run_pretraining_epoch:
            had_run_pretraining_epoch = True
            data_iter = iter(pretraining_dataloader)
            n_batches = len(pretraining_dataloader)
        else:
            data_iter = iter(train_loader)
            n_batches = len(train_loader)
        n_batches -= n_batches % gradient_accumulation_steps
        if n_batches == 0:
            raise ValueError(
                "The training loader must contain at least "
                f"{gradient_accumulation_steps} microbatches"
            )
        if fabric.global_rank == 0:
            tqdm_epoch = tqdm(
                total=n_batches // gradient_accumulation_steps,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                unit="optimizer step",
                position=1,
            )

        microbatches_accumulated = 0
        accumulation_started_at = None
        accumulated_dataloader_duration = 0.0
        accumulated_fwd_duration = 0.0
        accumulated_sync_duration = 0.0
        accumulated_bwd_duration = 0.0
        accumulated_loader_worker_seconds = 0.0
        accumulated_gpu_jpeg_decode_ms = 0.0
        accumulated_gpu_prepare_ms = 0.0
        accumulated_sampling_metrics = {}
        accumulated_dali_stream_metrics = {
            "batch_count": 0.0,
            "payload_bytes": 0.0,
            "batch_wait_seconds": 0.0,
        }
        accumulated_loss_value = None
        accumulated_component_losses = {}
        accumulated_metrics = {}
        accumulated_source_losses = {}
        accumulated_source_components = {}
        accumulated_source_metrics = {}
        accumulated_source_counts = {}
        accumulated_source_view_counts = {}
        accumulated_source_track_counts = {}
        accumulated_sample_count = 0
        accumulated_trajectory_count = 0.0
        physical_batching_metrics = None
        microbatch_gradient_norms = []
        microbatch_gradient_cosines = []
        mixed_step_batches = None
        physical_step = None
        physical_group_iterator = None
        physical_group_count = None

        while i_batch < n_batches and total_steps < cfg.trainer.num_steps:
            if accumulation_started_at is None:
                accumulation_started_at = time.time()
                optimizer.zero_grad()
            start_time_1 = time.time()
            logging.info(f"Gonna load batch {i_batch + 1}/{n_batches} (rank={fabric.global_rank})")
            current_sources = None
            if mixed_training:
                if planned_physical_batching:
                    if microbatches_accumulated == 0:
                        physical_step = next(physical_lookahead)
                        if physical_step.start_cursors != source_cursors:
                            raise RuntimeError(
                                "physical lookahead cursor does not match committed state"
                            )
                        if not bool(settings.get("rank_local", False)):
                            _assert_matching_step_fingerprint(
                                fabric, physical_step.fingerprint
                            )
                        materialization_succeeded = _all_ranks_succeeded(
                            fabric,
                            physical_step.materialization_error is None,
                        )
                        if not materialization_succeeded:
                            detail = physical_step.materialization_error or (
                                "materialization failed on another rank"
                            )
                            raise RuntimeError(
                                f"physical step materialization failed: {detail}"
                            )
                        physical_group_count = len(physical_step.groups)
                        physical_group_iterator = iter(
                            PhysicalGroupPrefetchIterator(
                                physical_step.groups, physical_decoder
                            )
                        )
                        total_batches_loaded += physical_step.logical_scene_count
                        total_batches_failed += physical_step.retry_count
                        physical_batching_metrics = {
                            "planning_seconds": physical_step.planning_seconds,
                            "materialization_seconds": physical_step.materialization_seconds,
                            "encoded_cache_gib": physical_step.encoded_bytes / 1024**3,
                            "pair_count": float(physical_step.pair_count),
                            "padding_tracks": float(physical_step.padding_tracks),
                            "physical_group_count": float(physical_group_count),
                        }
                    physical_group, batch = next(physical_group_iterator)
                    current_sources = physical_group.sources
                    gotit = [True] * len(current_sources)
                elif (
                    cfg.datasets.train.materialize_whole_step
                    and microbatches_accumulated == 0
                ):
                    (
                        mixed_step_batches,
                        step_batches_loaded,
                        step_batches_failed,
                    ) = _load_mixed_step(
                        fabric,
                        mixed_schedule.source_pattern,
                        data_iters,
                        source_samplers,
                        train_loaders,
                        source_cursors,
                    )
                    total_batches_loaded += step_batches_loaded
                    total_batches_failed += step_batches_failed
                    if step_batches_failed:
                        logging.info(
                            "whole mixed step discarded %d/%d paired candidates",
                            step_batches_failed,
                            step_batches_loaded,
                        )
                if (
                    not planned_physical_batching
                    and cfg.datasets.train.materialize_whole_step
                ):
                    current_source, batch = mixed_step_batches[
                        microbatches_accumulated
                    ]
                    current_sources = (current_source,)
                    mixed_step_batches[microbatches_accumulated] = None
                elif not planned_physical_batching:
                    current_source = mixed_schedule.source_pattern[
                        microbatches_accumulated
                    ]
                    current_sources = (current_source,)
                    try:
                        batch = next(data_iters[current_source])
                    except StopIteration:
                        source_samplers[current_source].set_start_cursor(
                            source_cursors[current_source]
                        )
                        data_iters[current_source] = iter(
                            train_loaders[current_source]
                        )
                        batch = next(data_iters[current_source])
                    source_cursors[current_source] += 1
                    batch, gotit = batch
                    total_batches_loaded += 1
            else:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    n_batches = len(train_loader)
                    n_batches -= n_batches % gradient_accumulation_steps
                    batch = next(data_iter)
                batch, gotit = batch
                total_batches_loaded += 1

            if cfg.modes.debugging_hotfix_datapoint_path is not None:
                logging.info(f"Debugging hotfix: loading batch from {cfg.modes.debugging_hotfix_datapoint_path}")
                batch = torch.load(cfg.modes.debugging_hotfix_datapoint_path, map_location="cuda:0")
                logging.info(f"Debugging hotfix: loaded batch {batch.seq_name} "
                             f"with {len(batch.video)} views and {batch.video.shape[2]} frames")

            if (
                (
                    not mixed_training
                    or (
                        not planned_physical_batching
                        and not cfg.datasets.train.materialize_whole_step
                    )
                )
                and not _all_ranks_succeeded(fabric, all(gotit))
            ):
                total_batches_failed += 1
                accumulated_dataloader_duration += time.time() - start_time_1
                logging.info(f"batch is None: "
                             f"failed {total_batches_failed} / {total_batches_loaded} "
                             f"({total_batches_failed / total_batches_loaded * 100:.2f}%) batches")
                continue

            batch_scene_count = int(batch.video.shape[0])
            i_batch += batch_scene_count if planned_physical_batching else 1
            if torch_profiler is None and total_steps >= profiler_start_step:
                torch_profiler = _create_torch_profiler(
                    cfg,
                    cfg.experiment_path,
                    fabric.global_rank,
                )
                if torch_profiler is not None:
                    logging.info(
                        "Starting torch.profiler at optimizer step %d",
                        total_steps,
                    )
                    torch_profiler.start()
            host_to_device_started_at = time.time()
            memory_recorder.begin_group(
                total_steps,
                microbatches_accumulated,
                batch,
                host_to_device_started_at - start_time_1,
            )
            with torch.profiler.record_function("train/host_to_device"):
                dataclass_to_cuda_(batch)
            memory_recorder.set_host_to_device_seconds(
                time.time() - host_to_device_started_at
            )
            memory_recorder.capture("after_batch_to_cuda", batch=batch)
            assert model.training

            start_time_2 = time.time()
            microbatch_dataloader_duration = start_time_2 - start_time_1
            accumulated_dataloader_duration += microbatch_dataloader_duration
            logging.info(
                f"Datapoint: {batch.seq_name} "
                f"(microbatch {microbatches_accumulated + 1}/"
                f"{physical_group_count or gradient_accumulation_steps}, "
                f"waited {microbatch_dataloader_duration:>5.2f}s)"
            )
            if batch.sample_metadata:
                metadata = batch.sample_metadata
                accumulated_loader_worker_seconds += float(np.mean([
                    item.get("worker_prepare_seconds", 0.0) for item in metadata
                ]))
                accumulated_gpu_jpeg_decode_ms += float(np.mean([
                    item.get("gpu_jpeg_decode_ms", 0.0) for item in metadata
                ]))
                accumulated_gpu_prepare_ms += float(np.mean([
                    item.get("gpu_prepare_total_ms", 0.0) for item in metadata
                ]))
                for name in metadata[0]:
                    if name.startswith("motion_"):
                        accumulated_sampling_metrics[name] = (
                            accumulated_sampling_metrics.get(name, 0.0)
                            + float(np.mean([item[name] for item in metadata]))
                        )
                for item in metadata:
                    if item.get("record_store") != "dali-webdataset":
                        continue
                    batch_index = int(item["dali_batch_index"])
                    if batch_index in dali_stream_batches_seen:
                        continue
                    dali_stream_batches_seen.add(batch_index)
                    accumulated_dali_stream_metrics["batch_count"] += 1.0
                    accumulated_dali_stream_metrics["payload_bytes"] += float(
                        item["dali_payload_bytes"]
                    )
                    accumulated_dali_stream_metrics[
                        "batch_wait_seconds"
                    ] += float(item["dali_read_seconds"])
            source_view_count = int(batch.video.shape[1])
            if batch.track_padding_mask is not None:
                scene_track_counts = (
                    (~batch.track_padding_mask.bool()).sum(dim=1).tolist()
                )
            else:
                scene_track_counts = [
                    int(batch.trajectory.shape[-2])
                ] * batch_scene_count
            accumulated_sample_count += batch_scene_count
            accumulated_trajectory_count += sum(scene_track_counts)

            train_iters = cfg.trainer.train_iters
            if cfg.trainer.augment_train_iters:
                train_iters = augment_train_iters(train_iters, total_steps, cfg.trainer.augment_train_iters_warmup)

            is_final_microbatch = (
                microbatches_accumulated + 1
                == (physical_group_count or gradient_accumulation_steps)
            )
            # Rank-local physical plans may contain different numbers of
            # groups.  Each rank still enables DDP synchronization exactly
            # once, on its final group; the collective is matched by order,
            # while no_backward_sync suppresses all earlier groups.
            run_expensive_diagnostics = (
                bool(cfg.trainer.get("expensive_diagnostics_enabled", True))
                and total_steps % expensive_diagnostics_interval == 0
            )

            try:
                (
                    output,
                    microbatch_loss,
                    component_losses,
                    microbatch_metrics,
                    scene_losses,
                    microbatch_gradient,
                    forward_duration,
                    backward_duration,
                ) = _forward_backward_microbatch(
                    fabric=fabric,
                    model=model,
                    batch=batch,
                    cfg=cfg,
                    completed_steps=total_steps,
                    train_iters=train_iters,
                    gradient_accumulation_steps=(
                        gradient_accumulation_steps / batch_scene_count
                    ),
                    is_final_microbatch=is_final_microbatch,
                    run_expensive_diagnostics=run_expensive_diagnostics,
                    gradient_diagnostics=gradient_diagnostics,
                    memory_recorder=memory_recorder,
                )
                if not is_final_microbatch:
                    memory_recorder.end_group(
                        forward_duration, backward_duration
                    )
            except Exception as e:
                logging.critical(f"Forward pass crashed at step {total_steps}: {e}")

                # Save current checkpoint
                save_path = Path(f"{cfg.experiment_path}/test_{total_steps:06d}.pth")
                state = AttributeDict(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    total_steps=total_steps,
                )
                if mixed_training:
                    state.mixed_schedule_state = mixed_schedule.state_dict()
                    state.source_cursors = dict(checkpoint_source_cursors)
                fabric._strategy.checkpoint_io.save_checkpoint(
                    checkpoint=fabric._strategy._convert_stateful_objects_in_state(_unwrap_objects(state), filter={}),
                    path=save_path,
                )
                logging.info(f"Saved crash checkpoint to {save_path}")

                # Save the batch
                batch_path = Path(f"{cfg.experiment_path}/crash_batch_step_{total_steps:06d}.pt")
                try:
                    torch.save(batch, batch_path)
                    logging.info(f"Saved crashing batch to {batch_path}")
                except Exception as batch_exc:
                    logging.error(f"Failed to save crashing batch as .pt: {batch_exc}")

                raise  # re-raise to crash the job after saving artifacts

            for metric_name, metric_value in microbatch_metrics.items():
                accumulated_metrics[metric_name] = (
                    accumulated_metrics.get(metric_name, 0.0)
                    + metric_value * batch_scene_count
                )
            for component_name, component_loss in component_losses.items():
                accumulated_component_losses[component_name] = (
                    accumulated_component_losses.get(component_name, 0.0)
                    + component_loss * batch_scene_count
                )
            accumulated_loss_value = (
                microbatch_loss * batch_scene_count
                if accumulated_loss_value is None
                else accumulated_loss_value + microbatch_loss * batch_scene_count
            )
            if current_sources is not None:
                scene_total_losses = (
                    scene_losses["flow"] + scene_losses["visibility"]
                )
                for scene_index, current_source in enumerate(current_sources):
                    accumulated_source_counts[current_source] = (
                        accumulated_source_counts.get(current_source, 0) + 1
                    )
                    accumulated_source_losses[current_source] = (
                        accumulated_source_losses.get(current_source, 0.0)
                        + scene_total_losses[scene_index]
                    )
                    accumulated_source_view_counts[current_source] = (
                        accumulated_source_view_counts.get(current_source, 0.0)
                        + source_view_count
                    )
                    accumulated_source_track_counts[current_source] = (
                        accumulated_source_track_counts.get(current_source, 0.0)
                        + scene_track_counts[scene_index]
                    )
                    for component_name, per_scene_losses in scene_losses.items():
                        key = (current_source, component_name)
                        accumulated_source_components[key] = (
                            accumulated_source_components.get(key, 0.0)
                            + per_scene_losses[scene_index]
                        )
                if len(set(current_sources)) == 1:
                    current_source = current_sources[0]
                    for metric_name, metric_value in microbatch_metrics.items():
                        key = (current_source, metric_name)
                        accumulated_source_metrics[key] = (
                            accumulated_source_metrics.get(key, 0.0)
                            + metric_value * batch_scene_count
                        )
            accumulated_fwd_duration += forward_duration
            accumulated_bwd_duration += backward_duration
            if microbatch_gradient is not None:
                microbatch_gradient_norms.append(microbatch_gradient["norm"])
                cosine = microbatch_gradient["cosine_to_running_accumulator"]
                if cosine is not None:
                    microbatch_gradient_cosines.append(cosine)
            microbatches_accumulated += 1
            if not planned_physical_batching:
                if microbatches_accumulated < gradient_accumulation_steps:
                    if torch_profiler is not None:
                        torch_profiler.step()
                    continue
            if (
                planned_physical_batching
                and microbatches_accumulated < physical_group_count
            ):
                if torch_profiler is not None:
                    torch_profiler.step()
                continue

            mean_loss_value = _reduce_scalar(
                fabric,
                accumulated_loss_value / gradient_accumulation_steps,
            )
            reduced_metrics = _reduce_scalar_dict(
                fabric,
                {
                    name: total / gradient_accumulation_steps
                    for name, total in accumulated_metrics.items()
                },
            )
            reduced_component_losses = _reduce_scalar_dict(
                fabric,
                {
                    name: total / gradient_accumulation_steps
                    for name, total in accumulated_component_losses.items()
                },
            )
            reduced_source_values = {}
            if mixed_training:
                for source in mixed_schedule.scene_counts:
                    local_count = accumulated_source_counts.get(source, 0)
                    sums = _reduce_scalar_dict(
                        fabric,
                        {
                            "loss": accumulated_source_losses.get(source, 0.0),
                            "view_count": accumulated_source_view_counts.get(source, 0.0),
                            "track_count": accumulated_source_track_counts.get(source, 0.0),
                            "component/flow": accumulated_source_components.get(
                                (source, "flow"), 0.0
                            ),
                            "component/visibility": accumulated_source_components.get(
                                (source, "visibility"), 0.0
                            ),
                        },
                        reduce_op="sum",
                    )
                    global_count = _reduce_scalar(
                        fabric, local_count, reduce_op="sum"
                    )
                    reduced_source_values[f"source/{source}/sample_count"] = global_count
                    expected_count = (
                        mixed_schedule.source_pattern.count(source)
                        * fabric.world_size
                    )
                    if global_count != expected_count:
                        raise RuntimeError(
                            f"source {source} contributed an unexpected global sample count"
                        )
                    for name in ("loss", "view_count", "track_count"):
                        reduced_source_values[f"source/{source}/{name}"] = (
                            sums[name] / global_count
                        )
                    for component in ("flow", "visibility"):
                        reduced_source_values[
                            f"source/{source}/component/{component}"
                        ] = sums[f"component/{component}"] / global_count
            if tb_writer is not None:
                for metric_name, metric_value in reduced_metrics.items():
                    tb_writer.add_scalar(metric_name, metric_value, total_steps + 1)
                for component_name, component_value in reduced_component_losses.items():
                    tb_writer.add_scalar(
                        f"live_{component_name}_loss",
                        component_value,
                        total_steps + 1,
                    )
                for metric_name, metric_value in reduced_source_values.items():
                    tb_writer.add_scalar(metric_name, metric_value, total_steps + 1)

            # Log a limited number of grad + optimizer state pairs, also log current learning rate
            if (total_steps <= 10) or (total_steps % cfg.trainer.viz_freq == 0):
                log_limit = 5
                logged = 0
                prefix = f"[DEBUG] [RANK={fabric.global_rank:03d}]"
                logging.info(f"{prefix} RNG seed: {torch.initial_seed()}")
                logging.info(f"{prefix} Step={total_steps} – Gradients and Optimizer State")
                for name, param in model.named_parameters():
                    if param.grad is not None and param in optimizer.state:
                        state = optimizer.state[param]
                        exp_avg_norm = state['exp_avg'].norm().item() if 'exp_avg' in state else float('nan')
                        exp_avg_sq_norm = state['exp_avg_sq'].norm().item() if 'exp_avg_sq' in state else float('nan')
                        grad_norm = param.grad.norm().item()
                        logging.info(
                            f"{prefix} Param: {name:<60s} | "
                            f"grad_norm={grad_norm:>14.9f} | "
                            f"exp_avg_norm={exp_avg_norm:>14.9f} | "
                            f"exp_avg_sq_norm={exp_avg_sq_norm:>14.9f}"
                        )
                        logged += 1
                        if logged >= log_limit:
                            break
                for name, param in model.named_parameters():
                    if param.grad_fn:
                        print(f"{prefix} {name} grad_fn: {param.grad_fn}")
                logging.info(f"{prefix} LR at step {total_steps}: {scheduler.get_last_lr()}")
            optimizer_update_started_at = time.time()
            with torch.profiler.record_function("train/gradient_clip_and_optimizer"):
                model_parameters = list(model.parameters())
                if run_expensive_diagnostics:
                    pre_clip_gradient_norm = float(
                        _global_gradient_l2_norm(model_parameters).item()
                    )
                    max_abs_gradient_pre_clip, clipped_element_fraction = (
                        _gradient_value_clip_stats(
                            model_parameters,
                            float(cfg.trainer.grad_clip),
                        )
                    )
                fabric.clip_gradients(model, optimizer, clip_val=cfg.trainer.grad_clip)
                if run_expensive_diagnostics:
                    post_clip_gradient_norm = float(
                        _global_gradient_l2_norm(model_parameters).item()
                    )
                    gradient_norm_retention = (
                        post_clip_gradient_norm / pre_clip_gradient_norm
                        if pre_clip_gradient_norm > 0.0
                        else 1.0
                    )
                    was_clipped = clipped_element_fraction > 0.0
                    clipped_optimizer_steps += int(was_clipped)
                    diagnostic_optimizer_steps += 1
                    clipped_step_fraction = (
                        clipped_optimizer_steps / float(diagnostic_optimizer_steps)
                    )
                optimizer.step()
                scheduler.step()
            optimizer_duration = time.time() - optimizer_update_started_at
            accumulated_bwd_duration += optimizer_duration
            memory_recorder.capture("after_optimizer", batch=batch)
            memory_recorder.end_group(
                forward_duration,
                backward_duration,
                optimizer_seconds=optimizer_duration,
            )
            training_step_duration = time.time() - accumulation_started_at
            total_steps += 1
            if total_steps > memory_recorder.start_step:
                memory_recorder.finish()
            if planned_physical_batching:
                source_cursors = dict(physical_step.end_cursors)
            hardware_metrics = {}
            if total_steps == 1 or total_steps % hardware_metrics_interval == 0:
                hardware_metrics = _gather_rank_metrics(
                    fabric, gpu_monitor.sample()
                )
                if container_monitor is not None:
                    hardware_metrics.update(container_monitor.sample())
            if mixed_training:
                checkpoint_source_cursors = dict(source_cursors)

            microbatch_gradient_norm_mean = (
                statistics.fmean(microbatch_gradient_norms)
                if microbatch_gradient_norms
                else None
            )
            microbatch_gradient_norm_min = (
                min(microbatch_gradient_norms) if microbatch_gradient_norms else None
            )
            microbatch_gradient_norm_max = (
                max(microbatch_gradient_norms) if microbatch_gradient_norms else None
            )
            microbatch_gradient_cosine_mean = (
                statistics.fmean(microbatch_gradient_cosines)
                if microbatch_gradient_cosines
                else None
            )
            microbatch_gradient_cosine_min = (
                min(microbatch_gradient_cosines)
                if microbatch_gradient_cosines
                else None
            )
            if run_expensive_diagnostics:
                reduced_optimization = _reduce_scalar_dict(
                    fabric,
                    {
                        "pre_clip_gradient_norm": pre_clip_gradient_norm,
                        "post_clip_gradient_norm": post_clip_gradient_norm,
                        "max_abs_gradient_pre_clip": max_abs_gradient_pre_clip,
                        "gradient_norm_retention": gradient_norm_retention,
                        "clipped_element_fraction": clipped_element_fraction,
                        "clipped_step_fraction": clipped_step_fraction,
                    },
                )
                pre_clip_gradient_norm = reduced_optimization["pre_clip_gradient_norm"]
                post_clip_gradient_norm = reduced_optimization["post_clip_gradient_norm"]
                max_abs_gradient_pre_clip = reduced_optimization["max_abs_gradient_pre_clip"]
                gradient_norm_retention = reduced_optimization["gradient_norm_retention"]
                clipped_element_fraction = reduced_optimization["clipped_element_fraction"]
                clipped_step_fraction = reduced_optimization["clipped_step_fraction"]
                if microbatch_gradient_norm_mean is not None:
                    microbatch_gradient_norm_mean = _reduce_scalar(
                        fabric, microbatch_gradient_norm_mean
                    )
                    microbatch_gradient_norm_min = _reduce_scalar(
                        fabric, microbatch_gradient_norm_min, reduce_op="min"
                    )
                    microbatch_gradient_norm_max = _reduce_scalar(
                        fabric, microbatch_gradient_norm_max, reduce_op="max"
                    )
                if microbatch_gradient_cosine_mean is not None:
                    microbatch_gradient_cosine_mean = _reduce_scalar(
                        fabric, microbatch_gradient_cosine_mean
                    )
                    microbatch_gradient_cosine_min = _reduce_scalar(
                        fabric, microbatch_gradient_cosine_min, reduce_op="min"
                    )
                logging.info(
                    "[optimizer:%06d] loss=%.8f grad_pre=%.8f grad_post=%.8f "
                    "max_abs_pre=%.8f value_clip=%.8f norm_retention=%.8f "
                    "elements_clipped=%.8f clipped=%d clipped_step_fraction=%.6f "
                    "micro_grad_mean=%s micro_grad_cos_mean=%s micro_grad_cos_min=%s",
                    total_steps,
                    mean_loss_value,
                    pre_clip_gradient_norm,
                    post_clip_gradient_norm,
                    max_abs_gradient_pre_clip,
                    float(cfg.trainer.grad_clip),
                    gradient_norm_retention,
                    clipped_element_fraction,
                    int(was_clipped),
                    clipped_step_fraction,
                    (
                        f"{microbatch_gradient_norm_mean:.8f}"
                        if microbatch_gradient_norm_mean is not None
                        else "n/a"
                    ),
                    (
                        f"{microbatch_gradient_cosine_mean:.8f}"
                        if microbatch_gradient_cosine_mean is not None
                        else "n/a"
                    ),
                    (
                        f"{microbatch_gradient_cosine_min:.8f}"
                        if microbatch_gradient_cosine_min is not None
                        else "n/a"
                    ),
                )

            dataloader_duration = accumulated_dataloader_duration
            fwd_duration = accumulated_fwd_duration
            sync_duration = accumulated_sync_duration
            bwd_duration = accumulated_bwd_duration

            if fabric.global_rank == 0:
                if not mixed_training and (
                    total_steps % cfg.trainer.viz_freq == 0
                    or total_steps == cfg.trainer.num_steps
                    or total_steps in [1, 10, 100]
                ):
                    logging.info(f"Creating training viz logs (rank: {fabric.global_rank}, step: {total_steps})")
                    # Training visualization intentionally shows scene zero;
                    # the loss path above handles every scene in the batch.
                    video = batch.video[:1].clone().cpu()
                    video_depth = batch.videodepth[:1].clone().cpu()
                    gt_viz, vector_colors = visualizer.visualize(
                        video=video,
                        video_depth=video_depth,
                        tracks=batch.trajectory[:1].clone().cpu(),
                        visibility=batch.visibility[:1].clone().cpu(),
                        query_frame=batch.query_points_3d[:1, ..., 0].long().clone().cpu(),
                        filename="train_gt_traj",
                        writer=tb_writer,
                        step=total_steps,
                        save_video=False,
                    )
                    pred_viz, _ = visualizer.visualize(
                        video=video,
                        video_depth=video_depth,
                        tracks=output["flow"]["predictions"][:1].cpu(),
                        visibility=(output["visibility"]["predictions"][:1] > 0.5).cpu(),
                        query_frame=batch.query_points_3d[:1, ..., 0].long().clone().cpu(),
                        filename="train_pred_traj",
                        writer=tb_writer,
                        step=total_steps,
                        save_video=False,
                    )
                    viz = torch.cat([gt_viz[..., :gt_viz.shape[-1] // 2], pred_viz], dim=-1)
                    thread = threading.Thread(
                        target=Visualizer.save_video,
                        args=(viz, visualizer.save_dir, f"train", tb_writer, visualizer.fps, total_steps)
                    )
                    thread.start()
                    threads.append(thread)

                if len(output) > 1:
                    tb_writer.add_scalar(f"live_total_loss", mean_loss_value, total_steps)
                tb_writer.add_scalar(f"learning_rate", optimizer.param_groups[0]["lr"], total_steps)
                if run_expensive_diagnostics:
                    tb_writer.add_scalar(
                        "optimization/grad_norm_pre_clip",
                        pre_clip_gradient_norm,
                        total_steps,
                    )
                    tb_writer.add_scalar(
                        "optimization/grad_norm_post_clip",
                        post_clip_gradient_norm,
                        total_steps,
                    )
                    tb_writer.add_scalar(
                        "optimization/max_abs_grad_pre_clip",
                        max_abs_gradient_pre_clip,
                        total_steps,
                    )
                    tb_writer.add_scalar(
                        "optimization/norm_retention_after_value_clip",
                        gradient_norm_retention,
                        total_steps,
                    )
                    tb_writer.add_scalar(
                        "optimization/gradient_elements_clipped_fraction",
                        clipped_element_fraction,
                        total_steps,
                    )
                    tb_writer.add_scalar(
                        "optimization/value_clipped_step_fraction",
                        clipped_step_fraction,
                        total_steps,
                    )
                if microbatch_gradient_norms:
                    tb_writer.add_scalar(
                        "optimization/microbatch_grad_norm_mean",
                        microbatch_gradient_norm_mean,
                        total_steps,
                    )
                    tb_writer.add_scalar(
                        "optimization/microbatch_grad_norm_min",
                        microbatch_gradient_norm_min,
                        total_steps,
                    )
                    tb_writer.add_scalar(
                        "optimization/microbatch_grad_norm_max",
                        microbatch_gradient_norm_max,
                        total_steps,
                    )
                if microbatch_gradient_cosines:
                    tb_writer.add_scalar(
                        "optimization/microbatch_grad_cosine_mean",
                        microbatch_gradient_cosine_mean,
                        total_steps,
                    )
                    tb_writer.add_scalar(
                        "optimization/microbatch_grad_cosine_min",
                        microbatch_gradient_cosine_min,
                        total_steps,
                    )

            if total_steps % cfg.trainer.save_ckpt_freq == 0:
                save_path = Path(cfg.experiment_path) / f"model_{total_steps:06d}.pth"
                logging.info(f"Saving file {save_path}")
                _save_training_checkpoint(
                    fabric,
                    cfg.experiment_path,
                    save_path,
                    model,
                    optimizer,
                    scheduler,
                    total_steps,
                    cfg.reproducibility.seed,
                    wandb_run_id,
                    mixed_schedule.state_dict() if mixed_training else None,
                    source_cursors,
                )

            if total_steps % cfg.trainer.eval_freq == 0:
                _run_eval(
                    fabric,
                    cfg,
                    evaluator,
                    model,
                    eval_dataloaders,
                    tb_writer,
                    total_steps,
                )
                last_eval_step = total_steps

            if fabric.global_rank == 0:
                tqdm_epoch.update(1)
                tqdm_total_steps.update(1)
                tqdm_epoch.set_postfix(
                    loss=mean_loss_value,
                    lr=optimizer.param_groups[0]["lr"],
                    train_iters=cfg.trainer.train_iters,
                    gamma=cfg.trainer.gamma,
                    seq_name=batch.seq_name,
                )

            total_duration = training_step_duration
            reduced_timing = _reduce_scalar_dict(
                fabric,
                {
                    "total": total_duration,
                    "dataloader": dataloader_duration,
                    "forward": fwd_duration,
                    "sync": sync_duration,
                    "backward": bwd_duration,
                    "loader_worker": (
                        accumulated_loader_worker_seconds / microbatches_accumulated
                    ),
                    "gpu_jpeg_decode": (
                        accumulated_gpu_jpeg_decode_ms / microbatches_accumulated
                    ),
                    "gpu_prepare": (
                        accumulated_gpu_prepare_ms / microbatches_accumulated
                    ),
                },
            )
            total_duration = reduced_timing["total"]
            dataloader_duration = reduced_timing["dataloader"]
            fwd_duration = reduced_timing["forward"]
            sync_duration = reduced_timing["sync"]
            bwd_duration = reduced_timing["backward"]
            step_wall_seconds = _reduce_scalar(
                fabric, training_step_duration, reduce_op="max"
            )
            global_sample_count = _reduce_scalar(
                fabric, accumulated_sample_count, reduce_op="sum"
            )
            global_trajectory_count = _reduce_scalar(
                fabric, accumulated_trajectory_count, reduce_op="sum"
            )
            throughput_metrics = _throughput_metrics(
                step_wall_seconds,
                global_sample_count,
                global_trajectory_count,
            )
            sampling_metrics = _reduce_scalar_dict(
                fabric,
                {
                    name: value / microbatches_accumulated
                    for name, value in accumulated_sampling_metrics.items()
                },
            )
            logging.info(
                "[dali_stream:%06d rank=%d] batches=%d payload_bytes=%d "
                "batch_wait_seconds=%.3f",
                total_steps,
                fabric.global_rank,
                int(accumulated_dali_stream_metrics["batch_count"]),
                int(accumulated_dali_stream_metrics["payload_bytes"]),
                accumulated_dali_stream_metrics["batch_wait_seconds"],
            )
            reduced_dali_stream_metrics = {
                "batch_count": _reduce_scalar(
                    fabric,
                    accumulated_dali_stream_metrics["batch_count"],
                    reduce_op="sum",
                ),
                "payload_bytes": _reduce_scalar(
                    fabric,
                    accumulated_dali_stream_metrics["payload_bytes"],
                    reduce_op="sum",
                ),
                "batch_wait_seconds": _reduce_scalar(
                    fabric,
                    accumulated_dali_stream_metrics["batch_wait_seconds"],
                    reduce_op="max",
                ),
            }
            dali_wait_seconds = reduced_dali_stream_metrics[
                "batch_wait_seconds"
            ]
            reduced_dali_stream_metrics["effective_mib_per_second"] = (
                reduced_dali_stream_metrics["payload_bytes"]
                / dali_wait_seconds
                / 1024**2
                if dali_wait_seconds > 0
                else 0.0
            )
            reduced_physical_batching_metrics = None
            if physical_batching_metrics is not None:
                reduced_physical_batching_metrics = {
                    "planning_seconds": _reduce_scalar(
                        fabric,
                        physical_batching_metrics["planning_seconds"],
                        reduce_op="max",
                    ),
                    "materialization_seconds": _reduce_scalar(
                        fabric,
                        physical_batching_metrics["materialization_seconds"],
                        reduce_op="max",
                    ),
                    "encoded_cache_gib": _reduce_scalar(
                        fabric,
                        physical_batching_metrics["encoded_cache_gib"],
                        reduce_op="sum",
                    ),
                    "pair_count": _reduce_scalar(
                        fabric, physical_batching_metrics["pair_count"]
                    ),
                    "padding_tracks": _reduce_scalar(
                        fabric, physical_batching_metrics["padding_tracks"]
                    ),
                    "physical_group_count": _reduce_scalar(
                        fabric,
                        physical_batching_metrics["physical_group_count"],
                        reduce_op="sum",
                    ),
                }
            logging.info(
                f"[timing:{total_steps:06d}] "
                f"Total: {total_duration:>6.2f}s | "
                f"Data: {dataloader_duration:>6.2f}s | "
                f"Fwd: {fwd_duration:>6.2f}s | "
                f"Sync: {sync_duration:>6.2f}s | "
                f"Bwd: {bwd_duration:>6.2f}s | "
            )
            if fabric.global_rank == 0:
                dataloader_durations.append(dataloader_duration)
                fwd_durations.append(fwd_duration)
                sync_durations.append(sync_duration)
                bwd_durations.append(bwd_duration)
                total_durations.append(total_duration)

                tb_writer.add_scalar(f"timing/step", total_duration, total_steps)
                tb_writer.add_scalar(
                    "timing/step_wall", step_wall_seconds, total_steps
                )
                tb_writer.add_scalar(f"timing/only_fwd", fwd_durations[-1], total_steps)
                tb_writer.add_scalar(f"timing/only_sync", sync_durations[-1], total_steps)
                tb_writer.add_scalar(f"timing/only_bwd", bwd_durations[-1], total_steps)
                tb_writer.add_scalar(f"timing/only_dataloader", dataloader_duration, total_steps)
                tb_writer.add_scalar(
                    "timing/loader_worker_prepare_seconds",
                    reduced_timing["loader_worker"],
                    total_steps,
                )
                tb_writer.add_scalar(
                    "timing/gpu_jpeg_decode_ms",
                    reduced_timing["gpu_jpeg_decode"],
                    total_steps,
                )
                tb_writer.add_scalar(
                    "timing/gpu_batch_prepare_ms",
                    reduced_timing["gpu_prepare"],
                    total_steps,
                )
                for name, value in throughput_metrics.items():
                    tb_writer.add_scalar(name, value, total_steps)
                for name, value in hardware_metrics.items():
                    tb_writer.add_scalar(name, value, total_steps)
                if reduced_physical_batching_metrics is not None:
                    for name, value in reduced_physical_batching_metrics.items():
                        tb_writer.add_scalar(
                            f"batching/{name}", value, total_steps
                        )
                for name, value in reduced_dali_stream_metrics.items():
                    tb_writer.add_scalar(
                        f"io/dali_stream/{name}", value, total_steps
                    )
                if sampling_metrics:
                    for name, value in sampling_metrics.items():
                        tb_writer.add_scalar(f"sampling/{name}", value, total_steps)
                    logging.info(
                        f"[sampling:{total_steps:06d}] "
                        f"window_mean={sampling_metrics['motion_window_mean_m']:.3f}m "
                        f"window_static={sampling_metrics['motion_window_static_count']:.1f} "
                        f"full_dynamic_window_static="
                        f"{sampling_metrics['motion_full_dynamic_window_static_count']:.1f}"
                    )

                if len(total_durations) >= timing_log_freq:
                    total_durations_np = np.array(total_durations)
                    fwd_durations_np = np.array(fwd_durations)
                    sync_durations_np = np.array(sync_durations)
                    bwd_durations_np = np.array(bwd_durations)
                    dataloader_durations_np = np.array(dataloader_durations)

                    total_duration_mean = np.mean(total_durations_np)
                    fwd_duration_mean = np.mean(fwd_durations_np)
                    sync_duration_mean = np.mean(sync_durations_np)
                    bwd_duration_mean = np.mean(bwd_durations_np)
                    dataloader_duration_mean = np.mean(dataloader_durations_np)

                    total_duration_median = np.median(total_durations_np)
                    fwd_duration_median = np.median(fwd_durations_np)
                    sync_duration_median = np.median(sync_durations_np)
                    bwd_duration_median = np.median(bwd_durations_np)
                    dataloader_duration_median = np.median(dataloader_durations_np)

                    total_duration_std = np.std(total_durations_np)
                    fwd_duration_std = np.std(fwd_durations_np)
                    sync_duration_std = np.std(sync_durations_np)
                    bwd_duration_std = np.std(bwd_durations_np)
                    dataloader_duration_std = np.std(dataloader_durations_np)

                    tb_writer.add_scalar("timing/step_mean", total_duration_mean, total_steps)
                    tb_writer.add_scalar("timing/step_median", total_duration_median, total_steps)
                    tb_writer.add_scalar("timing/only_fwd_mean", fwd_duration_mean, total_steps)
                    tb_writer.add_scalar("timing/only_fwd_median", fwd_duration_median, total_steps)
                    tb_writer.add_scalar("timing/only_sync_mean", sync_duration_mean, total_steps)
                    tb_writer.add_scalar("timing/only_sync_median", sync_duration_median, total_steps)
                    tb_writer.add_scalar("timing/only_bwd_mean", bwd_duration_mean, total_steps)
                    tb_writer.add_scalar("timing/only_bwd_median", bwd_duration_median, total_steps)
                    tb_writer.add_scalar("timing/only_dataloader_mean", dataloader_duration_mean, total_steps)
                    tb_writer.add_scalar("timing/only_dataloader_median", dataloader_duration_median, total_steps)

                    logging.info(
                        f"[timing:total] "
                        f"Mean: {total_duration_mean:>6.2f}s | "
                        f"Median: {total_duration_median:>6.2f}s | "
                        f"Std: {total_duration_std:6.2f}s"
                    )
                    logging.info(
                        f"[timing:fwd]   "
                        f"Mean: {fwd_duration_mean:>6.2f}s | "
                        f"Median: {fwd_duration_median:>6.2f}s | "
                        f"Std: {fwd_duration_std:6.2f}s"
                    )
                    logging.info(
                        f"[timing:sync]  "
                        f"Mean: {sync_duration_mean:>6.2f}s | "
                        f"Median: {sync_duration_median:>6.2f}s | "
                        f"Std: {sync_duration_std:6.2f}s"
                    )
                    logging.info(
                        f"[timing:bwd]   "
                        f"Mean: {bwd_duration_mean:>6.2f}s | "
                        f"Median: {bwd_duration_median:>6.2f}s | "
                        f"Std: {bwd_duration_std:6.2f}s"
                    )
                    logging.info(
                        f"[timing:datal] "
                        f"Mean: {dataloader_duration_mean:>6.2f}s | "
                        f"Median: {dataloader_duration_median:>6.2f}s | "
                        f"Std: {dataloader_duration_std:6.2f}s"
                    )

                    total_durations.clear()
                    fwd_durations.clear()
                    sync_durations.clear()
                    bwd_durations.clear()
                    dataloader_durations.clear()

            microbatches_accumulated = 0
            accumulation_started_at = None
            accumulated_dataloader_duration = 0.0
            accumulated_fwd_duration = 0.0
            accumulated_sync_duration = 0.0
            accumulated_bwd_duration = 0.0
            accumulated_loader_worker_seconds = 0.0
            accumulated_gpu_jpeg_decode_ms = 0.0
            accumulated_gpu_prepare_ms = 0.0
            accumulated_sampling_metrics = {}
            accumulated_dali_stream_metrics = {
                "batch_count": 0.0,
                "payload_bytes": 0.0,
                "batch_wait_seconds": 0.0,
            }
            accumulated_loss_value = None
            accumulated_component_losses = {}
            accumulated_metrics = {}
            accumulated_source_losses = {}
            accumulated_source_components = {}
            accumulated_source_metrics = {}
            accumulated_source_counts = {}
            accumulated_source_view_counts = {}
            accumulated_source_track_counts = {}
            accumulated_sample_count = 0
            accumulated_trajectory_count = 0.0
            microbatch_gradient_norms = []
            microbatch_gradient_cosines = []
            physical_step = None
            physical_group_iterator = None
            physical_group_count = None

            if torch_profiler is not None:
                torch_profiler.step()

            if total_steps >= cfg.trainer.num_steps:
                should_keep_training = False
                break

        if fabric.global_rank == 0:
            tqdm_epoch.close()

    if fabric.global_rank == 0:
        tqdm_total_steps.close()
    logging.info("FINISHED TRAINING")

    if torch_profiler is not None:
        torch_profiler.stop()
    gradient_diagnostics.close()
    gpu_monitor.close()

    save_path = f"{cfg.experiment_path}/model_final.pth"
    logging.info(f"Saving file {save_path}")
    _save_training_checkpoint(
        fabric,
        cfg.experiment_path,
        save_path,
        model,
        optimizer,
        scheduler,
        total_steps,
        cfg.reproducibility.seed,
        wandb_run_id,
        mixed_schedule.state_dict() if mixed_training else None,
        source_cursors,
    )
    if eval_dataloaders and last_eval_step != total_steps:
        _run_eval(
            fabric,
            cfg,
            evaluator,
            model,
            eval_dataloaders,
            tb_writer,
            total_steps,
        )
    for thread in threads:
        thread.join()
    if tb_writer is not None:
        tb_writer.flush()
        tb_writer.close()


if __name__ == "__main__":
    main()
