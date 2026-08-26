"""Profile one real MVTracker training shape on one CUDA device."""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
from pathlib import Path
from types import SimpleNamespace
import statistics
import threading
import time

import hydra
from hydra import compose, initialize_config_dir
from lightning.fabric import Fabric
from omegaconf import OmegaConf
import torch

from mvtracker.cli.train import (
    _scale_microbatch_loss,
    dataclass_to_cuda_,
    fetch_optimizer,
    forward_batch_multi_view,
)
from mvtracker.datasets import KubricMultiViewDataset
from mvtracker.datasets.utils import collate_fn
from mvtracker.profiling.modal_training import is_memory_safe, validate_view_count


class _NvmlPeak:
    def __init__(self) -> None:
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread = None

    def __enter__(self):
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

        def sample() -> None:
            while not self._stop.wait(0.02):
                used = int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
                self.peak_bytes = max(self.peak_bytes, used)

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        import pynvml

        self._stop.set()
        self._thread.join()
        pynvml.nvmlShutdown()


def _compose_config(arguments):
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    overrides = [
        "+experiment=diegesis",
        f"experiment_path={arguments.output.parent}",
        f"restore_ckpt_path={arguments.checkpoint}",
        f"datasets.root={arguments.data_root}",
        "datasets.train.name=kubric-multiview-v3-training",
        f"datasets.train.batch_size={arguments.batch_size}",
        f"datasets.train.traj_per_sample={arguments.trajectories}",
        f"datasets.train.num_workers={arguments.workers}",
        f"trainer.gradient_accumulation_steps={arguments.accumulation}",
        f"trainer.num_steps={arguments.warmup_updates + arguments.measure_updates + 10}",
        "augmentations.variable_trajpersample=false",
        "augmentations.variable_num_views=false",
        "logging.log_wandb=false",
        "datasets.eval.names=[]",
    ]
    if getattr(arguments, "checkpoint_updateformer", False):
        overrides.append("model.checkpoint_updateformer=true")
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name="train", overrides=overrides)


def _build_loader(cfg, fabric, arguments):
    kwargs = KubricMultiViewDataset.from_name(
        cfg.datasets.train.name,
        cfg.datasets.root,
        training_args=cfg,
        fabric=fabric,
        just_return_kwargs=True,
    )
    kwargs.update(
        num_views=arguments.views,
        views_to_return=None,
        enable_variable_num_views_augs=False,
    )
    dataset = KubricMultiViewDataset(**kwargs)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=arguments.batch_size,
        num_workers=arguments.workers,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=2,
        persistent_workers=arguments.workers > 0,
    )
    return loader


def prepare_profile_batch(
    *,
    data_root: Path,
    output: Path,
    views: int,
    batch_size: int,
    trajectories: int = 2048,
) -> dict:
    arguments = SimpleNamespace(
        data_root=data_root,
        output=output,
        checkpoint=output.with_suffix(".pth"),
        views=views,
        batch_size=batch_size,
        trajectories=trajectories,
        workers=0,
        accumulation=1,
        warmup_updates=0,
        measure_updates=1,
    )
    cfg = _compose_config(arguments)
    fabric = SimpleNamespace(world_size=1)
    kwargs = KubricMultiViewDataset.from_name(
        cfg.datasets.train.name,
        cfg.datasets.root,
        training_args=cfg,
        fabric=fabric,
        just_return_kwargs=True,
    )
    kwargs.update(
        num_views=views,
        views_to_return=None,
        enable_variable_num_views_augs=False,
        virtual_dataset_size=1000,
    )
    dataset = KubricMultiViewDataset(**kwargs)
    accepted = []
    attempted = 0
    for virtual_index in range(len(dataset)):
        attempted += 1
        sample, gotit = dataset[virtual_index]
        if not gotit or sample.trajectory.shape[-2] != trajectories:
            continue
        accepted.append((sample, True))
        if len(accepted) == batch_size:
            break
    if len(accepted) != batch_size:
        raise RuntimeError(
            f"found {len(accepted)}/{batch_size} exact {trajectories}-track samples "
            f"after {attempted} attempts"
        )
    batch, gotit = collate_fn(accepted)
    if not all(gotit) or batch.video.shape[:2] != (batch_size, views):
        raise RuntimeError("prepared profile batch has an unexpected shape")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(batch, output)
    return {
        "path": str(output),
        "views": views,
        "batch_size": batch_size,
        "trajectories": trajectories,
        "attempted_samples": attempted,
        "bytes": output.stat().st_size,
    }


def _slice_cached_batch(batch, batch_size: int, trajectories: int):
    cached_batch_size = int(batch.video.shape[0])
    if not 1 <= batch_size <= cached_batch_size:
        raise ValueError(
            f"requested batch size {batch_size} is outside cached batch size "
            f"{cached_batch_size}"
        )
    for field in dataclasses.fields(batch):
        value = getattr(batch, field.name)
        if isinstance(value, torch.Tensor) and value.ndim > 0:
            if value.shape[0] != cached_batch_size:
                raise ValueError(
                    f"cached batch field {field.name} has leading shape "
                    f"{value.shape[0]}, expected {cached_batch_size}"
                )
            setattr(batch, field.name, value[:batch_size])
        elif isinstance(value, list) and len(value) == cached_batch_size:
            setattr(batch, field.name, value[:batch_size])

    track_axes = {
        "trajectory": -2,
        "trajectory_3d": -2,
        "visibility": -1,
        "valid": -1,
        "query_points": -2,
        "query_points_3d": -2,
        "track_padding_mask": -1,
    }
    for name, axis in track_axes.items():
        value = getattr(batch, name)
        if value is None:
            continue
        if value.shape[axis] < trajectories:
            raise ValueError(
                f"cached batch has only {value.shape[axis]} tracks in {name}"
            )
        index = [slice(None)] * value.ndim
        index[axis] = slice(0, trajectories)
        setattr(batch, name, value[tuple(index)])
    batch.track_padding_mask = batch.track_padding_mask.bool()
    return batch


def _events(count: int) -> list[torch.cuda.Event]:
    return [torch.cuda.Event(enable_timing=True) for _ in range(count)]


def _run_update(
    *,
    iterator,
    model,
    optimizer,
    scheduler,
    fabric,
    cfg,
    arguments,
) -> dict[str, float | int]:
    optimizer.zero_grad(set_to_none=True)
    data_wait_seconds = 0.0
    jpeg_decode_ms = 0.0
    gpu_prepare_ms = 0.0
    forward_ms = 0.0
    backward_ms = 0.0
    source_tracks = []
    update_started = time.perf_counter()

    for _ in range(arguments.accumulation):
        load_started = time.perf_counter()
        for _ in range(16):
            try:
                batch, gotit = next(iterator)
            except StopIteration as error:
                raise RuntimeError(
                    "profile dataset exhausted before producing an exact batch"
                ) from error
            if all(gotit):
                counts = [
                    int((~batch.track_padding_mask[index].bool()).sum().item())
                    for index in range(arguments.batch_size)
                ]
                if all(count == arguments.trajectories for count in counts):
                    break
        else:
            raise RuntimeError(
                "profile dataset did not produce an exact batch in 16 attempts"
            )
        data_wait_seconds += time.perf_counter() - load_started
        if batch.video.shape[0] != arguments.batch_size:
            raise RuntimeError("profile batch size differs from the request")
        if batch.video.shape[1] != arguments.views:
            raise RuntimeError("profile view count differs from the request")
        source_tracks.extend(counts)
        for metadata in batch.sample_metadata or ():
            jpeg_decode_ms += float(metadata.get("gpu_jpeg_decode_ms", 0.0))
            gpu_prepare_ms += float(metadata.get("gpu_prepare_total_ms", 0.0))
        dataclass_to_cuda_(batch)

        forward_start, forward_end, backward_end = _events(3)
        forward_start.record()
        output = forward_batch_multi_view(
            batch=batch,
            model=model,
            cfg=cfg,
            step=1,
            train_iters=cfg.trainer.train_iters,
            gamma=cfg.trainer.gamma,
            save_debug_logs=False,
            debug_logs_path=None,
            run_expensive_diagnostics=False,
        )
        loss = sum(value["loss"] for value in output.values() if "loss" in value)
        forward_end.record()
        fabric.backward(_scale_microbatch_loss(loss, arguments.accumulation))
        backward_end.record()
        backward_end.synchronize()
        forward_ms += forward_start.elapsed_time(forward_end)
        backward_ms += forward_end.elapsed_time(backward_end)

    optimizer_start, optimizer_end = _events(2)
    optimizer_start.record()
    fabric.clip_gradients(model, optimizer, max_norm=cfg.trainer.grad_clip)
    optimizer.step()
    scheduler.step()
    optimizer_end.record()
    optimizer_end.synchronize()
    return {
        "data_wait_ms": data_wait_seconds * 1000.0,
        "jpeg_decode_ms": jpeg_decode_ms / arguments.accumulation,
        "gpu_prepare_ms": gpu_prepare_ms / arguments.accumulation,
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "optimizer_ms": optimizer_start.elapsed_time(optimizer_end),
        "total_update_ms": (time.perf_counter() - update_started) * 1000.0,
        "source_track_min": min(source_tracks),
        "source_track_max": max(source_tracks),
    }


def _median_metrics(updates: list[dict[str, float | int]]) -> dict[str, float]:
    return {
        name: float(statistics.median(float(update[name]) for update in updates))
        for name in updates[0]
    }


def run(arguments) -> dict:
    if torch.cuda.device_count() != 1:
        raise RuntimeError("training shape profiler requires exactly one visible GPU")
    validate_view_count(arguments.views)

    if arguments.gpu_lane is not None:
        expected_name = arguments.gpu_lane.rstrip("!").upper()
        actual_name = torch.cuda.get_device_name().upper()
        if expected_name not in actual_name:
            raise RuntimeError(
                f"requested GPU lane {arguments.gpu_lane!r}, got {actual_name!r}"
            )

    import flash_attn  # noqa: F401
    from mvtracker.models.core.mvtracker import mvtracker as mvtracker_module

    if mvtracker_module.knn.__name__ != "_knn_capturable":
        raise RuntimeError("capture-safe KNN is required for profiling")

    cfg = _compose_config(arguments)
    fabric = Fabric(devices=1, precision=cfg.trainer.precision)
    fabric.launch()
    fabric.seed_everything(int(cfg.reproducibility.seed), workers=True)
    model = hydra.utils.instantiate(cfg.model).cuda()
    checkpoint_updateformer = bool(model.updateformer.checkpoint_updateformer)
    optimizer, scheduler = fetch_optimizer(cfg.trainer, model)
    model, optimizer = fabric.setup(model, optimizer)
    fabric.load_raw(str(arguments.checkpoint), model)
    model.train()
    from mvtracker.models.core.mvtracker.indexed_correlation import (
        warmup_indexed_correlation,
    )
    warmup_seconds = warmup_indexed_correlation(fabric.device)
    print(f"indexed_correlation_warmup_seconds={warmup_seconds:.3f}")
    if arguments.batch_cache is None:
        loader = _build_loader(cfg, fabric, arguments)
        iterator = iter(loader)
    else:
        batch = torch.load(arguments.batch_cache, map_location="cpu", weights_only=False)
        batch = _slice_cached_batch(
            batch, arguments.batch_size, arguments.trajectories
        )
        iterator = itertools.repeat((batch, [True] * arguments.batch_size))

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    measured = []
    with _NvmlPeak() as nvml:
        for update_index in range(arguments.warmup_updates + arguments.measure_updates):
            result = _run_update(
                iterator=iterator,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                fabric=fabric,
                cfg=cfg,
                arguments=arguments,
            )
            if update_index >= arguments.warmup_updates:
                measured.append(result)

    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    peak_allocated = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    peak_observed = max(nvml.peak_bytes, peak_allocated, peak_reserved)
    median = _median_metrics(measured)
    scenes_per_second = (
        arguments.batch_size
        * arguments.accumulation
        / (median["total_update_ms"] / 1000.0)
    )
    return {
        "status": (
            "safe" if is_memory_safe(peak_observed, total_memory) else "unsafe"
        ),
        "checkpoint_updateformer": checkpoint_updateformer,
        "views": arguments.views,
        "batch_size": arguments.batch_size,
        "accumulation": arguments.accumulation,
        "trajectories": arguments.trajectories,
        "warmup_updates": arguments.warmup_updates,
        "measure_updates": arguments.measure_updates,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "peak_nvml_bytes": nvml.peak_bytes,
        "peak_memory_bytes": peak_observed,
        "total_memory_bytes": total_memory,
        "peak_memory_fraction": peak_observed / total_memory,
        "median": median,
        "updates": measured,
        "scenes_per_second": scenes_per_second,
        "trajectories_per_second": scenes_per_second * arguments.trajectories,
        "indexed_correlation_warmup_seconds": warmup_seconds,
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
    }


def compatibility_check(gpu_lane: str | None = None) -> dict[str, str | bool | float]:
    """Import every required compiled backend on exactly one visible GPU."""
    if torch.cuda.device_count() != 1:
        raise RuntimeError("compatibility check requires exactly one visible GPU")
    actual_name = torch.cuda.get_device_name().upper()
    if gpu_lane is not None and gpu_lane.rstrip("!").upper() not in actual_name:
        raise RuntimeError(
            f"requested GPU lane {gpu_lane!r}, got {actual_name!r}"
        )
    import flash_attn
    import pointops
    import spconv

    from mvtracker.models.core.mvtracker import indexed_correlation
    warmup_seconds = indexed_correlation.warmup_indexed_correlation()

    return {
        "gpu": actual_name,
        "torch": str(torch.__version__),
        "flash_attn": getattr(flash_attn, "__version__", "unknown"),
        "pointops": getattr(pointops, "__version__", "installed"),
        "spconv": getattr(spconv, "__version__", "installed"),
        "indexed_correlation_source": str(indexed_correlation.__file__),
        "indexed_correlation_warmup_seconds": warmup_seconds,
    }


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-cache", type=Path)
    parser.add_argument("--views", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation", type=int)
    parser.add_argument("--trajectories", type=int)
    parser.add_argument("--warmup-updates", type=int)
    parser.add_argument("--measure-updates", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--gpu-lane", choices=("H100!", "H200", "B200"))
    parser.add_argument("--compatibility-only", action="store_true")
    parser.add_argument("--checkpoint-updateformer", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        if arguments.compatibility_only:
            result = {
                "status": "compatible",
                **compatibility_check(arguments.gpu_lane),
            }
        else:
            required = (arguments.data_root, arguments.checkpoint)
            required += (
                arguments.views,
                arguments.batch_size,
                arguments.accumulation,
                arguments.trajectories,
                arguments.warmup_updates,
                arguments.measure_updates,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "all data, shape, and measurement arguments are required "
                    "for profiling"
                )
            result = run(arguments)
    except torch.cuda.OutOfMemoryError as error:
        total_memory = int(torch.cuda.get_device_properties(0).total_memory)
        result = {
            "status": "oom",
            "views": arguments.views,
            "batch_size": arguments.batch_size,
            "accumulation": arguments.accumulation,
            "trajectories": arguments.trajectories,
            "peak_memory_bytes": int(torch.cuda.max_memory_reserved()),
            "total_memory_bytes": total_memory,
            "error": str(error),
        }
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
