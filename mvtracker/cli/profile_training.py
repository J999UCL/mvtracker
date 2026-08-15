"""Profile one real MVTracker training shape on one CUDA device."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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
    fetch_optimizer,
    forward_batch_multi_view,
)
from mvtracker.datasets import TapVid3DMultiViewDataset
from mvtracker.datasets.tapvid3d_multiview_dataset import CudaPrefetchLoader
from mvtracker.datasets.utils import HomogeneousViewBatchSampler
from mvtracker.profiling.modal_training import is_memory_safe


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
    os.environ["DIEGESIS_MVTRACKER_ROOT"] = str(arguments.data_root)
    config_dir = Path(__file__).resolve().parents[2] / "configs"
    probabilities = [0.0] * 4
    probabilities[arguments.views - 1] = 1.0
    overrides = [
        "+experiment=diegesis",
        f"experiment_path={arguments.output.parent}",
        f"restore_ckpt_path={arguments.checkpoint}",
        f"datasets.root={arguments.data_root}",
        f"datasets.train.batch_size={arguments.batch_size}",
        f"datasets.train.traj_per_sample={arguments.trajectories}",
        f"datasets.train.num_workers={arguments.workers}",
        f"trainer.gradient_accumulation_steps={arguments.accumulation}",
        f"trainer.num_steps={arguments.warmup_updates + arguments.measure_updates + 10}",
        f"datasets.tapvid3d_view_count_probabilities={probabilities}",
        "augmentations.variable_trajpersample=false",
        "logging.log_wandb=false",
        "datasets.eval.names=[]",
    ]
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        return compose(config_name="train", overrides=overrides)


def _build_loader(cfg, fabric, arguments):
    dataset = TapVid3DMultiViewDataset.from_name(
        cfg.datasets.train.name,
        cfg.datasets.root,
        cfg,
        fabric,
    )
    sampler = HomogeneousViewBatchSampler(
        dataset,
        arguments.batch_size,
        seed=int(cfg.reproducibility.seed),
        view_count_probabilities=cfg.datasets.tapvid3d_view_count_probabilities,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=arguments.workers,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        prefetch_factor=2,
        persistent_workers=arguments.workers > 0,
    )
    return CudaPrefetchLoader(loader, fabric.device)


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
        batch, gotit = next(iterator)
        data_wait_seconds += time.perf_counter() - load_started
        if not all(gotit):
            raise RuntimeError("profile dataset returned an invalid sample")
        if batch.video.shape[0] != arguments.batch_size:
            raise RuntimeError("profile batch size differs from the request")
        if batch.video.shape[1] != arguments.views:
            raise RuntimeError("profile view count differs from the request")
        source_tracks.extend(
            int((~batch.track_padding_mask[index]).sum().item())
            for index in range(arguments.batch_size)
        )
        if any(count != arguments.trajectories for count in source_tracks):
            raise RuntimeError(
                "profile samples did not realize the requested trajectory count: "
                f"requested={arguments.trajectories}, actual={source_tracks}"
            )
        for metadata in batch.sample_metadata:
            jpeg_decode_ms += float(metadata["gpu_jpeg_decode_ms"])
            gpu_prepare_ms += float(metadata["gpu_prepare_total_ms"])

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
    fabric.clip_gradients(model, optimizer, clip_val=cfg.trainer.grad_clip)
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
    if not 1 <= arguments.views <= 4:
        raise ValueError("views must be between one and four")

    import flash_attn  # noqa: F401
    from mvtracker.models.core.mvtracker import mvtracker as mvtracker_module

    if mvtracker_module.knn.__name__ != "_knn_pointops":
        raise RuntimeError("PointOps KNN is required for profiling")

    cfg = _compose_config(arguments)
    fabric = Fabric(devices=1, precision=cfg.trainer.precision)
    fabric.launch()
    fabric.seed_everything(int(cfg.reproducibility.seed), workers=True)
    model = hydra.utils.instantiate(cfg.model).cuda()
    optimizer, scheduler = fetch_optimizer(cfg.trainer, model)
    model, optimizer = fabric.setup(model, optimizer)
    fabric.load_raw(str(arguments.checkpoint), model)
    model.train()
    loader = _build_loader(cfg, fabric, arguments)
    iterator = iter(loader)

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
        "resolved_config": OmegaConf.to_container(cfg, resolve=True),
    }


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--accumulation", type=int, required=True)
    parser.add_argument("--trajectories", type=int, required=True)
    parser.add_argument("--warmup-updates", type=int, required=True)
    parser.add_argument("--measure-updates", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    arguments = _arguments()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    try:
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
