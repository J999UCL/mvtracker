"""One real-update correctness and timing gate for UpdateFormer backends."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import time
import statistics

import hydra
from lightning.fabric import Fabric
import torch

from mvtracker.cli.profile_training import _compose_config
from mvtracker.cli.train import dataclass_to_cuda_, fetch_optimizer, forward_batch_multi_view


TRAJECTORY_RMS_METERS = 0.001
TRAJECTORY_P99_METERS = 0.005
VISIBILITY_MEAN_ABS = 0.001
VISIBILITY_FLIP_FRACTION = 0.001
LOSS_RELATIVE_ERROR = 0.001
GRADIENT_COSINE = 0.9999
GRADIENT_NORM_RELATIVE_ERROR = 0.01


def _arguments(data_root, checkpoint, output, batch_size, views, trajectories):
    return SimpleNamespace(
        data_root=Path(data_root),
        checkpoint=Path(checkpoint),
        output=Path(output),
        views=int(views),
        batch_size=int(batch_size),
        trajectories=int(trajectories),
        workers=0,
        accumulation=1,
        warmup_updates=0,
        measure_updates=1,
    )


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu()
    if isinstance(value, list):
        return [_cpu(item) for item in value]
    if isinstance(value, dict):
        return {name: _cpu(item) for name, item in value.items()}
    return value


def _run_backend(cfg, checkpoint, batch, backend, warm_updates=4):
    cfg.model.updateformer_backend = backend
    cfg.model.checkpoint_updateformer = False
    model = hydra.utils.instantiate(cfg.model).cuda().train()
    fabric = Fabric(devices=1, precision=cfg.trainer.precision)
    fabric.load_raw(str(checkpoint), model)
    optimizer, _ = fetch_optimizer(cfg.trainer, model)
    initial = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    batch = copy.deepcopy(batch)
    dataclass_to_cuda_(batch)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, cache_enabled=False):
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
            capture_training_trace=True,
        )
        loss = output["flow"]["loss"] + output["visibility"]["loss"]
    torch.cuda.synchronize()
    forward_seconds = time.perf_counter() - started

    started = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize()
    backward_seconds = time.perf_counter() - started
    gradients = {
        name: parameter.grad.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    started = time.perf_counter()
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.grad_clip)
    optimizer.step()
    torch.cuda.synchronize()
    optimizer_seconds = time.perf_counter() - started
    updated = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    result = {
        "loss": float(loss.detach()),
        "flow_loss": float(output["flow"]["loss"].detach()),
        "visibility_loss": float(output["visibility"]["loss"].detach()),
        "final_trajectories": _cpu(output["flow"]["predictions_worldspace"]),
        "final_visibility": _cpu(output["visibility"]["predictions"]),
        "trace": _cpu(output["training_trace"]),
        "gradients": gradients,
        "parameter_updates": {
            name: updated[name] - value for name, value in initial.items()
        },
        "timing": {
            "forward_seconds": forward_seconds,
            "backward_seconds": backward_seconds,
            "optimizer_seconds": optimizer_seconds,
        },
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    warm_timings = []
    for _ in range(warm_updates):
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, cache_enabled=False
        ):
            warm_output = forward_batch_multi_view(
                batch=batch,
                model=model,
                cfg=cfg,
                step=2,
                train_iters=cfg.trainer.train_iters,
                gamma=cfg.trainer.gamma,
                save_debug_logs=False,
                debug_logs_path=None,
                run_expensive_diagnostics=False,
            )
            warm_loss = (
                warm_output["flow"]["loss"]
                + warm_output["visibility"]["loss"]
            )
        warm_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.trainer.grad_clip)
        optimizer.step()
        torch.cuda.synchronize()
        warm_timings.append(time.perf_counter() - started)
    result["timing"]["warm_update_seconds"] = warm_timings
    result["timing"]["warm_update_median_seconds"] = statistics.median(
        warm_timings
    )
    final_parameters = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    result["multi_update"] = {
        "updates": warm_updates + 1,
        "loss": float(warm_loss.detach()),
        "final_trajectories": _cpu(
            warm_output["flow"]["predictions_worldspace"]
        ),
        "final_visibility": _cpu(warm_output["visibility"]["predictions"]),
        "parameter_updates": {
            name: final_parameters[name] - value
            for name, value in initial.items()
        },
    }
    del model, optimizer, batch, output, loss
    torch.cuda.empty_cache()
    return result


def _difference(left, right):
    difference = (left.float() - right.float()).abs().reshape(-1)
    return {
        "rms": float(difference.square().mean().sqrt()),
        "mean": float(difference.mean()),
        "p99": float(torch.quantile(difference, 0.99)),
        "max": float(difference.max()),
    }


def _vector_agreement(left, right):
    if left.keys() != right.keys():
        raise RuntimeError("candidate gradient/state keys differ from eager")
    dot = torch.zeros((), dtype=torch.float64)
    left_norm = torch.zeros((), dtype=torch.float64)
    right_norm = torch.zeros((), dtype=torch.float64)
    max_abs = 0.0
    for name in left:
        lhs = left[name].double().reshape(-1)
        rhs = right[name].double().reshape(-1)
        dot += torch.dot(lhs, rhs)
        left_norm += torch.dot(lhs, lhs)
        right_norm += torch.dot(rhs, rhs)
        max_abs = max(max_abs, float((lhs - rhs).abs().max()))
    left_norm = left_norm.sqrt()
    right_norm = right_norm.sqrt()
    return {
        "cosine": float(dot / (left_norm * right_norm).clamp_min(1e-30)),
        "norm_relative_error": float(
            (right_norm - left_norm).abs() / left_norm.clamp_min(1e-30)
        ),
        "max_absolute_error": max_abs,
    }


def _trace_difference(eager, fused):
    result = {}
    for name in ("coordinates", "visibility_logits"):
        left = eager[name]
        right = fused[name]
        if len(left) != len(right):
            raise RuntimeError(f"candidate {name} trace length differs from eager")
        result[name] = _difference(
            torch.cat([value.reshape(-1) for value in left]),
            torch.cat([value.reshape(-1) for value in right]),
        )
    return result


def compare_real_update(
    *,
    data_root: Path,
    checkpoint: Path,
    batch_cache: Path,
    output: Path,
    candidate_backend: str = "fused",
):
    batch = torch.load(batch_cache, map_location="cpu", weights_only=False)
    batch_size = int(batch.video.shape[0])
    views = int(batch.video.shape[1])
    trajectories = int(batch.query_points_3d.shape[1])
    arguments = _arguments(
        data_root, checkpoint, output, batch_size, views, trajectories
    )
    cfg = _compose_config(arguments)
    eager = _run_backend(cfg, checkpoint, batch, "eager")
    fused = _run_backend(cfg, checkpoint, batch, candidate_backend)

    final_trajectory = _difference(
        eager["final_trajectories"], fused["final_trajectories"]
    )
    visibility = _difference(eager["final_visibility"], fused["final_visibility"])
    visibility_flips = float(
        (
            (eager["final_visibility"] >= 0.5)
            != (fused["final_visibility"] >= 0.5)
        ).float().mean()
    )
    loss_relative_error = abs(fused["loss"] - eager["loss"]) / max(
        abs(eager["loss"]), 1e-12
    )
    gradients = _vector_agreement(eager["gradients"], fused["gradients"])
    updates = _vector_agreement(
        eager["parameter_updates"], fused["parameter_updates"]
    )
    trace = _trace_difference(eager["trace"], fused["trace"])
    multi_trajectory = _difference(
        eager["multi_update"]["final_trajectories"],
        fused["multi_update"]["final_trajectories"],
    )
    multi_visibility = _difference(
        eager["multi_update"]["final_visibility"],
        fused["multi_update"]["final_visibility"],
    )
    multi_updates = _vector_agreement(
        eager["multi_update"]["parameter_updates"],
        fused["multi_update"]["parameter_updates"],
    )
    passed = all((
        final_trajectory["rms"] <= TRAJECTORY_RMS_METERS,
        final_trajectory["p99"] <= TRAJECTORY_P99_METERS,
        visibility["mean"] <= VISIBILITY_MEAN_ABS,
        visibility_flips <= VISIBILITY_FLIP_FRACTION,
        loss_relative_error <= LOSS_RELATIVE_ERROR,
        gradients["cosine"] >= GRADIENT_COSINE,
        gradients["norm_relative_error"] <= GRADIENT_NORM_RELATIVE_ERROR,
        updates["cosine"] >= GRADIENT_COSINE,
        updates["norm_relative_error"] <= GRADIENT_NORM_RELATIVE_ERROR,
    ))
    return {
        "passed": passed,
        "candidate_backend": candidate_backend,
        "shape": {
            "views": views,
            "batch_size": batch_size,
            "trajectories": trajectories,
        },
        "loss": {
            "eager": eager["loss"],
            "fused": fused["loss"],
            "relative_error": loss_relative_error,
        },
        "final_trajectory": final_trajectory,
        "final_visibility": {**visibility, "flip_fraction": visibility_flips},
        "trace": trace,
        "gradients": gradients,
        "parameter_updates": updates,
        "multi_update": {
            "updates": eager["multi_update"]["updates"],
            "loss": {
                "eager": eager["multi_update"]["loss"],
                "fused": fused["multi_update"]["loss"],
            },
            "final_trajectory": multi_trajectory,
            "final_visibility": multi_visibility,
            "parameter_updates": multi_updates,
        },
        "timing": {"eager": eager["timing"], "fused": fused["timing"]},
        "memory": {
            backend: {
                "peak_allocated_bytes": result["peak_allocated_bytes"],
                "peak_reserved_bytes": result["peak_reserved_bytes"],
            }
            for backend, result in (("eager", eager), ("fused", fused))
        },
    }
