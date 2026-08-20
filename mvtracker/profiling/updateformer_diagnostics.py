"""Numerical diagnostics against the locked UpdateFormer golden tensors."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

from mvtracker.profiling.updateformer_contract import (
    WORKLOADS,
    Workload,
    _build_model,
    _close_mismatch,
    _run_loss_contract,
    _run_updateformer_case,
    _workload_tensors,
)


def _pairs(expected, actual, path="root"):
    if isinstance(expected, torch.Tensor):
        yield path, expected, actual
    elif isinstance(expected, dict):
        for key in expected:
            yield from _pairs(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        for index, (left, right) in enumerate(zip(expected, actual)):
            yield from _pairs(left, right, f"{path}[{index}]")


def _ordered_low_precision(tensor):
    bits = tensor.view(torch.int16).to(torch.int32) & 0xFFFF
    return torch.where(
        bits & 0x8000 != 0,
        0xFFFF - bits,
        0x8000 + bits,
    )


def _record(path, expected, actual):
    difference = (actual.float() - expected.float()).abs()
    record = {
        "path": path,
        "shape": list(expected.shape),
        "dtype": str(expected.dtype),
        "maximum_absolute_error": difference.max().item(),
        "mean_absolute_error": difference.mean().item(),
        "changed_elements": int((difference > 0).sum().item()),
        "elements": expected.numel(),
    }
    if expected.dtype in {torch.bfloat16, torch.float16}:
        ulp = (_ordered_low_precision(actual) - _ordered_low_precision(expected)).abs()
        record["maximum_ulp_error"] = int(ulp.max().item())
        record["elements_over_one_ulp"] = int((ulp > 1).sum().item())
    return record


def compare_candidate(root: Path) -> dict[str, object]:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    state = torch.load(
        root / manifest["baseline_state"], map_location="cpu", weights_only=True
    )
    reference = torch.load(
        root / manifest["reference"], map_location="cpu", weights_only=True
    )
    device = torch.device("cuda:0")
    actual_cases = {
        workload.name: _run_updateformer_case(workload, state, device)
        for workload in WORKLOADS
    }


def diagnose_fused_components(root: Path) -> dict[str, object]:
    """Separate numerical drift from track padding and fused self-QKV."""
    from mvtracker.models.core.cotracker2.blocks import (
        FlashAttention,
        FusedFlashAttention,
        updateformer_track_capacity,
    )

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    state = torch.load(
        root / manifest["baseline_state"], map_location="cpu", weights_only=True
    )
    device = torch.device("cuda:0")
    workload = next(item for item in WORKLOADS if item.name == "paired_ragged_900")
    inputs, target, weights, mask = _workload_tensors(workload, device)
    base_inputs = inputs.detach()

    def run(*, padded, fused_qkv):
        model = _build_model(device)
        model.load_state_dict(state, strict=True)
        model.checkpoint_updateformer = False
        if fused_qkv:
            for module in model.modules():
                if type(module) is FlashAttention:
                    module.__class__ = FusedFlashAttention
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=5e-5, weight_decay=1e-5
        )
        value = base_inputs.clone().requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, cache_enabled=False
        ):
            if padded:
                capacity = updateformer_track_capacity(value.shape[1])
                padding = capacity - value.shape[1]
                padded_value = F.pad(value, (0, 0, 0, 0, 0, padding))
                padded_mask = F.pad(mask, (0, padding), value=False)
                output = model._forward_impl(padded_value, padded_mask)[
                    :, :value.shape[1]
                ]
            else:
                output = model._forward_impl(value, mask)
            valid = mask[:, :, None, None]
            loss = ((((output.float() - target) * weights).square()) * valid).sum()
            loss = loss / (valid.sum() * output.shape[2] * output.shape[3])
        loss.backward()
        result = {
            "output": output.detach().cpu(),
            "loss": loss.detach().cpu(),
            "input_gradient": value.grad.detach().cpu(),
            "parameter_gradients": {
                name: parameter.grad.detach().cpu()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            },
        }
        optimizer.step()
        result["updated_parameters"] = {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
        }
        return result

    variants = {
        "padding": run(padded=True, fused_qkv=False),
        "qkv": run(padded=False, fused_qkv=True),
        "padding_qkv": run(padded=True, fused_qkv=True),
    }
    eager = run(padded=False, fused_qkv=False)
    summaries = {}
    for name, value in variants.items():
        records = [
            _record(path, expected, actual)
            for path, expected, actual in _pairs(eager, value)
            if expected.is_floating_point()
        ]
        changed = [record for record in records if record["changed_elements"]]
        summaries[name] = {
            "changed_tensors": len(changed),
            "maximum_absolute_error": max(
                (record["maximum_absolute_error"] for record in changed),
                default=0.0,
            ),
            "mean_absolute_error": sum(
                record["mean_absolute_error"] * record["elements"]
                for record in changed
            ) / max(sum(record["elements"] for record in changed), 1),
            "elements_over_one_ulp": sum(
                record.get("elements_over_one_ulp", 0) for record in changed
            ),
            "largest": sorted(
                changed,
                key=lambda record: record["maximum_absolute_error"],
                reverse=True,
            )[:10],
        }
    return {"workload": workload.name, "variants": summaries}
    actual_losses = _run_loss_contract(device)
    records = [
        _record(path, expected, actual)
        for path, expected, actual in _pairs(
            reference,
            {"cases": actual_cases, "loss_contract": actual_losses},
        )
        if expected.is_floating_point()
    ]
    changed = [record for record in records if record["changed_elements"]]
    return {
        "floating_tensors": len(records),
        "changed_tensors": len(changed),
        "low_precision_over_one_ulp": sum(
            record.get("elements_over_one_ulp", 0) for record in changed
        ),
        "largest_ulp_errors": sorted(
            (
                record
                for record in changed
                if "maximum_ulp_error" in record
            ),
            key=lambda record: record["maximum_ulp_error"],
            reverse=True,
        )[:20],
        "largest_absolute_errors": sorted(
            changed,
            key=lambda record: record["maximum_absolute_error"],
            reverse=True,
        )[:20],
    }


def benchmark_checkpointing(root: Path, calls=12, warmup=1, measured=5):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    state = torch.load(
        root / manifest["baseline_state"], map_location="cpu", weights_only=True
    )
    device = torch.device("cuda:0")
    selected = tuple(
        workload
        for workload in WORKLOADS
        if workload.name in {"single_1536", "quad_ragged_333"}
    )
    results = {}
    for workload in selected:
        base_inputs, target, weights, mask = _workload_tensors(workload, device)
        base_inputs = base_inputs.detach()
        for enabled in (False, True):
            model = _build_model(device)
            model.load_state_dict(state, strict=True)
            model.checkpoint_updateformer = enabled
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=5e-5, weight_decay=1e-5
            )

            def step():
                inputs = [
                    base_inputs.clone().requires_grad_(True) for _ in range(calls)
                ]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    losses = []
                    for value in inputs:
                        output = model(value, point_mask=mask)
                        losses.append(((output.float() - target) * weights).square().mean())
                    loss = torch.stack(losses).mean()
                loss.backward()
                optimizer.step()

            for _ in range(warmup):
                step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            durations = []
            for _ in range(measured):
                torch.cuda.synchronize()
                started = time.perf_counter()
                step()
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - started)
            results[f"{workload.name}/checkpoint_{enabled}"] = {
                "median_seconds": statistics.median(durations),
                "durations_seconds": durations,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
    return {
        "calls": calls,
        "warmup": warmup,
        "measured": measured,
        "results": results,
    }


def benchmark_checkpoint_capacity(root: Path, calls=12, warmup=1, measured=3):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    state = torch.load(
        root / manifest["baseline_state"], map_location="cpu", weights_only=True
    )
    device = torch.device("cuda:0")
    cases = [
        Workload(f"batch_{batch}_tracks_512", batch, 512, (512,) * batch, 3000 + batch)
        for batch in (1, 2, 4, 8, 16, 32)
    ] + [
        Workload(
            f"batch_{batch}_tracks_1536",
            batch,
            1536,
            (1536,) * batch,
            4000 + batch,
        )
        for batch in (1, 2, 4, 8, 16)
    ]
    results = {}
    for workload in cases:
        model = _build_model(device)
        model.load_state_dict(state, strict=True)
        model.checkpoint_updateformer = True
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=5e-5, weight_decay=1e-5
        )
        base_inputs, target, weights, mask = _workload_tensors(workload, device)
        base_inputs = base_inputs.detach()

        def step():
            inputs = [
                base_inputs.clone().requires_grad_(True) for _ in range(calls)
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                losses = []
                for value in inputs:
                    output = model(value, point_mask=mask)
                    losses.append(((output.float() - target) * weights).square().mean())
                loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()

        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        durations = []
        for _ in range(measured):
            torch.cuda.synchronize()
            started = time.perf_counter()
            step()
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
        median = statistics.median(durations)
        results[workload.name] = {
            "median_seconds": median,
            "scenes_per_second": workload.batch_size / median,
            "track_sequences_per_second": (
                workload.batch_size * workload.tracks / median
            ),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    return {
        "calls": calls,
        "warmup": warmup,
        "measured": measured,
        "results": results,
    }


def benchmark_cuda_graphs(root: Path, calls=12, warmup=3, measured=20):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    state = torch.load(
        root / manifest["baseline_state"], map_location="cpu", weights_only=True
    )
    device = torch.device("cuda:0")
    cases = (
        Workload("batch_1_tracks_512", 1, 512, (512,), 5101),
        Workload("batch_4_tracks_512", 4, 512, (512,) * 4, 5104),
        Workload("batch_1_tracks_1536", 1, 1536, (1536,), 5201),
        Workload("batch_2_tracks_1536", 2, 1536, (1536,) * 2, 5202),
    )
    results = {}
    for workload in cases:
        for mode in ("eager", "cuda_graph"):
            model = _build_model(device)
            model.load_state_dict(state, strict=True)
            model.checkpoint_updateformer = True
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=5e-5,
                weight_decay=1e-5,
                capturable=True,
            )
            value, target, weights, mask = _workload_tensors(workload, device)

            def step():
                optimizer.zero_grad(set_to_none=False)
                if value.grad is not None:
                    value.grad.zero_()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    losses = []
                    for _ in range(calls):
                        output = model(value, point_mask=mask)
                        losses.append(
                            ((output.float() - target) * weights).square().mean()
                        )
                    loss = torch.stack(losses).mean()
                loss.backward()
                optimizer.step()

            for _ in range(warmup):
                step()
            torch.cuda.synchronize()
            replay = step
            if mode == "cuda_graph":
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    step()
                replay = graph.replay
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            durations = []
            for _ in range(measured):
                torch.cuda.synchronize()
                started = time.perf_counter()
                replay()
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - started)
            median = statistics.median(durations)
            results[f"{workload.name}/{mode}"] = {
                "median_seconds": median,
                "scenes_per_second": workload.batch_size / median,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
    return {
        "calls": calls,
        "warmup": warmup,
        "measured": measured,
        "results": results,
    }


def benchmark_graphed_callables(root: Path, calls=12, warmup=3, measured=20):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    state = torch.load(
        root / manifest["baseline_state"], map_location="cpu", weights_only=True
    )
    device = torch.device("cuda:0")
    cases = (
        Workload("batch_1_tracks_512", 1, 512, (512,), 6101),
        Workload("batch_4_tracks_512", 4, 512, (512,) * 4, 6104),
        Workload("batch_1_tracks_1536", 1, 1536, (1536,), 6201),
    )
    results = {}
    for workload in cases:
        value, target, weights, mask = _workload_tensors(workload, device)

        def build_step(graphed):
            model = _build_model(device)
            model.load_state_dict(state, strict=True)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=5e-5, weight_decay=1e-5
            )
            callables = None
            if graphed:
                class Slot(torch.nn.Module):
                    def __init__(self, core):
                        super().__init__()
                        self.core = core

                    def forward(self, input_tensor, point_mask):
                        return self.core(input_tensor, point_mask)

                slots = tuple(Slot(model) for _ in range(calls))
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, cache_enabled=False
                ):
                    callables = torch.cuda.make_graphed_callables(
                        slots,
                        tuple((value, mask) for _ in slots),
                        num_warmup_iters=3,
                    )

            def step():
                optimizer.zero_grad(set_to_none=True)
                if value.grad is not None:
                    value.grad = None
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, cache_enabled=False
                ):
                    outputs = []
                    for index in range(calls):
                        output = (
                            callables[index](value, mask)
                            if callables is not None
                            else model(value, point_mask=mask)
                        )
                        outputs.append(output.clone())
                    loss = torch.stack(
                        [
                            ((output.float() - target) * weights).square().mean()
                            for output in outputs
                        ]
                    ).mean()
                loss.backward()
                optimizer.step()
                return {
                    "outputs": [output.detach().cpu() for output in outputs],
                    "input_gradient": value.grad.detach().cpu(),
                    "parameters": {
                        name: parameter.detach().cpu()
                        for name, parameter in model.named_parameters()
                    },
                }

            return step

        eager_step = build_step(False)
        graphed_step = build_step(True)
        eager_result = eager_step()
        graphed_result = graphed_step()
        mismatch = _close_mismatch(eager_result, graphed_result)
        if mismatch:
            raise RuntimeError(
                f"graphed callable changed {workload.name}: {mismatch}"
            )
        for name, step in (("eager", eager_step), ("graphed", graphed_step)):
            for _ in range(warmup):
                step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            durations = []
            for _ in range(measured):
                torch.cuda.synchronize()
                started = time.perf_counter()
                step()
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - started)
            median = statistics.median(durations)
            results[f"{workload.name}/{name}"] = {
                "median_seconds": median,
                "scenes_per_second": workload.batch_size / median,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
    return {
        "calls": calls,
        "warmup": warmup,
        "measured": measured,
        "results": results,
    }


class _ForwardGraphCheckpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, engine, input_tensor, point_mask, *parameters):
        ctx.engine = engine
        ctx.parameters = parameters
        ctx.save_for_backward(input_tensor, point_mask)
        engine.static_input.copy_(input_tensor)
        engine.static_mask.copy_(point_mask)
        engine.graph.replay()
        return engine.static_output.clone()

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, point_mask = ctx.saved_tensors
        recompute_input = input_tensor.detach().requires_grad_(True)
        with torch.enable_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, cache_enabled=False
        ):
            output = ctx.engine.model._forward_impl(recompute_input, point_mask)
        gradients = torch.autograd.grad(
            output,
            (recompute_input, *ctx.parameters),
            grad_output,
        )
        return (None, gradients[0], None, *gradients[1:])


class _ForwardGraphEngine:
    def __init__(self, model, sample_input, sample_mask):
        self.model = model
        self.static_input = torch.empty_like(sample_input)
        self.static_mask = torch.empty_like(sample_mask)
        self.static_input.copy_(sample_input)
        self.static_mask.copy_(sample_mask)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, cache_enabled=False
        ):
            for _ in range(3):
                model._forward_impl(self.static_input, self.static_mask)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, cache_enabled=False
        ):
            self.static_output = model._forward_impl(
                self.static_input, self.static_mask
            )

    def __call__(self, input_tensor, point_mask):
        return _ForwardGraphCheckpoint.apply(
            self,
            input_tensor,
            point_mask,
            *tuple(self.model.parameters()),
        )


def benchmark_forward_graph_checkpoint(root: Path, calls=12, warmup=3, measured=20):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    state = torch.load(
        root / manifest["baseline_state"], map_location="cpu", weights_only=True
    )
    device = torch.device("cuda:0")
    cases = (
        Workload("batch_1_tracks_512", 1, 512, (512,), 7101),
        Workload("batch_4_tracks_512", 4, 512, (512,) * 4, 7104),
        Workload("batch_1_tracks_1536", 1, 1536, (1536,), 7201),
    )
    results = {}
    for workload in cases:
        value, target, weights, mask = _workload_tensors(workload, device)

        def build_step(graphed):
            model = _build_model(device)
            model.load_state_dict(state, strict=True)
            model.checkpoint_updateformer = True
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=5e-5, weight_decay=1e-5
            )
            engine = _ForwardGraphEngine(model, value, mask) if graphed else None

            def step():
                optimizer.zero_grad(set_to_none=True)
                if value.grad is not None:
                    value.grad = None
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, cache_enabled=False
                ):
                    outputs = [
                        engine(value, mask)
                        if engine is not None
                        else model(value, point_mask=mask)
                        for _ in range(calls)
                    ]
                    loss = torch.stack(
                        [
                            ((output.float() - target) * weights).square().mean()
                            for output in outputs
                        ]
                    ).mean()
                loss.backward()
                optimizer.step()
                return {
                    "outputs": [output.detach().cpu() for output in outputs],
                    "input_gradient": value.grad.detach().cpu(),
                    "parameters": {
                        name: parameter.detach().cpu()
                        for name, parameter in model.named_parameters()
                    },
                }

            return step

        eager_step = build_step(False)
        graphed_step = build_step(True)
        mismatch = _close_mismatch(eager_step(), graphed_step())
        if mismatch:
            raise RuntimeError(
                f"forward graph changed {workload.name}: {mismatch}"
            )
        for name, step in (("eager", eager_step), ("forward_graph", graphed_step)):
            for _ in range(warmup):
                step()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            durations = []
            for _ in range(measured):
                torch.cuda.synchronize()
                started = time.perf_counter()
                step()
                torch.cuda.synchronize()
                durations.append(time.perf_counter() - started)
            median = statistics.median(durations)
            results[f"{workload.name}/{name}"] = {
                "median_seconds": median,
                "scenes_per_second": workload.batch_size / median,
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
    return {
        "calls": calls,
        "warmup": warmup,
        "measured": measured,
        "results": results,
    }
