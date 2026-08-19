"""Numerical diagnostics against the locked UpdateFormer golden tensors."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import time

import torch

from mvtracker.profiling.updateformer_contract import (
    WORKLOADS,
    _build_model,
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
    selected = (WORKLOADS[1], WORKLOADS[3])
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
