"""Numerical diagnostics against the locked UpdateFormer golden tensors."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from mvtracker.profiling.updateformer_contract import (
    WORKLOADS,
    _run_loss_contract,
    _run_updateformer_case,
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
