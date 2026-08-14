"""Independent correctness harness and microbenchmark for point-cloud correlation.

This file deliberately does not provide a production operator.  It contains an
eager reference implementation, an indexed implementation candidate, and a
benchmark wrapper around the current :class:`PointcloudCorrBlock`.  The KNN
indices are supplied by the caller so that correlation arithmetic can be
measured independently of the KNN backend.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def eager_oracle_correlation(
    targets: torch.Tensor,
    xyz: torch.Tensor,
    fvec: torch.Tensor,
    neighbor_indices: torch.Tensor,
    groups: int,
    add_neighbor_offset: bool,
    add_neighbor_xyz: bool,
) -> torch.Tensor:
    """Reference correlation using the same grouped einsum as the model."""
    batch_size, num_queries, channels = targets.shape
    _, num_points, _ = xyz.shape
    _, _, neighbors = neighbor_indices.shape
    assert fvec.shape == (batch_size, num_points, channels)
    assert channels % groups == 0

    batch_indices = torch.arange(batch_size, device=xyz.device)[:, None, None]
    neighbor_xyz = xyz[batch_indices, neighbor_indices]
    neighbor_fvec = fvec[batch_indices, neighbor_indices]
    targets_grouped = targets.view(batch_size, num_queries, groups, -1)
    neighbor_grouped = neighbor_fvec.view(
        batch_size, num_queries, neighbors, groups, -1
    )
    correlations = torch.einsum(
        "BMGc,BMKGc->BMKG", targets_grouped, neighbor_grouped
    )
    correlations = correlations / ((channels / groups) ** 0.5)

    output = correlations
    offsets = neighbor_xyz - xyz.new_zeros(())
    # The query coordinates are passed separately by the candidate wrapper;
    # this local expression is replaced by ``correlation_from_indices`` below.
    del offsets
    if add_neighbor_offset or add_neighbor_xyz:
        raise ValueError("use correlation_from_indices when coordinates are needed")
    return output


def correlation_from_indices(
    targets: torch.Tensor,
    query_xyz: torch.Tensor,
    xyz: torch.Tensor,
    fvec: torch.Tensor,
    neighbor_indices: torch.Tensor,
    groups: int,
    add_neighbor_offset: bool,
    add_neighbor_xyz: bool,
) -> torch.Tensor:
    """Eager oracle including spatial features, with explicit query positions."""
    batch_size, num_queries, channels = targets.shape
    _, num_points, _ = xyz.shape
    _, _, neighbors = neighbor_indices.shape
    assert query_xyz.shape == (batch_size, num_queries, 3)
    assert fvec.shape == (batch_size, num_points, channels)
    assert channels % groups == 0

    batch_indices = torch.arange(batch_size, device=xyz.device)[:, None, None]
    neighbor_xyz = xyz[batch_indices, neighbor_indices]
    neighbor_fvec = fvec[batch_indices, neighbor_indices]
    targets_grouped = targets.view(batch_size, num_queries, groups, -1)
    neighbor_grouped = neighbor_fvec.view(
        batch_size, num_queries, neighbors, groups, -1
    )
    correlations = torch.einsum(
        "BMGc,BMKGc->BMKG", targets_grouped, neighbor_grouped
    )
    output = correlations / ((channels / groups) ** 0.5)
    if add_neighbor_offset:
        output = torch.cat([output, neighbor_xyz - query_xyz[..., None, :]], dim=-1)
    if add_neighbor_xyz:
        output = torch.cat([output, neighbor_xyz], dim=-1)
    return output


def indexed_grouped_candidate(
    targets: torch.Tensor,
    query_xyz: torch.Tensor,
    xyz: torch.Tensor,
    fvec: torch.Tensor,
    neighbor_indices: torch.Tensor,
    groups: int,
    add_neighbor_offset: bool,
    add_neighbor_xyz: bool,
) -> torch.Tensor:
    """Candidate gather using one flattened batch index and grouped channels."""
    batch_size, num_queries, channels = targets.shape
    _, num_points, _ = xyz.shape
    _, _, neighbors = neighbor_indices.shape
    assert channels % groups == 0

    batch_offsets = (
        torch.arange(batch_size, device=neighbor_indices.device) * num_points
    )[:, None, None]
    flat_indices = (neighbor_indices + batch_offsets).reshape(-1)
    flat_xyz = xyz.reshape(batch_size * num_points, 3)
    flat_fvec = fvec.reshape(batch_size * num_points, channels)
    neighbor_xyz = flat_xyz.index_select(0, flat_indices).view(
        batch_size, num_queries, neighbors, 3
    )
    neighbor_fvec = flat_fvec.index_select(0, flat_indices).view(
        batch_size, num_queries, neighbors, channels
    )

    targets_grouped = targets.view(batch_size, num_queries, groups, -1)
    neighbor_grouped = neighbor_fvec.view(
        batch_size, num_queries, neighbors, groups, -1
    )
    correlations = torch.einsum(
        "BMGc,BMKGc->BMKG", targets_grouped, neighbor_grouped
    )
    output = correlations / ((channels / groups) ** 0.5)
    if add_neighbor_offset:
        output = torch.cat([output, neighbor_xyz - query_xyz[..., None, :]], dim=-1)
    if add_neighbor_xyz:
        output = torch.cat([output, neighbor_xyz], dim=-1)
    return output


def deliberately_bad_grouping(
    targets: torch.Tensor,
    query_xyz: torch.Tensor,
    xyz: torch.Tensor,
    fvec: torch.Tensor,
    neighbor_indices: torch.Tensor,
    groups: int,
    add_neighbor_offset: bool,
    add_neighbor_xyz: bool,
) -> torch.Tensor:
    """Intentional channel grouping bug used to prove the oracle is sensitive."""
    batch_size, num_queries, channels = targets.shape
    neighbors = neighbor_indices.shape[-1]
    batch_indices = torch.arange(batch_size, device=xyz.device)[:, None, None]
    neighbor_xyz = xyz[batch_indices, neighbor_indices]
    neighbor_fvec = fvec[batch_indices, neighbor_indices]
    # Wrong: group is moved after the per-group channel dimension.
    targets_grouped = targets.view(batch_size, num_queries, -1, groups)
    neighbor_grouped = neighbor_fvec.view(
        batch_size, num_queries, neighbors, -1, groups
    )
    output = torch.einsum("BMGc,BMKGc->BMKG", targets_grouped, neighbor_grouped)
    output = output / ((channels / groups) ** 0.5)
    if add_neighbor_offset:
        output = torch.cat([output, neighbor_xyz - query_xyz[..., None, :]], dim=-1)
    if add_neighbor_xyz:
        output = torch.cat([output, neighbor_xyz], dim=-1)
    return output


def deliberately_bad_batch_indexing(
    targets: torch.Tensor,
    query_xyz: torch.Tensor,
    xyz: torch.Tensor,
    fvec: torch.Tensor,
    neighbor_indices: torch.Tensor,
    groups: int,
    add_neighbor_offset: bool,
    add_neighbor_xyz: bool,
) -> torch.Tensor:
    """Intentional missing-batch-offset bug used by the harness self-test."""
    batch_size, num_queries, channels = targets.shape
    neighbors = neighbor_indices.shape[-1]
    flat_indices = neighbor_indices.reshape(-1)
    flat_xyz = xyz.reshape(batch_size * xyz.shape[1], 3)
    flat_fvec = fvec.reshape(batch_size * fvec.shape[1], channels)
    # Deliberately indexes every batch from the first batch's local indices.
    bad_indices = flat_indices.remainder(xyz.shape[1])
    neighbor_xyz = flat_xyz.index_select(0, bad_indices).view(
        batch_size, num_queries, neighbors, 3
    )
    neighbor_fvec = flat_fvec.index_select(0, bad_indices).view(
        batch_size, num_queries, neighbors, channels
    )
    targets_grouped = targets.view(batch_size, num_queries, groups, -1)
    neighbor_grouped = neighbor_fvec.view(
        batch_size, num_queries, neighbors, groups, -1
    )
    output = torch.einsum("BMGc,BMKGc->BMKG", targets_grouped, neighbor_grouped)
    output = output / ((channels / groups) ** 0.5)
    if add_neighbor_offset:
        output = torch.cat([output, neighbor_xyz - query_xyz[..., None, :]], dim=-1)
    if add_neighbor_xyz:
        output = torch.cat([output, neighbor_xyz], dim=-1)
    return output


def deliberately_bad_detached_backward(
    targets: torch.Tensor,
    query_xyz: torch.Tensor,
    xyz: torch.Tensor,
    fvec: torch.Tensor,
    neighbor_indices: torch.Tensor,
    groups: int,
    add_neighbor_offset: bool,
    add_neighbor_xyz: bool,
) -> torch.Tensor:
    """Intentional autograd bug: forward values match, source gradients do not."""
    return correlation_from_indices(
        targets, query_xyz, xyz, fvec.detach(), neighbor_indices,
        groups, add_neighbor_offset, add_neighbor_xyz,
    )


def mocked_knn(neighbor_indices: torch.Tensor) -> Callable:
    """Build a KNN stub returning fixed local indices and matching distances."""

    def _knn(k: int, xyz_ref: torch.Tensor, xyz_query: torch.Tensor):
        assert k == neighbor_indices.shape[-1]
        assert xyz_ref.shape[0] == neighbor_indices.shape[0]
        assert xyz_query.shape[:2] == neighbor_indices.shape[:2]
        indices = neighbor_indices.to(device=xyz_ref.device)
        batch_indices = torch.arange(xyz_ref.shape[0], device=xyz_ref.device)[:, None, None]
        selected = xyz_ref[batch_indices, indices]
        distances = (selected - xyz_query[..., None, :]).norm(dim=-1)
        return distances, indices

    return _knn


def load_current_pointcloud_corr_block():
    """Load only the current production class, without importing its dependencies."""
    from mvtracker.models.core.mvtracker.indexed_correlation import (
        indexed_grouped_correlation,
    )

    source_path = (
        __file__.replace("scripts/indexed_grouped_correlation_benchmark.py", "")
        + "mvtracker/models/core/mvtracker/mvtracker.py"
    )
    with open(source_path, encoding="utf-8") as source_file:
        tree = ast.parse(source_file.read(), filename=source_path)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "PointcloudCorrBlock"
    )
    namespace = {
        "Callable": Callable,
        "Optional": Optional,
        "logging": logging,
        "np": np,
        "os": os,
        "save_pointcloud_to_ply": lambda *args, **kwargs: None,
        "time_now": lambda: "test",
        "torch": torch,
        "knn": None,
        "indexed_grouped_correlation": indexed_grouped_correlation,
    }
    module = ast.Module(body=[class_node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), source_path, "exec"), namespace)
    return namespace["PointcloudCorrBlock"], namespace


class BenchmarkResult:
    def __init__(self, name: str, seconds_per_call: float, calls: int, peak_cuda_bytes):
        self.name = name
        self.forward_backward_seconds_per_call = seconds_per_call
        self.calls = calls
        self.incremental_peak_cuda_allocated_bytes = peak_cuda_bytes

    def as_dict(self):
        return {
            "name": self.name,
            "forward_backward_seconds_per_call": self.forward_backward_seconds_per_call,
            "calls": self.calls,
            "incremental_peak_cuda_allocated_bytes": self.incremental_peak_cuda_allocated_bytes,
        }


def _time_calls(fn, warmup: int, iterations: int, device: torch.device):
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        before = torch.cuda.memory_allocated(device)
    else:
        before = 0
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)
    else:
        peak = None
    return (time.perf_counter() - start) / iterations, (
        None if peak is None else peak - before
    )


def benchmark(
    device: torch.device,
    batch_size: int,
    num_points: int,
    num_queries: int,
    channels: int,
    groups: int,
    neighbors: int,
    warmup: int,
    iterations: int,
    dtype: torch.dtype,
    backend: str,
) -> list[BenchmarkResult]:
    """Benchmark eager materialization against the production fused block."""
    torch.manual_seed(7)
    xyz = torch.randn(batch_size, num_points, 3, device=device)
    fvec = torch.randn(
        batch_size, num_points, channels, device=device, dtype=dtype
    )
    targets = torch.randn(
        batch_size, num_queries, channels, device=device, dtype=dtype
    )
    query_xyz = torch.randn(batch_size, num_queries, 3, device=device)
    indices = torch.randint(num_points, (batch_size, num_queries, neighbors), device=device)

    block_class, namespace = load_current_pointcloud_corr_block()
    namespace["knn"] = mocked_knn(indices)
    block = block_class(
        neighbors, groups, xyz, fvec, True, True
    )
    for value in (targets, query_xyz, xyz, fvec):
        value.requires_grad_()

    def eager_reference():
        for value in (targets, query_xyz, xyz, fvec):
            value.grad = None
        correlation_from_indices(
            targets,
            query_xyz,
            xyz,
            fvec,
            indices,
            groups,
            True,
            True,
        ).sum().backward()

    def production_fused():
        for value in (targets, query_xyz, xyz, fvec):
            value.grad = None
        block.corr_sample(targets, query_xyz).sum().backward()

    results = []
    if backend in {"eager", "both"}:
        baseline_time, baseline_peak = _time_calls(
            eager_reference, warmup, iterations, device
        )
        results.append(
            BenchmarkResult(
                "eager_reference", baseline_time, iterations, baseline_peak
            )
        )
    if backend in {"production", "both"}:
        candidate_time, candidate_peak = _time_calls(
            production_fused, warmup, iterations, device
        )
        results.append(
            BenchmarkResult(
                "production_fused", candidate_time, iterations, candidate_peak
            )
        )
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--num-queries", type=int, default=256)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16"
    )
    parser.add_argument(
        "--backend", choices=("eager", "production", "both"), default="both"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results = benchmark(
        torch.device(args.device),
        args.batch_size,
        args.num_points,
        args.num_queries,
        args.channels,
        args.groups,
        args.neighbors,
        args.warmup,
        args.iterations,
        getattr(torch, args.dtype),
        args.backend,
    )
    print(json.dumps([result.as_dict() for result in results], indent=2))


if __name__ == "__main__":
    main()
