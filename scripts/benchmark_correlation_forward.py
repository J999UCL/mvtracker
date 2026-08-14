#!/usr/bin/env python3
"""Benchmark the production indexed-correlation forward on realistic shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess

import torch


def parse_shape(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in value.split(","))
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("shape must be B,N,M with positive integers")
    return parts


def git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def benchmark_shape(
    operator,
    shape: tuple[int, int, int],
    *,
    channels: int,
    neighbors: int,
    groups: int,
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict:
    batch, source_points, queries = shape
    torch.manual_seed(42)
    targets = torch.randn(batch, queries, channels, device="cuda", dtype=dtype)
    source = torch.randn(batch, source_points, channels, device="cuda", dtype=dtype)
    indices = torch.randint(
        source_points,
        (batch, queries, neighbors),
        device="cuda",
        dtype=torch.int64,
    )

    with torch.inference_mode():
        for _ in range(warmup):
            output = operator(targets, source, indices, groups)
        torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        baseline_bytes = torch.cuda.memory_allocated()
        output = operator(targets, source, indices, groups)
        torch.cuda.synchronize()
        incremental_peak_bytes = torch.cuda.max_memory_allocated() - baseline_bytes
        output_bytes = output.numel() * output.element_size()
        del output

        timings_ms = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                output = operator(targets, source, indices, groups)
            end.record()
            end.synchronize()
            timings_ms.append(start.elapsed_time(end) / iterations)
        del output

    return {
        "batch_frames": batch,
        "source_points": source_points,
        "queries": queries,
        "channels": channels,
        "neighbors": neighbors,
        "groups": groups,
        "median_ms": statistics.median(timings_ms),
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
        "repeats_ms": timings_ms,
        "incremental_peak_allocated_bytes": incremental_peak_bytes,
        "output_bytes": output_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--shape", action="append", type=parse_shape, required=True)
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    from mvtracker.models.core.mvtracker import indexed_correlation

    implementation_root = Path(indexed_correlation.__file__).resolve().parents[4]
    dtype = getattr(torch, args.dtype)
    results = [
        benchmark_shape(
            indexed_correlation.indexed_grouped_correlation,
            shape,
            channels=args.channels,
            neighbors=args.neighbors,
            groups=args.groups,
            dtype=dtype,
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        for shape in args.shape
    ]
    print(
        json.dumps(
            {
                "label": args.label,
                "implementation_root": str(implementation_root),
                "git_revision": git_revision(implementation_root),
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "dtype": args.dtype,
                "warmup": args.warmup,
                "iterations": args.iterations,
                "repeats": args.repeats,
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
