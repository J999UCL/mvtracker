#!/usr/bin/env python3
"""Benchmark selective TAPVid-3D I/O plus GPU nvJPEG decoding."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from mvtracker.datasets.tapvid3d_multiview_dataset import (
    CudaPrefetchLoader,
    TapVid3DMultiViewDataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--prefetch-factors", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--wandb-project", default="mvtracker-loader-benchmark")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return parser.parse_args()


def build_dataset(dataset_root: Path):
    repo = Path(__file__).resolve().parents[1]
    config = OmegaConf.merge(
        OmegaConf.load(repo / "configs/train.yaml"),
        OmegaConf.load(repo / "configs/experiment/tapvid3d_procedural.yaml"),
    )
    config.datasets.root = str(dataset_root)
    return TapVid3DMultiViewDataset.from_name(
        config.datasets.train.name,
        str(dataset_root),
        training_args=config,
        fabric=SimpleNamespace(world_size=1),
    )


def run_lane(dataset, workers: int, prefetch_factor: int, warmup: int, samples: int):
    loader = StatefulDataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        collate_fn=dataset.collate_fn,
        drop_last=True,
        in_order=False,
    )
    loader = CudaPrefetchLoader(loader)
    iterator = iter(loader)
    for _ in range(warmup):
        batch, gotit = next(iterator)
        if not all(gotit):
            raise RuntimeError("warmup produced an invalid sample")
        torch.cuda.synchronize()
        del batch
    started = time.perf_counter()
    for _ in range(samples):
        batch, gotit = next(iterator)
        if not all(gotit):
            raise RuntimeError("measurement produced an invalid sample")
        del batch
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "workers": workers,
        "prefetch_factor": prefetch_factor,
        "samples": samples,
        "elapsed_seconds": elapsed,
        "samples_per_second": samples / elapsed,
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark requires CUDA nvJPEG")
    dataset = build_dataset(args.dataset_root.resolve())
    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            mode=args.wandb_mode,
            job_type="loader-benchmark",
            config={
                "workers": args.workers,
                "prefetch_factors": args.prefetch_factors,
                "warmup": args.warmup,
                "samples": args.samples,
            },
        )
    lanes = []
    for workers in args.workers:
        for prefetch_factor in args.prefetch_factors:
            lane = run_lane(dataset, workers, prefetch_factor, args.warmup, args.samples)
            lanes.append(lane)
            print(json.dumps(lane, sort_keys=True), flush=True)
            if wandb_run is not None:
                prefix = f"workers_{workers}/prefetch_{prefetch_factor}"
                wandb_run.log({f"{prefix}/samples_per_second": lane["samples_per_second"]})
    summary = {
        "lanes": lanes,
        "best": max(lanes, key=lambda lane: lane["samples_per_second"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if wandb_run is not None:
        wandb_run.log({"best/samples_per_second": summary["best"]["samples_per_second"]})
        wandb_run.finish()


if __name__ == "__main__":
    main()
