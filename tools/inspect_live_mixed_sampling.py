"""Inspect whole-step sampling through the exact mixed-training loader path."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch.distributed as dist
from lightning.fabric import Fabric
from lightning.fabric.strategies import DDPStrategy
from omegaconf import OmegaConf

from mvtracker.cli.train import (
    _ScheduledSourceSampler,
    _build_source_train_loader,
    _build_training_dataset,
    _load_mixed_step,
    _start_mixed_source_iterators,
)
from mvtracker.datasets.mixed_source_schedule import BalancedMixedSourceSchedule


SOURCE_PATTERN = ("diegesis", "mvkubric", "diegesis", "mvkubric")


def _load_config(args):
    repo_root = Path(__file__).resolve().parents[1]
    config = OmegaConf.merge(
        OmegaConf.load(repo_root / "configs/train.yaml"),
        OmegaConf.load(
            repo_root / "configs/experiment/diegesis_mvkubric_gt_ddp.yaml"
        ),
    )
    config.datasets.root = str(args.mvkubric_root)
    config.datasets.train.sources.diegesis.root = str(args.diegesis_root)
    config.datasets.train.sources.mvkubric.root = str(args.mvkubric_root)
    config.datasets.train.kubric_metadata_index_root = str(
        args.mvkubric_index_root
    )
    return config


def _row(step, microbatch, rank, source, attempt, batch):
    metadata = batch.sample_metadata[0]
    track_count = (
        int((~batch.track_padding_mask[0]).sum().item())
        if batch.track_padding_mask is not None
        else int(batch.trajectory.shape[-2])
    )
    return {
        "step": step,
        "microbatch": microbatch,
        "rank": rank,
        "source": source,
        "attempt": attempt,
        "virtual_index": int(metadata["virtual_index"]),
        "scene_index": int(metadata["scene_index"]),
        "scene": str(metadata["scene_name"]),
        "seed": int(metadata["seed"]),
        "window": [
            int(metadata["window_start"]),
            int(metadata["window_end_exclusive"]),
        ],
        "views": [int(view) for view in metadata["selected_views"]],
        "view_count": len(metadata["selected_views"]),
        "trajectories": track_count,
        "rgb_aug": bool(metadata["apply_rgb_aug"]),
        "depth_aug": bool(metadata["apply_depth_aug"]),
    }


def run(args):
    config = _load_config(args)
    print("starting two-rank Fabric live-loader inspection", flush=True)
    fabric = Fabric(
        accelerator="cuda",
        devices=2,
        strategy=DDPStrategy(find_unused_parameters=False),
    )
    fabric.launch()
    fabric.seed_everything(args.seed, workers=True)
    print(
        f"rank {fabric.global_rank}: Fabric ready on {fabric.device}",
        flush=True,
    )

    datasets = {
        source: _build_training_dataset(
            source_config.name,
            source_config.root,
            config,
            fabric,
            source_config,
        )
        for source, source_config in config.datasets.train.sources.items()
    }
    print(f"rank {fabric.global_rank}: datasets ready", flush=True)
    schedule = BalancedMixedSourceSchedule(
        {source: dataset.real_len for source, dataset in datasets.items()},
        SOURCE_PATTERN,
        world_size=fabric.world_size,
        master_seed=args.seed,
    )
    cursors = {source: 0 for source in datasets}
    samplers = {
        source: _ScheduledSourceSampler(
            schedule,
            source,
            fabric.global_rank,
            len(dataset),
        )
        for source, dataset in datasets.items()
    }
    loaders = {
        source: _build_source_train_loader(
            dataset, samplers[source], config, fabric
        )
        for source, dataset in datasets.items()
    }
    data_iters = _start_mixed_source_iterators(loaders)
    print(f"rank {fabric.global_rank}: live iterators ready", flush=True)

    local_rows = []
    step_seconds = []
    for step in range(args.steps):
        fabric.barrier()
        started = time.perf_counter()
        batches, _, _ = _load_mixed_step(
            fabric,
            SOURCE_PATTERN,
            data_iters,
            samplers,
            loaders,
            cursors,
        )
        for microbatch, (source, batch) in enumerate(batches):
            local_rows.append(
                _row(
                    step,
                    microbatch,
                    fabric.global_rank,
                    source,
                    int(batch.sample_metadata[0]["paired_retry_attempt"]),
                    batch,
                )
            )
        fabric.barrier()
        step_seconds.append(time.perf_counter() - started)
        print(
            f"rank {fabric.global_rank}: step {step + 1}/{args.steps} "
            f"loaded in {step_seconds[-1]:.3f}s",
            flush=True,
        )

    gathered_rows = [None] * fabric.world_size if fabric.global_rank == 0 else None
    gathered_seconds = [None] * fabric.world_size if fabric.global_rank == 0 else None
    dist.gather_object(local_rows, gathered_rows, dst=0)
    dist.gather_object(step_seconds, gathered_seconds, dst=0)
    print(f"rank {fabric.global_rank}: gather complete", flush=True)
    if fabric.global_rank != 0:
        return

    rows = sorted(
        (row for rank_rows in gathered_rows for row in rank_rows),
        key=lambda row: (row["step"], row["microbatch"], row["rank"]),
    )
    baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    comparison = {
        "identical": rows == baseline,
        "expected_samples": len(baseline),
        "actual_samples": len(rows),
    }
    timings = {
        "per_rank_step_seconds": gathered_seconds,
        "step_wall_seconds": [
            max(rank[step] for rank in gathered_seconds)
            for step in range(args.steps)
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "samples.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "timings.json").write_text(
        json.dumps(timings, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({**comparison, **timings}))
    if not comparison["identical"]:
        raise RuntimeError("live mixed loader output differs from sequential baseline")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diegesis-root", type=Path, required=True)
    parser.add_argument("--mvkubric-root", type=Path, required=True)
    parser.add_argument("--mvkubric-index-root", type=Path, required=True)
    parser.add_argument("--baseline-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=72)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    run(args)


if __name__ == "__main__":
    main()
