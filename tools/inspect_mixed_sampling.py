"""Inspect the current sequential mixed-source training sampler."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace


SOURCE_PATTERN = ("diegesis", "mvkubric", "diegesis", "mvkubric")


def collect_sequential_samples(schedule, datasets, steps: int) -> list[dict]:
    """Run the existing rank-local source cursors exactly as training does."""
    cursors = {
        rank: {source: 0 for source in datasets}
        for rank in range(schedule.world_size)
    }
    rows = []
    for step in range(steps):
        for microbatch, source in enumerate(SOURCE_PATTERN):
            attempt = 0
            while True:
                pair = []
                for rank in range(schedule.world_size):
                    cursor = cursors[rank][source]
                    scheduled = schedule.sample_source(source, cursor, rank)
                    sample, gotit = datasets[source][scheduled.request]
                    cursors[rank][source] += 1
                    pair.append((rank, scheduled, sample, bool(gotit)))
                if all(item[3] for item in pair):
                    break
                attempt += 1

            for rank, scheduled, sample, _ in pair:
                metadata = sample.metadata
                rows.append({
                    "step": step,
                    "microbatch": microbatch,
                    "rank": rank,
                    "source": source,
                    "attempt": attempt,
                    "virtual_index": int(scheduled.request.virtual_index),
                    "scene_index": int(scheduled.request.scene_index),
                    "scene": str(metadata["scene_name"]),
                    "seed": int(metadata["seed"]),
                    "window": [
                        int(metadata["window_start"]),
                        int(metadata["window_end_exclusive"]),
                    ],
                    "views": [int(view) for view in metadata["selected_views"]],
                    "view_count": len(metadata["selected_views"]),
                    "trajectories": int(sample.trajectory.shape[-2]),
                    "rgb_aug": bool(sample.apply_rgb_aug),
                    "depth_aug": bool(sample.apply_depth_aug),
                })
    return rows


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "| Step | Micro | Rank | Source | Scene | Virtual | Seed | Window | Views | Tracks | RGB aug | Depth aug |",
        "|---:|---:|---:|---|---|---:|---:|---|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {step} | {microbatch} | {rank} | {source} | {scene} | "
            "{virtual_index} | {seed} | {window[0]}:{window[1]} | {views} | "
            "{trajectories} | {rgb_aug} | {depth_aug} |".format(
                **row,
                views=",".join(str(view) for view in row["views"]),
            )
        )
    return "\n".join(lines) + "\n"


def _build_datasets(args):
    from omegaconf import OmegaConf

    from mvtracker.datasets.kubric_multiview_dataset import KubricMultiViewDataset
    from mvtracker.datasets.tapvid3d_multiview_dataset import TapVid3DMultiViewDataset

    repo_root = Path(__file__).resolve().parents[1]
    config = OmegaConf.merge(
        OmegaConf.load(repo_root / "configs/train.yaml"),
        OmegaConf.load(repo_root / "configs/experiment/diegesis_mvkubric_gt_ddp.yaml"),
    )
    config.datasets.root = str(args.mvkubric_root)
    config.datasets.train.kubric_metadata_index_root = str(args.mvkubric_index_root)
    fabric = SimpleNamespace(world_size=2)
    diegesis_source = config.datasets.train.sources.diegesis
    mvkubric_source = config.datasets.train.sources.mvkubric
    return {
        "diegesis": TapVid3DMultiViewDataset.from_name(
            diegesis_source.name,
            str(args.diegesis_root),
            training_args=config,
            fabric=fabric,
            include_scene_ids=list(diegesis_source.include_scene_ids),
        ),
        "mvkubric": KubricMultiViewDataset.from_name(
            mvkubric_source.name,
            str(args.mvkubric_root),
            training_args=config,
            fabric=fabric,
            include_scene_ids=list(mvkubric_source.include_scene_ids),
        ),
    }


def _log_wandb(rows, args, json_path, markdown_path):
    if args.wandb_project is None:
        return None
    import wandb

    run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        job_type="sampling-baseline",
        name=args.run_name,
        tags=["dopey", "sampling", "baseline", "sequential"],
        config={"steps": args.steps, "seed": args.seed, "world_size": 2},
    )
    run.log({
        "samples": wandb.Table(
            columns=list(rows[0]),
            data=[[row[column] for column in rows[0]] for row in rows],
        )
    })
    artifact = wandb.Artifact(args.run_name, type="sampling-baseline")
    artifact.add_file(str(json_path))
    artifact.add_file(str(markdown_path))
    run.log_artifact(artifact)
    url = run.url
    run.finish()
    return url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diegesis-root", type=Path, required=True)
    parser.add_argument("--mvkubric-root", type=Path, required=True)
    parser.add_argument("--mvkubric-index-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=72)
    parser.add_argument("--run-name", default="mixed-sampling-baseline")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")

    from mvtracker.datasets.mixed_source_schedule import BalancedMixedSourceSchedule

    datasets = _build_datasets(args)
    schedule = BalancedMixedSourceSchedule(
        {source: dataset.real_len for source, dataset in datasets.items()},
        SOURCE_PATTERN,
        world_size=2,
        master_seed=args.seed,
    )
    rows = collect_sequential_samples(schedule, datasets, args.steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "samples.json"
    markdown_path = args.output_dir / "samples.md"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(rows)
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    wandb_url = _log_wandb(rows, args, json_path, markdown_path)
    print(json.dumps({
        "samples": len(rows),
        "json": str(json_path),
        "markdown": str(markdown_path),
        "wandb": wandb_url,
    }))


if __name__ == "__main__":
    main()
