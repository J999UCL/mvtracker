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


def _sample_row(step, microbatch, rank, scheduled, sample, attempt):
    metadata = sample.metadata
    return {
        "step": step,
        "microbatch": microbatch,
        "rank": rank,
        "source": scheduled.source,
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
    }


def collect_whole_step_samples(schedule, datasets, steps: int) -> list[dict]:
    """Select each optimizer step's eight requests together, then preserve retries."""
    cursors = {source: 0 for source in datasets}
    rows = []
    for step in range(steps):
        selected = schedule.sample_step(cursors)
        queues = {source: [] for source in datasets}
        for item in selected:
            sample, gotit = datasets[item.source][item.request]
            scheduled = SimpleNamespace(source=item.source, request=item.request)
            queues[item.source].append(
                (scheduled, sample, bool(gotit))
            )

        next_cursors = {
            source: cursors[source] + SOURCE_PATTERN.count(source)
            for source in datasets
        }
        for microbatch, source in enumerate(SOURCE_PATTERN):
            attempt = 0
            while True:
                pair = queues[source][:schedule.world_size]
                del queues[source][:schedule.world_size]
                if len(pair) < schedule.world_size:
                    cursor = next_cursors[source]
                    pair = []
                    for rank in range(schedule.world_size):
                        scheduled = schedule.sample_source(source, cursor, rank)
                        sample, gotit = datasets[source][scheduled.request]
                        pair.append((scheduled, sample, bool(gotit)))
                    next_cursors[source] += 1
                cursors[source] += 1
                if all(item[2] for item in pair):
                    break
                attempt += 1

            for rank, (scheduled, sample, _) in enumerate(pair):
                rows.append(
                    _sample_row(
                        step, microbatch, rank, scheduled, sample, attempt
                    )
                )
    return rows


def render_markdown(rows: list[dict]) -> str:
    lines = [
        "| Step | Micro | Rank | Source | Scene | Virtual | Seed | Window | Views | Tracks | RGB aug | Depth aug |",
        "|---:|---:|---:|---|---|---:|---:|---|---|---:|---|---|",
    ]
    for row in rows:
        values = dict(row)
        values["views"] = ",".join(str(view) for view in row["views"])
        lines.append(
            "| {step} | {microbatch} | {rank} | {source} | {scene} | "
            "{virtual_index} | {seed} | {window[0]}:{window[1]} | {views} | "
            "{trajectories} | {rgb_aug} | {depth_aug} |".format(
                **values,
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
        job_type=f"sampling-{args.selection_mode}",
        name=args.run_name,
        tags=["dopey", "sampling", args.selection_mode],
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
    parser.add_argument(
        "--selection-mode",
        choices=("sequential", "whole-step"),
        default="sequential",
    )
    parser.add_argument("--baseline-json", type=Path)
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
    collector = (
        collect_sequential_samples
        if args.selection_mode == "sequential"
        else collect_whole_step_samples
    )
    rows = collector(schedule, datasets, args.steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "samples.json"
    markdown_path = args.output_dir / "samples.md"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    markdown = render_markdown(rows)
    markdown_path.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    comparison = None
    if args.baseline_json is not None:
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        comparison = {
            "identical": rows == baseline,
            "expected_samples": len(baseline),
            "actual_samples": len(rows),
        }
        comparison_path = args.output_dir / "comparison.json"
        comparison_path.write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(comparison))
    wandb_url = _log_wandb(rows, args, json_path, markdown_path)
    print(json.dumps({
        "samples": len(rows),
        "json": str(json_path),
        "markdown": str(markdown_path),
        "comparison": comparison,
        "wandb": wandb_url,
    }))


if __name__ == "__main__":
    main()
