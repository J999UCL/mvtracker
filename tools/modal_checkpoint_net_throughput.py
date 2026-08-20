"""Matched full-model H200 throughput comparison for UpdateFormer checkpointing."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import modal

from modal_training_profile import (
    DATA_ROOT,
    RUN_ROOT,
    _dependency_image,
    _source_commit,
    _source_image,
    data_volume,
    run_volume,
    wandb_secret,
)
from mvtracker.profiling.modal_continual_training import (
    preflight_active_containers,
    require_pushed_main_commit,
)


APP_NAME = "jeet-mvtracker-checkpoint-throughput-h200"
CACHE_ROOT = DATA_ROOT / "checkpoint-throughput-h200-batches"
TAGS = {"owner": "jeet", "project": "mvtracker", "purpose": "profiling"}
VIEWS = (1, 4)
TRAJECTORIES = 1024
GPU_LANE = "H200"
CACHE_BATCH_SIZE = 16
BASELINE = {
    1: {"batch_size": 5, "scenes_per_second": 3.300549786440229},
    4: {"batch_size": 3, "scenes_per_second": 1.9327912147330826},
}

app = modal.App(APP_NAME, tags={**TAGS, "experiment": "checkpoint-net-throughput"})
image = _source_image(_dependency_image())


def _cache_path(views: int) -> Path:
    return CACHE_ROOT / f"views{views}-traj{TRAJECTORIES}-batch{CACHE_BATCH_SIZE}.pt"


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=8,
    memory=65536,
    timeout=4 * 60 * 60,
    max_containers=1,
    include_source=False,
)
def prepare_batches_remote() -> dict:
    import dataclasses

    import torch
    import wandb

    from mvtracker.cli.profile_training import prepare_profile_batch

    run = wandb.init(
        entity="jeetucl-ucl",
        project="mvtracker-modal-profiling",
        group="updateformer-autoresearch-v3",
        job_type="checkpoint-throughput-batch-preparation",
        tags=["modal", "cpu", "checkpoint-throughput"],
        config={"source_commit": _source_commit(), **TAGS},
    )
    results = []
    for views in VIEWS:
        print(
            f"preparing {views}-view/{TRAJECTORIES}-track "
            f"B{CACHE_BATCH_SIZE} cache",
            flush=True,
        )
        source_path = CACHE_ROOT / f"views{views}-traj{TRAJECTORIES}-batch8.source.pt"
        result = prepare_profile_batch(
            data_root=DATA_ROOT / "datasets",
            output=source_path,
            views=views,
            batch_size=8,
            trajectories=TRAJECTORIES,
        )
        batch = torch.load(source_path, map_location="cpu", weights_only=False)
        for field in dataclasses.fields(batch):
            value = getattr(batch, field.name)
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                setattr(batch, field.name, torch.cat((value, value), dim=0))
            elif isinstance(value, list) and len(value) == 8:
                setattr(batch, field.name, value + value)
        target = _cache_path(views)
        torch.save(batch, target)
        source_path.unlink()
        result.update(
            path=str(target),
            batch_size=CACHE_BATCH_SIZE,
            bytes=target.stat().st_size,
        )
        results.append(result)
        data_volume.commit()
        print(f"prepared {views}-view cache: {result['bytes'] / 2**30:.2f} GiB", flush=True)
    run.summary.update({f"cache/views{result['views']}_gib": result["bytes"] / 2**30 for result in results})
    run.finish()
    return {"batches": results}


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
        str(RUN_ROOT): run_volume,
    },
    gpu=GPU_LANE,
    cpu=8,
    memory=32768,
    timeout=45 * 60,
    max_containers=1,
    include_source=False,
)
def sweep_remote(run_name: str) -> dict:
    import wandb

    output_root = RUN_ROOT / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        entity="jeetucl-ucl",
        project="mvtracker-modal-profiling",
        group="updateformer-autoresearch-v3",
        job_type="checkpoint-net-throughput",
        tags=["modal", "h200", "single-gpu", "full-model", "checkpoint"],
        config={
            "source_commit": _source_commit(),
            "views": list(VIEWS),
            "trajectories": TRAJECTORIES,
            "gpu_lane": GPU_LANE,
            **TAGS,
        },
    )
    trials = []
    for views in VIEWS:
        for batch_size in range(1, CACHE_BATCH_SIZE + 1):
            output = output_root / f"views{views}-batch{batch_size}.json"
            command = [
                sys.executable,
                "-m",
                "mvtracker.cli.profile_training",
                "--data-root",
                str(DATA_ROOT / "datasets"),
                "--checkpoint",
                str(DATA_ROOT / "checkpoints/mvtracker_200000_june2025_cleandepth.pth"),
                "--output",
                str(output),
                "--batch-cache",
                str(_cache_path(views)),
                "--views",
                str(views),
                "--batch-size",
                str(batch_size),
                "--accumulation",
                "1",
                "--trajectories",
                str(TRAJECTORIES),
                "--warmup-updates",
                "2",
                "--measure-updates",
                "3",
                "--workers",
                "0",
                "--gpu-lane",
                GPU_LANE,
                "--checkpoint-updateformer",
            ]
            completed = subprocess.run(command, cwd="/opt/mvtracker", check=False)
            if completed.returncode != 0:
                raise RuntimeError(f"profile trial failed: views={views} batch={batch_size}")
            trial = json.loads(output.read_text(encoding="utf-8"))
            trials.append(trial)
            run.log(
                {
                    "trial/views": views,
                    "trial/batch_size": batch_size,
                    "trial/safe": int(trial["status"] == "safe"),
                    "trial/scenes_per_second": trial.get("scenes_per_second", 0.0),
                    "trial/peak_memory_fraction": trial.get("peak_memory_fraction", 1.0),
                },
                step=len(trials),
            )
            print(
                f"views={views} batch={batch_size} status={trial['status']} "
                f"scenes/s={trial.get('scenes_per_second', 0.0):.3f}",
                flush=True,
            )
    comparisons = {}
    for views in VIEWS:
        safe = [
            trial
            for trial in trials
            if trial["views"] == views and trial["status"] == "safe"
        ]
        best = max(safe, key=lambda trial: trial["scenes_per_second"])
        baseline = BASELINE[views]
        comparisons[str(views)] = {
            "eager": baseline,
            "checkpoint": {
                "batch_size": best["batch_size"],
                "scenes_per_second": best["scenes_per_second"],
                "peak_memory_fraction": best["peak_memory_fraction"],
            },
            "throughput_ratio": best["scenes_per_second"] / baseline["scenes_per_second"],
        }
        run.summary[f"comparison/views{views}/throughput_ratio"] = comparisons[str(views)]["throughput_ratio"]
    result = {
        "source_commit": _source_commit(),
        "comparisons": comparisons,
        "trials": trials,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    run_volume.commit()
    run.summary["result_path"] = str(summary_path)
    run.finish()
    return {"result_path": str(summary_path), "comparisons": comparisons}


@app.local_entrypoint(name="prepare")
def prepare() -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    app.set_tags({**TAGS, "experiment": "checkpoint-throughput-preparation", "gpu": "cpu"})
    print(json.dumps(prepare_batches_remote.remote(), indent=2))


@app.local_entrypoint(name="sweep")
def sweep(run_name: str = "") -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=1)
    selected = run_name or f"checkpoint-net-throughput-h200-{commit[:8]}"
    app.set_tags({**TAGS, "experiment": selected, "gpu": "h200"})
    print(json.dumps(sweep_remote.remote(selected), indent=2))
