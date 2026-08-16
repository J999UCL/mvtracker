"""Measure archive-to-local-SSD staging and one CPU sample per dataset."""

from __future__ import annotations

from pathlib import Path
import time

import modal

from modal_training_profile import (
    DATA_ROOT,
    _runtime_image,
    _source_commit,
    data_volume,
    wandb_secret,
)
from mvtracker.profiling.modal_continual_training import (
    EPHEMERAL_DISK_MIB,
    PROFILE_TAGS,
    WANDB_ENTITY,
    WANDB_GROUP,
    WANDB_PROJECT,
    preflight_active_containers,
    require_pushed_main_commit,
)


APP_NAME = "jeet-mvtracker-data-loading-smoke"
LOCAL_ROOT = Path("/tmp/mvtracker-data")

app = modal.App(
    APP_NAME,
    tags={**PROFILE_TAGS, "experiment": "archive-staging-cpu", "gpu": "cpu"},
)
image = _runtime_image().apt_install("zstd")


def _stage(emit) -> dict:
    from mvtracker.profiling.modal_continual_data import stage_continual_training_data

    emit("staging_started")
    staging = stage_continual_training_data(DATA_ROOT, local_data_root=LOCAL_ROOT)
    metrics = {"total_staging_seconds": staging["elapsed_seconds"]}
    metrics["diegesis_scene_count"] = len(
        list((LOCAL_ROOT / "source/diegesis/scenes").iterdir())
    )
    metrics["mvkubric_scene_count"] = len(
        [
            path
            for path in (LOCAL_ROOT / "datasets/kubric-multiview/train").iterdir()
            if path.is_dir() and path.name.isdigit()
        ]
    )
    emit(
        "staging_complete",
        {
            "seconds": metrics["total_staging_seconds"],
            "diegesis_scene_count": metrics["diegesis_scene_count"],
            "mvkubric_scene_count": metrics["mvkubric_scene_count"],
        },
    )
    return metrics


def _load_one_sample(source: str, emit) -> dict:
    from mvtracker.profiling.modal_continual_data import profile_encoded_loader

    emit(f"{source}_loader_started")
    started = time.perf_counter()
    result = profile_encoded_loader(
        LOCAL_ROOT / "datasets",
        source=source,
        warmup=0,
        measured=1,
        workers=0,
        use_cuda=False,
    )
    total = time.perf_counter() - started
    metrics = {
        "total_seconds": total,
        "first_sample_seconds": result["elapsed_seconds"],
        "initialization_seconds": total - result["elapsed_seconds"],
        "encoded_frames": result["encoded_frames"],
    }
    emit(f"{source}_loader_complete", metrics)
    return metrics


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume.with_mount_options(read_only=True)},
    cpu=16,
    memory=32768,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    timeout=4 * 60 * 60,
    max_containers=1,
    include_source=False,
)
def run_smoke() -> dict:
    import wandb

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        job_type="data-loading-smoke",
        tags=["modal", "cpu", "archive-staging"],
        config={"source_commit": _source_commit(), **PROFILE_TAGS},
    )
    smoke_started = time.perf_counter()

    def emit(event: str, metrics: dict | None = None) -> None:
        values = metrics or {}
        elapsed = time.perf_counter() - smoke_started
        print(
            json.dumps(
                {"event": event, "elapsed_seconds": elapsed, **values},
                sort_keys=True,
            ),
            flush=True,
        )
        wandb_values = {f"events/{event}/elapsed_seconds": elapsed}
        wandb_values.update(
            {
                f"events/{event}/{key}": value
                for key, value in values.items()
                if isinstance(value, (int, float))
            }
        )
        run.log(wandb_values)

    try:
        result = {
            "staging": _stage(emit),
            "diegesis_loader": _load_one_sample("diegesis", emit),
            "mvkubric_loader": _load_one_sample("mvkubric", emit),
        }
        emit("smoke_complete", {"seconds": time.perf_counter() - smoke_started})
        run.summary.update(result)
        return result
    except Exception as error:
        emit("smoke_failed")
        run.summary["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        run.finish()


@app.local_entrypoint()
def main() -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=1)
    print(json.dumps(run_smoke.remote(), indent=2))
