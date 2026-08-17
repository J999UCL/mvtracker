"""Single-T4 production loader benchmark using the cached dataset image."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import modal

from modal_continual_training import dataset_image
from modal_training_profile import (
    BASE_TAGS,
    RUN_ROOT,
    _RUN_NAME,
    _source_commit,
    _source_image,
    preflight_active_containers,
    run_volume,
    wandb_secret,
)
from mvtracker.profiling.modal_continual_data import profile_encoded_loader
from mvtracker.profiling.modal_continual_training import (
    DATASET_IMAGE_ROOT,
    DATASET_IMAGE_VERSION,
)
from mvtracker.profiling.t4_loader_benchmark import (
    ContainerHardwareMonitor,
    GpuHardwareMonitor,
    SIMULATED_COMPUTE_SECONDS,
    T4_GPU_REQUEST,
    T4_MAX_CONTAINERS,
    T4_WORKERS,
    run_case_matrix,
)


APP_NAME = "jeet-mvtracker-t4-loader-benchmark"
RUN_SUBDIR = "t4-loader-benchmark"
WANDB_PROJECT = "mvtracker-modal-profiling"
WANDB_ENTITY = "jeetucl-ucl"
MODAL_TAGS = {**BASE_TAGS, "experiment": "t4-loader-benchmark", "gpu": "t4"}

app = modal.App(APP_NAME, tags=MODAL_TAGS)
image = _source_image(dataset_image)


def _flatten_scalars(value, prefix=""):
    if isinstance(value, bool):
        return {}
    if isinstance(value, (int, float)):
        return {prefix: value}
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(_flatten_scalars(item, f"{prefix}/{key}" if prefix else str(key)))
        return result
    return {}


def _default_run_name(commit: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"t4-loader-{commit[:8]}-{timestamp}"


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={str(RUN_ROOT): run_volume},
    gpu=T4_GPU_REQUEST,
    cpu=16,
    memory=32768,
    timeout=60 * 60,
    max_containers=T4_MAX_CONTAINERS,
    include_source=False,
)
def profile_t4_loader_remote(
    run_name: str,
    *,
    warmup: int = 4,
    measured: int = 32,
) -> dict:
    import wandb

    if _RUN_NAME.fullmatch(run_name) is None:
        raise ValueError("run name contains unsupported characters")
    if warmup < 0 or measured <= 0:
        raise ValueError("warmup must be non-negative and measured must be positive")

    container_monitor = ContainerHardwareMonitor()
    gpu_monitor = GpuHardwareMonitor()
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        job_type="t4-loader-benchmark",
        tags=["modal", "t4", "loader", "cold-warm", "production-path"],
        config={
            "source_commit": _source_commit(),
            "gpu": T4_GPU_REQUEST,
            "workers": T4_WORKERS,
            "simulated_compute_seconds": SIMULATED_COMPUTE_SECONDS,
            **MODAL_TAGS,
        },
    )
    try:
        def sample_hardware():
            return {
                "cpu_ram": container_monitor.sample(),
                "gpu": gpu_monitor.sample(),
            }

        profiles = run_case_matrix(
            profile_encoded_loader,
            warmup=warmup,
            measured=measured,
            workers=T4_WORKERS,
            simulated_compute_seconds=SIMULATED_COMPUTE_SECONDS,
            hardware_sampler=sample_hardware,
        )
        hardware = {
            "cpu_ram": container_monitor.sample(),
            "gpu": gpu_monitor.sample(),
        }
        artifact = {
            "format": "mvtracker_t4_loader_benchmark",
            "source_commit": _source_commit(),
            "gpu": T4_GPU_REQUEST,
            "modal_tags": MODAL_TAGS,
            "dataset_image_root": DATASET_IMAGE_ROOT,
            "dataset_image_version": DATASET_IMAGE_VERSION,
            "profiles": profiles,
            "hardware": hardware,
        }
        output_dir = RUN_ROOT / RUN_SUBDIR
        output_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = output_dir / f"{run_name}.json"
        artifact_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        run.summary.update(_flatten_scalars({"hardware": hardware, "profiles": profiles}))
        run.summary["artifact_path"] = str(artifact_path)
        run.finish()
        run_volume.commit()
        return {"artifact_path": str(artifact_path), **artifact}
    finally:
        gpu_monitor.close()


@app.local_entrypoint(name="profile")
def profile(run_name: str = "", warmup: int = 4, measured: int = 32) -> None:
    commit = _source_commit()
    preflight_active_containers(required_free_slots=1)
    selected = run_name or _default_run_name(commit)
    if _RUN_NAME.fullmatch(selected) is None:
        raise ValueError("run name contains unsupported characters")
    app.set_tags({**MODAL_TAGS, "experiment": selected, "gpu": "t4"})
    print(json.dumps(profile_t4_loader_remote.remote(selected, warmup=warmup, measured=measured), indent=2))
