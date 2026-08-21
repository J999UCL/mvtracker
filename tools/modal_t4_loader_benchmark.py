"""Single-T4 production loader benchmark reading Modal Volume v2."""

from __future__ import annotations

from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path

import modal

from modal_continual_training import preflight_active_containers, training_image
from modal_training_profile import (
    BASE_TAGS,
    DATA_ROOT,
    RUN_ROOT,
    _RUN_NAME,
    _source_commit,
    data_volume,
    run_volume,
    wandb_secret,
)
from mvtracker.profiling.modal_continual_training import (
    DATA_LAYOUT_VERSION,
    DATA_VOLUME_ROOT,
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
image = training_image


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
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
        str(RUN_ROOT): run_volume,
    },
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
    measured: int = 16,
    source: str = "matrix",
    view_count: int = 4,
) -> dict:
    import wandb
    from mvtracker.profiling.modal_continual_data import profile_encoded_loader

    if _RUN_NAME.fullmatch(run_name) is None:
        raise ValueError("run name contains unsupported characters")
    if warmup < 0 or measured <= 0:
        raise ValueError("warmup must be non-negative and measured must be positive")
    if source not in {"matrix", "syn4d"}:
        raise ValueError("source must be matrix or syn4d")
    if not 1 <= view_count <= 6:
        raise ValueError("view_count must be between one and six")

    container_monitor = ContainerHardwareMonitor()
    gpu_monitor = GpuHardwareMonitor()
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        job_type="t4-loader-benchmark",
        tags=["modal", "t4", "loader", "warm-only", "production-path"],
        config={
            "source_commit": _source_commit(),
            "gpu": T4_GPU_REQUEST,
            "workers": T4_WORKERS,
            "source": source,
            "view_count": view_count,
            "simulated_compute_seconds": SIMULATED_COMPUTE_SECONDS,
            **MODAL_TAGS,
        },
    )
    output_dir = RUN_ROOT / RUN_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"{run_name}.json"
    progress = {
        "format": "mvtracker_t4_loader_benchmark_progress",
        "source_commit": _source_commit(),
        "status": "running",
        "current_case": None,
        "completed": {},
    }

    def write_progress():
        artifact_path.write_text(
            json.dumps(progress, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        run_volume.commit()

    try:
        def sample_hardware():
            return {
                "cpu_ram": container_monitor.sample(),
                "gpu": gpu_monitor.sample(),
            }

        profile_loader = partial(
            profile_encoded_loader,
            Path(DATA_VOLUME_ROOT),
        )

        case_total = 1 if source == "syn4d" else 4

        def report_progress(event, case_name, result):
            progress["current_case"] = case_name if event == "started" else None
            if result is not None:
                progress["completed"][case_name] = result
            print(
                f"PROFILE_PROGRESS event={event} case={case_name} "
                f"completed={len(progress['completed'])}/{case_total}",
                flush=True,
            )
            metrics = {
                "progress/cases_completed": len(progress["completed"]),
                f"progress/{case_name}/completed": int(event == "completed"),
            }
            if result is not None:
                metrics.update(_flatten_scalars(result, f"profiles/{case_name}"))
            run.log(metrics)
            write_progress()

        if source == "syn4d":
            case_name = f"syn4d-views{view_count}"
            report_progress("started", case_name, None)
            result = profile_loader(
                source="syn4d",
                view_count=view_count,
                warmup=warmup,
                measured=measured,
                workers=T4_WORKERS,
                use_cuda=True,
                simulated_compute_seconds=0.0,
                hardware_sampler=sample_hardware,
            )
            report_progress("completed", case_name, result)
            profiles = {"case": case_name, "profile": result}
        else:
            profiles = run_case_matrix(
                profile_loader,
                warmup=warmup,
                measured=measured,
                workers=T4_WORKERS,
                simulated_compute_seconds=SIMULATED_COMPUTE_SECONDS,
                hardware_sampler=sample_hardware,
                progress_callback=report_progress,
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
            "data_volume_root": DATA_VOLUME_ROOT,
            "data_layout_version": DATA_LAYOUT_VERSION,
            "profiles": profiles,
            "hardware": hardware,
        }
        progress["status"] = "completed"
        progress["current_case"] = None
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
def profile(run_name: str = "", warmup: int = 4, measured: int = 16) -> None:
    commit = _source_commit()
    preflight_active_containers(required_free_slots=1)
    selected = run_name or _default_run_name(commit)
    if _RUN_NAME.fullmatch(selected) is None:
        raise ValueError("run name contains unsupported characters")
    app.set_tags({**MODAL_TAGS, "experiment": selected, "gpu": "t4"})
    print(json.dumps(profile_t4_loader_remote.remote(selected, warmup=warmup, measured=measured), indent=2))


@app.local_entrypoint(name="syn4d")
def profile_syn4d(
    run_name: str = "",
    warmup: int = 1,
    measured: int = 5,
    view_count: int = 4,
) -> None:
    commit = _source_commit()
    preflight_active_containers(required_free_slots=1)
    selected = run_name or _default_run_name(commit)
    if _RUN_NAME.fullmatch(selected) is None:
        raise ValueError("run name contains unsupported characters")
    app.set_tags({**MODAL_TAGS, "experiment": selected, "gpu": "t4", "source": "syn4d"})
    result = profile_t4_loader_remote.remote(
        selected,
        warmup=warmup,
        measured=measured,
        source="syn4d",
        view_count=view_count,
    )
    print(json.dumps(result, indent=2))
