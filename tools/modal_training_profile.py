"""Common Modal runtime and single-GPU MVTracker capacity profiling."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone

import modal


APP_NAME = "jeet-mvtracker-profile"
DATA_VOLUME_NAME = "jeet-mvtracker-data-v2"
RUN_VOLUME_NAME = "jeet-mvtracker-runs-v2"
HF_SECRET_NAME = "jeet-mvtracker-huggingface"
WANDB_SECRET_NAME = "jeet-mvtracker-wandb"
GPU_LANES = ("H100!", "H200", "B200")
BASE_TAGS = {
    "owner": "jeet",
    "project": "mvtracker",
    "purpose": "profiling",
}
SOURCE_ROOT = Path("/opt/mvtracker")
DATA_ROOT = Path("/mnt/mvtracker-data")
RUN_ROOT = Path("/mnt/mvtracker-runs")
PROFILE_BATCH_ROOT = DATA_ROOT / "profile-batches"
_COMMIT = re.compile(r"[0-9a-f]{40}")
_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
REQUIREMENTS_FILE = Path(__file__).resolve().parents[1] / "requirements.full.txt"
CORRELATION_SOURCE_DIR = (
    Path(__file__).resolve().parents[1]
    / "mvtracker"
    / "models"
    / "core"
    / "mvtracker"
)


def _source_commit() -> str:
    commit = os.environ.get("MVTRACKER_MODAL_COMMIT", "")
    if _COMMIT.fullmatch(commit) is None:
        raise RuntimeError("MVTRACKER_MODAL_COMMIT must be one full lowercase Git commit")
    return commit


def _dependency_image() -> modal.Image:
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu22.04",
            add_python="3.10",
        )
        .env({"DALI_DISABLE_NVML": "1"})
        .apt_install(
            "build-essential",
            "ffmpeg",
            "git",
            "libgl1",
            "libglib2.0-0",
            "ninja-build",
            "python3-dev",
            "zstd",
        )
        .pip_install(
            "torch==2.7.1",
            "torchvision==0.22.1",
            "torchaudio==2.7.1",
            index_url="https://download.pytorch.org/whl/cu128",
        )
        .pip_install_from_requirements(str(REQUIREMENTS_FILE))
        .pip_install(
            "hf-xet",
            "nvidia-ml-py",
            # DALI 1.53 supports nvImageCodec 0.7 and ships against
            # libnvcomp 5.1. Keep this set coherent; nvImageCodec 0.9 and
            # libnvcomp 5.3 are incompatible with the DALI wheel.
            "nvidia-dali-cuda120==1.53.0",
            "nvidia-nvimgcodec-cu12[nvtiff]==0.7.0.11",
            "nvidia-libnvcomp-cu12==5.1.0.21",
            "wids==0.1.11",
            "rapidgzip==0.16.0",
            # Modal's mirror currently publishes cu120 through 2.3.6.
            "spconv-cu120==2.3.6",
        )
        .run_commands(
            "python -m pip install --no-build-isolation flash-attn==2.8.3.post1",
            "python -m pip install 'git+https://github.com/ethz-vlg/pointcept.git@2082918#subdirectory=libs/pointops'",
            env={
                "CC": "gcc",
                "CXX": "g++",
                "CUDA_HOME": "/usr/local/cuda",
                "TORCH_CUDA_ARCH_LIST": "8.6;9.0;10.0",
                "CUMM_CUDA_ARCH_LIST": "8.6;9.0;10.0",
                "MAX_JOBS": "8",
            },
        )
        .add_local_file(
            str(CORRELATION_SOURCE_DIR / "indexed_correlation_cuda.cpp"),
            "/opt/mvtracker-extension-src/indexed_correlation_cuda.cpp",
            copy=True,
        )
        .add_local_file(
            str(CORRELATION_SOURCE_DIR / "indexed_correlation_cuda.cu"),
            "/opt/mvtracker-extension-src/indexed_correlation_cuda.cu",
            copy=True,
        )
        .add_local_file(
            str(Path(__file__).resolve().parent / "build_indexed_correlation_extension.py"),
            "/opt/mvtracker-extension-src/build.py",
            copy=True,
        )
        .run_commands(
            "python /opt/mvtracker-extension-src/build.py "
            "--source-dir /opt/mvtracker-extension-src "
            "--build-directory /opt/mvtracker-extension",
            env={
                "CC": "gcc",
                "CXX": "g++",
                "CUDA_HOME": "/usr/local/cuda",
                "TORCH_CUDA_ARCH_LIST": "8.6;9.0;10.0",
                "MAX_JOBS": "8",
            },
        )
    )


def _source_image(base: modal.Image) -> modal.Image:
    commit = _source_commit()
    clone = (
        "git init /opt/mvtracker && "
        "git -C /opt/mvtracker remote add origin https://github.com/J999UCL/mvtracker.git && "
        f"git -C /opt/mvtracker fetch --depth=1 origin {commit} && "
        "git -C /opt/mvtracker checkout --detach FETCH_HEAD && "
        f'test "$(git -C /opt/mvtracker rev-parse HEAD)" = "{commit}"'
    )
    return (
        base
        .run_commands(clone)
        .run_commands(
            "cp /opt/mvtracker-extension/mvtracker_indexed_correlation_cuda*.so "
            "/opt/mvtracker/mvtracker/models/core/mvtracker/",
        )
        .env(
            {
                "CUDA_HOME": "/usr/local/cuda",
                "MVTRACKER_MODAL_COMMIT": commit,
                "PYTHONPATH": f"{SOURCE_ROOT}:{SOURCE_ROOT / 'tools'}",
                "TORCH_CUDA_ARCH_LIST": "8.6;9.0;10.0",
                "CUMM_CUDA_ARCH_LIST": "8.6;9.0;10.0",
                "TORCH_EXTENSIONS_DIR": "/tmp/torch-extensions",
                "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor",
            }
        )
    )


def _runtime_image() -> modal.Image:
    return _source_image(_dependency_image())


app = modal.App(
    APP_NAME,
    tags={**BASE_TAGS, "experiment": "unclassified", "gpu": "unclassified"},
)
image = _runtime_image()
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True, version=2)
run_volume = modal.Volume.from_name(RUN_VOLUME_NAME, create_if_missing=True, version=2)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])
wandb_secret = modal.Secret.from_name(WANDB_SECRET_NAME, required_keys=["WANDB_API_KEY"])


@app.function(
    image=image,
    secrets=[hf_secret, wandb_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=8,
    memory=32768,
    ephemeral_disk=512 * 1024,
    timeout=4 * 60 * 60,
    max_containers=1,
    include_source=False,
)
def setup_data_remote() -> dict:
    import wandb

    from mvtracker.profiling.modal_data import materialize_profile_data

    run = wandb.init(
        project="mvtracker-modal-profiling",
        job_type="data-setup",
        tags=["modal", "data-setup", "diegesis", "mv-kubric-micro"],
        config={"source_commit": _source_commit(), **BASE_TAGS},
    )
    manifest = materialize_profile_data(DATA_ROOT)
    data_volume.commit()
    run.log(
        {
            "diegesis/scenes": sum(manifest["diegesis"]["splits"].values()),
            "diegesis/source_bytes": manifest["diegesis"]["source"]["size_bytes"],
            "mvkubric/scenes": manifest["mvkubric"]["scene_count"],
            "mvkubric/extracted_bytes": manifest["mvkubric"]["extracted"]["size_bytes"],
        }
    )
    run.finish()
    return manifest


@app.function(
    image=image,
    secrets=[wandb_secret],
    cpu=1,
    memory=4096,
    timeout=20 * 60,
    max_containers=1,
    include_source=False,
)
def validate_common_image_remote() -> dict:
    """Validate the common stack without requesting a GPU resource.

    CUDA extensions that need a device (the indexed-correlation extension in
    particular) are intentionally validated by ``compatibility`` on the
    requested GPU lane. There is no CPU fallback in this validation path.
    """
    import importlib.metadata
    import torch
    import wandb

    import flash_attn
    import pointops
    import spconv
    import triton

    versions = {
        "python": __import__("platform").python_version(),
        "torch": str(torch.__version__),
        "torchvision": importlib.metadata.version("torchvision"),
        "torchaudio": importlib.metadata.version("torchaudio"),
        "flash_attn": getattr(flash_attn, "__version__", "unknown"),
        "pointops": getattr(pointops, "__version__", "installed"),
        "spconv": getattr(spconv, "__version__", "installed"),
        "triton": triton.__version__,
        "cuda_runtime": str(torch.version.cuda),
        "cuda_available": torch.cuda.is_available(),
    }
    run = wandb.init(
        project="mvtracker-modal-profiling",
        job_type="common-image-validation",
        tags=["modal", "common-stack", "cpu-only", "no-gpu"],
        config={"source_commit": _source_commit(), **BASE_TAGS},
    )
    run.summary.update(versions)
    run.finish()
    return versions


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=8,
    memory=32768,
    timeout=4 * 60 * 60,
    max_containers=1,
    include_source=False,
)
def prepare_profile_batches_remote() -> dict:
    import wandb

    from mvtracker.cli.profile_training import prepare_profile_batch
    from mvtracker.profiling.modal_training import ProfileCase

    run = wandb.init(
        project="mvtracker-modal-profiling",
        job_type="batch-preparation",
        tags=["modal", "cpu", "mv-kubric-micro"],
        config={
            "source_commit": _source_commit(),
            "trajectories": 2048,
            **BASE_TAGS,
        },
    )
    results = []
    cases = tuple(
        ProfileCase(views=views, trajectories=2048, batch_size=8, accumulation=1)
        for views in range(1, 5)
    )
    for case in cases:
        results.append(
            prepare_profile_batch(
                data_root=DATA_ROOT / "datasets",
                output=PROFILE_BATCH_ROOT / f"{case.name}.pt",
                views=case.views,
                batch_size=case.batch_size,
            )
        )
    data_volume.commit()
    for index, result in enumerate(results):
        run.log(
            {
                "batch/index": index,
                "batch/views": result["views"],
                "batch/batch_size": result["batch_size"],
                "batch/attempted_samples": result["attempted_samples"],
                "batch/bytes": result["bytes"],
            },
            step=index,
        )
    run.finish()
    return {"batches": results}


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
def prepare_frontier_batches_remote() -> dict:
    import wandb

    from mvtracker.cli.profile_training import prepare_profile_batch
    from mvtracker.profiling.modal_training import FRONTIER_VIEWS

    cache_shapes = ((2048, 12), (6144, 1))
    run = wandb.init(
        project="mvtracker-modal-profiling",
        job_type="batch-preparation",
        tags=["modal", "cpu", "mv-kubric-micro", "views-5-6"],
        config={
            "source_commit": _source_commit(),
            "views": list(FRONTIER_VIEWS),
            "cache_shapes": [list(shape) for shape in cache_shapes],
            **BASE_TAGS,
        },
    )
    results = []
    for views in FRONTIER_VIEWS:
        for trajectories, batch_size in cache_shapes:
            result = prepare_profile_batch(
                data_root=DATA_ROOT / "datasets",
                output=(
                    PROFILE_BATCH_ROOT
                    / f"views{views}-traj{trajectories}-batch{batch_size}-accum1.pt"
                ),
                views=views,
                batch_size=batch_size,
                trajectories=trajectories,
            )
            results.append(result)
            run.log(
                {
                    "batch/views": views,
                    "batch/batch_size": batch_size,
                    "batch/trajectories": trajectories,
                    "batch/attempted_samples": result["attempted_samples"],
                    "batch/bytes": result["bytes"],
                },
                step=len(results),
            )
    data_volume.commit()
    run.finish()
    return {"batches": results}


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=8,
    memory=32768,
    timeout=4 * 60 * 60,
    max_containers=1,
    include_source=False,
)
def prepare_high_trajectory_batches_remote() -> dict:
    import wandb

    from mvtracker.cli.profile_training import prepare_profile_batch
    from mvtracker.profiling.modal_training import FRONTIER_VIEWS

    trajectories = 8192
    run = wandb.init(
        project="mvtracker-modal-profiling",
        job_type="batch-preparation",
        tags=["modal", "cpu", "mv-kubric-micro", "high-trajectory"],
        config={
            "source_commit": _source_commit(),
            "views": list(FRONTIER_VIEWS),
            "trajectories": trajectories,
            **BASE_TAGS,
        },
    )
    results = []
    for views in FRONTIER_VIEWS:
        result = prepare_profile_batch(
            data_root=DATA_ROOT / "datasets",
            output=(
                PROFILE_BATCH_ROOT
                / f"views{views}-traj{trajectories}-batch1-accum1.pt"
            ),
            views=views,
            batch_size=1,
            trajectories=trajectories,
        )
        results.append(result)
        run.log(
            {
                "batch/views": views,
                "batch/trajectories": trajectories,
                "batch/attempted_samples": result["attempted_samples"],
                "batch/bytes": result["bytes"],
            },
            step=len(results),
        )
    data_volume.commit()
    run.finish()
    return {"batches": results}


def _trial_command(case, warmup, measured, output, gpu_spec):
    if case.views in (5, 6):
        if case.trajectories > 2048:
            cache_name = f"views{case.views}-traj8192-batch1-accum1.pt"
        else:
            cache_name = f"views{case.views}-traj2048-batch12-accum1.pt"
    else:
        cache_name = f"views{case.views}-traj2048-batch8-accum1.pt"
    return [
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
        str(PROFILE_BATCH_ROOT / cache_name),
        "--views",
        str(case.views),
        "--batch-size",
        str(case.batch_size),
        "--accumulation",
        str(case.accumulation),
        "--trajectories",
        str(case.trajectories),
        "--warmup-updates",
        str(warmup),
        "--measure-updates",
        str(measured),
        "--gpu-lane",
        gpu_spec,
    ]


def _profile_options(gpu_spec: str) -> dict:
    if gpu_spec not in GPU_LANES:
        raise ValueError(f"unsupported GPU lane {gpu_spec!r}")
    return {
        "image": image,
        "secrets": [wandb_secret],
        "gpu": gpu_spec,
        "volumes": {
            str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
            str(RUN_ROOT): run_volume,
        },
        "cpu": 8,
        "memory": 32768,
        "timeout": 45 * 60,
        "max_containers": 1,
        "include_source": False,
    }


def _profile_remote(gpu_spec: str, mode: str, run_name: str) -> dict:
    """Run one exact GPU lane; each trial is an isolated child process."""
    import wandb

    from mvtracker.profiling.modal_training import (
        BATCH_CANDIDATES,
        PROFILE_CASES,
        ProfileCase,
        TrialResult,
        find_largest_safe_batch,
        validate_gpu_request,
    )

    validate_gpu_request(gpu_spec)
    if mode not in {
        "smoke",
        "sweep",
        "frontier56",
        "trajectory56",
        "compatibility",
    }:
        raise ValueError("unsupported profiling mode")
    if _RUN_NAME.fullmatch(run_name) is None:
        raise ValueError("run name contains unsupported characters")

    output_root = RUN_ROOT / run_name
    output_root.mkdir(parents=True, exist_ok=False)
    trials_root = output_root / "trials"
    trials_root.mkdir()
    log_root = output_root / "logs"
    log_root.mkdir()
    run = wandb.init(
        project="mvtracker-modal-profiling",
        job_type="training-shape-profile",
        name=run_name,
        tags=[
            "modal",
            "gpu-economics-profile",
            gpu_spec.rstrip("!").lower(),
            mode,
            "mv-kubric-micro",
        ],
        config={
            "source_commit": _source_commit(),
            "gpu": gpu_spec,
            "mode": mode,
            "memory_safety_fraction": 0.90,
            "trajectory_targets": [1024, 2048],
            "batch_candidates": (
                list(range(1, 13)) if mode == "frontier56" else list(BATCH_CANDIDATES)
            ),
            **BASE_TAGS,
        },
    )

    if mode == "compatibility":
        result_path = output_root / "compatibility.json"
        command = [
            sys.executable,
            "-m",
            "mvtracker.cli.profile_training",
            "--output",
            str(result_path),
            "--gpu-lane",
            gpu_spec,
            "--compatibility-only",
        ]
        log_path = log_root / "compatibility.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command,
                cwd=SOURCE_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0 or not result_path.is_file():
            raise RuntimeError(f"compatibility check failed; see {log_path}")
        summary = json.loads(result_path.read_text(encoding="utf-8"))
        summary["mode"] = mode
        summary["gpu_lane"] = gpu_spec
    else:
        trial_number = 0

        def execute(
            case: ProfileCase,
            warmup: int,
            measured: int,
            phase: str,
            *,
            requested: int | None = None,
        ):
            nonlocal trial_number
            trial_number += 1
            stem = (
                f"{trial_number:03d}-{case.name}-{phase}"
            )
            result_path = trials_root / f"{stem}.json"
            log_path = log_root / f"{stem}.log"
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    _trial_command(case, warmup, measured, result_path, gpu_spec),
                    cwd=SOURCE_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result_path.is_file():
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            else:
                payload = {
                    "status": "error",
                    "views": case.views,
                    "batch_size": case.batch_size,
                    "trajectories": case.trajectories,
                    "error": f"profile subprocess exited {completed.returncode}",
                }
                result_path.write_text(json.dumps(payload, indent=2) + "\n")
            status = payload.get("status")
            if status not in {"safe", "unsafe", "oom", "error"}:
                raise RuntimeError(f"profile trial {stem} returned {status!r}")
            metrics = {
                "trial/index": trial_number,
                "trial/views": case.views,
                "trial/batch_size": case.batch_size,
                "trial/accumulation": case.accumulation,
                "trial/trajectories": case.trajectories,
                "trial/safe": int(status == "safe"),
            }
            peak = payload.get("peak_memory_bytes")
            total = payload.get("total_memory_bytes")
            if peak is not None and total is not None:
                metrics["trial/peak_memory_fraction"] = peak / total
            if "median" in payload:
                metrics.update(
                    {
                        "trial/total_update_ms": payload["median"]["total_update_ms"],
                        "trial/forward_ms": payload["median"]["forward_ms"],
                        "trial/backward_ms": payload["median"]["backward_ms"],
                        "trial/scenes_per_second": payload["scenes_per_second"],
                        "trial/trajectories_per_second": payload[
                            "trajectories_per_second"
                        ],
                    }
                )
            run.log(metrics, step=trial_number)
            run_volume.commit()
            return TrialResult(
                requested=case.batch_size if requested is None else requested,
                status=status,
                peak_memory_bytes=peak,
                total_memory_bytes=total,
                result_path=str(result_path.relative_to(output_root)),
            )

        if mode == "smoke":
            case = ProfileCase(views=1, trajectories=1024, batch_size=1)
            result = execute(case, 1, 1, "smoke")
            summary = {
                "mode": mode,
                "run_name": run_name,
                "case": case.__dict__,
                "trial": result.__dict__,
            }
        elif mode == "sweep":
            cases = []
            for target in PROFILE_CASES:
                search = find_largest_safe_batch(
                    lambda batch_size: execute(
                        ProfileCase(
                            views=target.views,
                            trajectories=target.trajectories,
                            batch_size=batch_size,
                        ),
                        1,
                        1,
                        "capacity",
                    )
                )
                selected = search.selected
                confirmation = None
                if selected is not None:
                    candidate_index = BATCH_CANDIDATES.index(selected)
                    while candidate_index >= 0:
                        candidate = BATCH_CANDIDATES[candidate_index]
                        confirmation = execute(
                            ProfileCase(
                                views=target.views,
                                trajectories=target.trajectories,
                                batch_size=candidate,
                            ),
                            2,
                            3,
                            "confirm",
                        )
                        if confirmation.safe:
                            selected = candidate
                            break
                        candidate_index -= 1
                    else:
                        selected = None
                cases.append(
                    {
                        "case": target.__dict__,
                        "selected_batch_size": selected,
                        "search_trials": [trial.__dict__ for trial in search.trials],
                        "confirmation": (
                            confirmation.__dict__ if confirmation is not None else None
                        ),
                    }
                )
            summary = {
                "mode": mode,
                "run_name": run_name,
                "gpu_lane": gpu_spec,
                "cases": cases,
            }
        elif mode == "frontier56":
            from mvtracker.profiling.modal_training import (
                FRONTIER_BATCH_CANDIDATES,
                FRONTIER_CASES,
                FRONTIER_TRAJECTORY_CANDIDATES,
                FRONTIER_VIEWS,
                find_largest_safe,
            )

            capacity = []
            for views in FRONTIER_VIEWS:
                search = find_largest_safe(
                    lambda trajectories: execute(
                        ProfileCase(
                            views=views,
                            trajectories=trajectories,
                            batch_size=1,
                        ),
                        1,
                        1,
                        "trajectory-capacity",
                        requested=trajectories,
                    ),
                    FRONTIER_TRAJECTORY_CANDIDATES,
                )
                selected = search.selected
                confirmation = None
                if selected is not None:
                    confirmation = execute(
                        ProfileCase(views=views, trajectories=selected, batch_size=1),
                        2,
                        3,
                        "trajectory-confirm",
                        requested=selected,
                    )
                    if not confirmation.safe:
                        selected = None
                capacity.append(
                    {
                        "views": views,
                        "selected_trajectories": selected,
                        "search_trials": [trial.__dict__ for trial in search.trials],
                        "confirmation": (
                            confirmation.__dict__ if confirmation is not None else None
                        ),
                    }
                )

            cases = []
            for target in FRONTIER_CASES:
                search = find_largest_safe_batch(
                    lambda batch_size: execute(
                        ProfileCase(
                            views=target.views,
                            trajectories=target.trajectories,
                            batch_size=batch_size,
                        ),
                        1,
                        1,
                        "batch-capacity",
                    ),
                    FRONTIER_BATCH_CANDIDATES,
                )
                selected = search.selected
                confirmation = None
                if selected is not None:
                    confirmation = execute(
                        ProfileCase(
                            views=target.views,
                            trajectories=target.trajectories,
                            batch_size=selected,
                        ),
                        2,
                        3,
                        "batch-confirm",
                    )
                    if not confirmation.safe:
                        selected = None
                cases.append(
                    {
                        "case": target.__dict__,
                        "selected_batch_size": selected,
                        "search_trials": [trial.__dict__ for trial in search.trials],
                        "confirmation": (
                            confirmation.__dict__ if confirmation is not None else None
                        ),
                    }
                )
            summary = {
                "mode": mode,
                "run_name": run_name,
                "gpu_lane": gpu_spec,
                "trajectory_capacity": capacity,
                "batch_capacity": cases,
            }
        else:
            from mvtracker.profiling.modal_training import (
                HIGH_TRAJECTORY_CANDIDATES,
                find_largest_safe,
            )

            views_to_profile = (5,) if gpu_spec == "H200" else (5, 6)
            capacity = []
            for views in views_to_profile:
                search = find_largest_safe(
                    lambda trajectories: execute(
                        ProfileCase(
                            views=views,
                            trajectories=trajectories,
                            batch_size=1,
                        ),
                        1,
                        1,
                        "high-trajectory-capacity",
                        requested=trajectories,
                    ),
                    HIGH_TRAJECTORY_CANDIDATES,
                )
                selected = search.selected
                confirmation = None
                if selected is not None:
                    confirmation = execute(
                        ProfileCase(views=views, trajectories=selected, batch_size=1),
                        2,
                        3,
                        "high-trajectory-confirm",
                        requested=selected,
                    )
                    if not confirmation.safe:
                        selected = None
                capacity.append(
                    {
                        "views": views,
                        "selected_trajectories": selected,
                        "search_trials": [trial.__dict__ for trial in search.trials],
                        "confirmation": (
                            confirmation.__dict__ if confirmation is not None else None
                        ),
                    }
                )
            summary = {
                "mode": mode,
                "run_name": run_name,
                "gpu_lane": gpu_spec,
                "trajectory_capacity": capacity,
            }

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    run.summary["result_path"] = str(summary_path)
    run.summary["gpu_lane"] = gpu_spec
    run.finish()
    run_volume.commit()
    return summary


@app.function(**_profile_options("H100!"))
def profile_h100_remote(mode: str, run_name: str) -> dict:
    return _profile_remote("H100!", mode, run_name)


@app.function(**_profile_options("H200"))
def profile_h200_remote(mode: str, run_name: str) -> dict:
    return _profile_remote("H200", mode, run_name)


@app.function(**_profile_options("B200"))
def profile_b200_remote(mode: str, run_name: str) -> dict:
    return _profile_remote("B200", mode, run_name)


def _default_run_name(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}"


def _set_run_tags(*, experiment: str, gpu: str) -> None:
    app.set_tags(
        {
            **BASE_TAGS,
            "experiment": experiment,
            "gpu": gpu,
        }
    )


@app.local_entrypoint(name="setup-data")
def setup_data() -> None:
    _set_run_tags(experiment="data-setup", gpu="cpu")
    print(json.dumps(setup_data_remote.remote(), indent=2))


@app.local_entrypoint(name="smoke")
def smoke(run_name: str = "", gpu: str = "H100!") -> None:
    selected = run_name or _default_run_name("smoke")
    _set_run_tags(experiment=selected, gpu=gpu)
    profile = {
        "H100!": profile_h100_remote,
        "H200": profile_h200_remote,
        "B200": profile_b200_remote,
    }[gpu]
    print(json.dumps(profile.remote("smoke", selected), indent=2))


@app.local_entrypoint(name="prepare-batches")
def prepare_batches() -> None:
    _set_run_tags(experiment="batch-preparation", gpu="cpu")
    print(json.dumps(prepare_profile_batches_remote.remote(), indent=2))


@app.local_entrypoint(name="prepare-frontier56-batches")
def prepare_frontier56_batches() -> None:
    _set_run_tags(experiment="frontier56-batch-preparation", gpu="cpu")
    print(json.dumps(prepare_frontier_batches_remote.remote(), indent=2))


@app.local_entrypoint(name="prepare-high-trajectory-batches")
def prepare_high_trajectory_batches() -> None:
    _set_run_tags(experiment="high-trajectory-batch-preparation", gpu="cpu")
    print(json.dumps(prepare_high_trajectory_batches_remote.remote(), indent=2))


@app.local_entrypoint(name="run-profile")
def run_profile(run_name: str = "", gpu: str = "H100!") -> None:
    selected = run_name or _default_run_name("profile")
    _set_run_tags(experiment=selected, gpu=gpu)
    profile = {
        "H100!": profile_h100_remote,
        "H200": profile_h200_remote,
        "B200": profile_b200_remote,
    }[gpu]
    print(json.dumps(profile.remote("sweep", selected), indent=2))


@app.local_entrypoint(name="run-frontier56")
def run_frontier56(run_name: str = "", gpu: str = "H200") -> None:
    selected = run_name or _default_run_name("frontier56")
    _set_run_tags(experiment=selected, gpu=gpu)
    profile = {
        "H200": profile_h200_remote,
        "B200": profile_b200_remote,
    }[gpu]
    print(json.dumps(profile.remote("frontier56", selected), indent=2))


@app.local_entrypoint(name="run-trajectory56")
def run_trajectory56(run_name: str = "", gpu: str = "H200") -> None:
    selected = run_name or _default_run_name("trajectory56")
    _set_run_tags(experiment=selected, gpu=gpu)
    profile = {
        "H200": profile_h200_remote,
        "B200": profile_b200_remote,
    }[gpu]
    print(json.dumps(profile.remote("trajectory56", selected), indent=2))


@app.local_entrypoint(name="validate-image")
def validate_image() -> None:
    _set_run_tags(experiment="common-stack", gpu="cpu")
    print(json.dumps(validate_common_image_remote.remote(), indent=2))


@app.local_entrypoint(name="compatibility")
def compatibility(run_name: str = "", gpu: str = "H100!") -> None:
    selected = run_name or _default_run_name("compatibility")
    _set_run_tags(experiment=selected, gpu=gpu)
    profile = {
        "H100!": profile_h100_remote,
        "H200": profile_h200_remote,
        "B200": profile_b200_remote,
    }[gpu]
    print(json.dumps(profile.remote("compatibility", selected), indent=2))
