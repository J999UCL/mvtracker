"""Modal setup and single-H100 MVTracker training-shape profiling."""

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
PROFILE_GPU = "H100!"
SOURCE_ROOT = Path("/opt/mvtracker")
DATA_ROOT = Path("/mnt/mvtracker-data")
RUN_ROOT = Path("/mnt/mvtracker-runs")
_COMMIT = re.compile(r"[0-9a-f]{40}")
_RUN_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _source_commit() -> str:
    commit = os.environ.get("MVTRACKER_MODAL_COMMIT", "")
    if _COMMIT.fullmatch(commit) is None:
        raise RuntimeError("MVTRACKER_MODAL_COMMIT must be one full lowercase Git commit")
    return commit


def _runtime_image() -> modal.Image:
    commit = _source_commit()
    clone = (
        "git init /opt/mvtracker && "
        "git -C /opt/mvtracker remote add origin https://github.com/J999UCL/mvtracker.git && "
        f"git -C /opt/mvtracker fetch --depth=1 origin {commit} && "
        "git -C /opt/mvtracker checkout --detach FETCH_HEAD && "
        f'test "$(git -C /opt/mvtracker rev-parse HEAD)" = "{commit}"'
    )
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.1.1-devel-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install(
            "build-essential",
            "ffmpeg",
            "git",
            "libgl1",
            "libglib2.0-0",
            "ninja-build",
            "python3-dev",
        )
        .pip_install(
            "torch==2.5.1",
            "torchvision==0.20.1",
            index_url="https://download.pytorch.org/whl/cu121",
        )
        .run_commands(clone)
        .run_commands("python -m pip install -r /opt/mvtracker/requirements.full.txt")
        .pip_install("hf-xet", "nvidia-ml-py")
        .run_commands(
            "python -m pip install --no-build-isolation flash-attn==2.8.3.post1",
            "python -m pip install 'git+https://github.com/ethz-vlg/pointcept.git@2082918#subdirectory=libs/pointops'",
            env={"CUDA_HOME": "/usr/local/cuda", "TORCH_CUDA_ARCH_LIST": "9.0"},
        )
        .env(
            {
                "CUDA_HOME": "/usr/local/cuda",
                "PYTHONPATH": str(SOURCE_ROOT),
                "TORCH_CUDA_ARCH_LIST": "9.0",
                "TORCH_EXTENSIONS_DIR": "/tmp/torch-extensions",
                "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor",
            }
        )
    )


app = modal.App(APP_NAME)
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
    ephemeral_disk=100 * 1024,
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
        config={"source_commit": _source_commit()},
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


def _trial_command(case, trajectories, warmup, measured, output):
    return [
        sys.executable,
        "-m",
        "mvtracker.cli.profile_training",
        "--data-root",
        str(DATA_ROOT / "datasets/diegesis-mvtracker"),
        "--checkpoint",
        str(DATA_ROOT / "checkpoints/mvtracker_200000_june2025_cleandepth.pth"),
        "--output",
        str(output),
        "--views",
        str(case.views),
        "--batch-size",
        str(case.batch_size),
        "--accumulation",
        str(case.accumulation),
        "--trajectories",
        str(trajectories),
        "--warmup-updates",
        str(warmup),
        "--measure-updates",
        str(measured),
    ]


@app.function(
    image=image,
    secrets=[wandb_secret],
    gpu=PROFILE_GPU,
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
        str(RUN_ROOT): run_volume,
    },
    cpu=8,
    memory=32768,
    timeout=45 * 60,
    max_containers=1,
    include_source=False,
)
def profile_remote(mode: str, run_name: str) -> dict:
    import wandb

    from mvtracker.profiling.modal_training import (
        PROFILE_CASES,
        TRAJECTORY_CANDIDATES,
        ProfileCase,
        TrialResult,
        find_largest_safe,
        validate_gpu_request,
    )

    validate_gpu_request(PROFILE_GPU)
    if mode not in {"smoke", "sweep"}:
        raise ValueError("mode must be smoke or sweep")
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
        tags=["modal", "h100", mode, "diegesis"],
        config={
            "source_commit": _source_commit(),
            "gpu": PROFILE_GPU,
            "mode": mode,
            "memory_safety_fraction": 0.90,
            "trajectory_ceiling": 2048,
        },
    )
    trial_number = 0

    def execute(case, trajectories, warmup, measured, phase):
        nonlocal trial_number
        trial_number += 1
        stem = f"{trial_number:02d}-{case.name}-n{trajectories}-{phase}"
        result_path = trials_root / f"{stem}.json"
        log_path = log_root / f"{stem}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                _trial_command(
                    case,
                    trajectories,
                    warmup,
                    measured,
                    result_path,
                ),
                cwd=SOURCE_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0 or not result_path.is_file():
            raise RuntimeError(
                f"profile trial {stem} failed; see {log_path}"
            )
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        status = payload["status"]
        if status not in {"safe", "unsafe", "oom"}:
            raise RuntimeError(f"profile trial {stem} returned {status!r}")
        peak = payload.get("peak_memory_bytes")
        total = payload.get("total_memory_bytes")
        metrics = {
            "trial/index": trial_number,
            "trial/views": case.views,
            "trial/batch_size": case.batch_size,
            "trial/accumulation": case.accumulation,
            "trial/trajectories": trajectories,
            "trial/safe": int(status == "safe"),
        }
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
            trajectories=trajectories,
            status=status,
            peak_memory_bytes=peak,
            total_memory_bytes=total,
            result_path=str(result_path.relative_to(output_root)),
        )

    if mode == "smoke":
        case = ProfileCase(views=1, batch_size=1, accumulation=1)
        result = execute(case, 256, 1, 1, "smoke")
        summary = {
            "mode": mode,
            "run_name": run_name,
            "trial": result.__dict__,
        }
    else:
        cases = []
        for case in PROFILE_CASES:
            search = find_largest_safe(
                lambda trajectories: execute(
                    case, trajectories, 1, 1, "capacity"
                )
            )
            selected = search.selected_trajectories
            confirmation = None
            if selected is not None:
                candidate_index = TRAJECTORY_CANDIDATES.index(selected)
                while candidate_index >= 0:
                    candidate = TRAJECTORY_CANDIDATES[candidate_index]
                    confirmation = execute(case, candidate, 2, 3, "confirm")
                    if confirmation.safe:
                        selected = candidate
                        break
                    candidate_index -= 1
                else:
                    selected = None
            cases.append(
                {
                    "case": case.__dict__,
                    "selected_trajectories": selected,
                    "search_trials": [trial.__dict__ for trial in search.trials],
                    "confirmation": (
                        confirmation.__dict__ if confirmation is not None else None
                    ),
                }
            )
        summary = {"mode": mode, "run_name": run_name, "cases": cases}

    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    run.summary["result_path"] = str(summary_path)
    run.finish()
    run_volume.commit()
    return summary


def _default_run_name(prefix: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{timestamp}"


@app.local_entrypoint(name="setup-data")
def setup_data() -> None:
    print(json.dumps(setup_data_remote.remote(), indent=2))


@app.local_entrypoint(name="smoke")
def smoke(run_name: str = "") -> None:
    selected = run_name or _default_run_name("smoke")
    print(json.dumps(profile_remote.remote("smoke", selected), indent=2))


@app.local_entrypoint(name="run-profile")
def run_profile(run_name: str = "") -> None:
    selected = run_name or _default_run_name("profile")
    print(json.dumps(profile_remote.remote("sweep", selected), indent=2))
