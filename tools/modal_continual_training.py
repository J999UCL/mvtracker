"""Modal setup and two-H200 launcher for the continual-training experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import modal

from modal_training_profile import (
    DATA_ROOT,
    RUN_ROOT,
    _dependency_image,
    _source_image,
    _source_commit,
    data_volume,
    run_volume,
    wandb_secret,
)

from mvtracker.profiling.modal_continual_training import (
    CONTINUAL_RUN_SUBDIR,
    DATA_LAYOUT_VERSION,
    DATA_VOLUME_ROOT,
    GPU_REQUEST,
    MAIN_CONFIRMATION,
    MAX_CONTAINERS,
    TRAIN_MEMORY_LIMIT_MIB,
    TRAIN_MEMORY_REQUEST_MIB,
    MODAL_TAGS,
    PROFILE_TAGS,
    WANDB_ENTITY,
    WANDB_GROUP,
    WANDB_PROJECT,
    preflight_active_containers,
    require_main_confirmation,
    require_pushed_main_commit,
    require_remote_main_confirmation,
    validate_run_name,
)
APP_NAME = "jeet-mvtracker-continual-training"
SOURCE_ROOT = Path("/opt/mvtracker")
LOADER_PROFILE_WARMUP = 4
LOADER_PROFILE_MEASURED = 32
H100_LOADER_PROFILE_WARMUP = 20
H100_LOADER_PROFILE_MEASURED = 100
EXPERIMENT_PHASES = {
    "smoke": (
        {
            "name": "initial",
            "config": "diegesis_mvkubric_gt_ddp_smoke_phase1",
            "target_completed_steps": 3,
        },
        {
            "name": "resume",
            "config": "diegesis_mvkubric_gt_ddp_smoke",
            "target_completed_steps": 5,
        },
    ),
    "smoke10": (
        {
            "name": "smoke10",
            "config": "diegesis_mvkubric_gt_ddp_smoke10",
            "target_completed_steps": 10,
        },
    ),
    "production_smoke10": (
        {
            "name": "production_smoke10",
            "config": "diegesis_mvkubric_gt_ddp_production_smoke10",
            "target_completed_steps": 10,
        },
    ),
    "memory_profile": (
        {
            "name": "memory_profile",
            "config": "diegesis_mvkubric_gt_ddp_memory_profile",
            "target_completed_steps": 2,
        },
    ),
    "main": (
        {
            "name": "main",
            "config": "diegesis_mvkubric_gt_ddp",
            "target_completed_steps": 1000,
        },
    ),
}

app = modal.App(
    APP_NAME,
    tags={**MODAL_TAGS, "experiment": "continual-training-worker"},
)


training_image = _source_image(_dependency_image())


def _run_identity(run_name: str, commit: str) -> tuple[int, str]:
    digest = hashlib.sha256(f"{commit}:{run_name}".encode()).digest()
    return int.from_bytes(digest[:4], "big"), digest.hex()[:12]


def _profile_volume_cpu() -> dict:
    from mvtracker.profiling.modal_continual_data import profile_encoded_loader

    profiles = {}
    for source in ("diegesis", "mvkubric"):
        profiles[source] = {
            "cold": profile_encoded_loader(
                Path(DATA_VOLUME_ROOT),
                source=source,
                warmup=0,
                measured=4,
                workers=0,
                use_cuda=False,
            ),
            "warm": profile_encoded_loader(
                Path(DATA_VOLUME_ROOT),
                source=source,
                warmup=LOADER_PROFILE_WARMUP,
                measured=LOADER_PROFILE_MEASURED,
                workers=0,
                use_cuda=False,
            ),
        }
    return {
        "data_volume_root": DATA_VOLUME_ROOT,
        "data_layout_version": DATA_LAYOUT_VERSION,
        "profiles": profiles,
    }


def _profile_h100_mvkubric_loader() -> dict:
    from mvtracker.profiling.modal_continual_data import profile_encoded_loader

    manifest = json.loads(
        (Path(DATA_VOLUME_ROOT) / "direct-volume-data-manifest.json").read_text()
    )
    scene_ids = manifest["train_scene_ids"][:25]
    profile = profile_encoded_loader(
        Path(DATA_VOLUME_ROOT),
        source="mvkubric",
        warmup=H100_LOADER_PROFILE_WARMUP,
        measured=H100_LOADER_PROFILE_MEASURED,
        workers=8,
        use_cuda=True,
        mvkubric_scene_ids=scene_ids,
    )
    return {
        "profiles": {"mvkubric": profile},
        "staging": {"scene_ids": scene_ids, "data_root": DATA_VOLUME_ROOT},
    }


def _profile_summary(result: dict) -> dict:
    summary = {}
    if "staging" in result:
        staging = result["staging"]
        if "elapsed_seconds" in staging:
            summary["staging/elapsed_seconds"] = staging["elapsed_seconds"]
        if "copied_size_bytes" in staging:
            summary["staging/copied_bytes"] = staging["copied_size_bytes"]
    for source, profile in result["profiles"].items():
        phases = profile if "cold" in profile else {"measured": profile}
        for phase, measurements in phases.items():
            prefix = f"loader/{source}/{phase}"
            summary.update(
                {
                    f"{prefix}/samples_per_second": measurements["samples_per_second"],
                    f"{prefix}/sample_seconds_median": measurements["sample_seconds_median"],
                    f"{prefix}/sample_seconds_p95": measurements["sample_seconds_p95"],
                }
            )
    return summary


@app.function(
    image=training_image,
    secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume.with_mount_options(read_only=True)},
    cpu=32,
    memory=65536,
    timeout=4 * 60 * 60,
    max_containers=MAX_CONTAINERS,
    include_source=False,
)
def profile_cpu_loader_remote() -> dict:
    import wandb

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        job_type="volume-cpu-validation",
        tags=["modal", "volume-v2", "cpu", "cold-warm-loader"],
        config={
            "source_commit": _source_commit(),
            "data_layout_version": DATA_LAYOUT_VERSION,
            **PROFILE_TAGS,
        },
    )
    result = _profile_volume_cpu()
    run.summary.update(_profile_summary(result))
    run.finish()
    return result


@app.function(
    image=training_image,
    secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume.with_mount_options(read_only=True)},
    gpu="H100!",
    cpu=32,
    memory=65536,
    timeout=4 * 60 * 60,
    max_containers=MAX_CONTAINERS,
    include_source=False,
)
def profile_h100_loader_remote() -> dict:
    import wandb

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        job_type="encoded-loader-h100",
        tags=["modal", "encoded-loader", "local-ssd", "h100"],
        config={"source_commit": _source_commit(), **PROFILE_TAGS},
    )
    result = _profile_h100_mvkubric_loader()
    run.summary.update(_profile_summary(result))
    run.finish()
    return result


@app.function(
    image=training_image,
    secrets=[wandb_secret],
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
        str(RUN_ROOT): run_volume,
    },
    gpu=GPU_REQUEST,
    cpu=32,
    memory=(TRAIN_MEMORY_REQUEST_MIB, TRAIN_MEMORY_LIMIT_MIB),
    timeout=6 * 60 * 60,
    max_containers=MAX_CONTAINERS,
    retries=0,
    include_source=False,
)
def train_remote(
    mode: str,
    run_name: str,
    confirmation: str = "",
    materialize_whole_step: bool = False,
    seed_override: int = 0,
) -> dict:
    if mode not in EXPERIMENT_PHASES:
        raise ValueError("unsupported training mode")
    require_remote_main_confirmation(mode, confirmation)
    validate_run_name(run_name)
    commit = _source_commit()
    derived_seed, wandb_run_id = _run_identity(run_name, commit)
    seed = seed_override or derived_seed
    run_dir = RUN_ROOT / CONTINUAL_RUN_SUBDIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "modal-run-manifest.json"
    data_inventory = json.loads(
        (Path(DATA_VOLUME_ROOT) / "direct-volume-data-manifest.json").read_text()
    )
    modal_tags = PROFILE_TAGS if mode == "memory_profile" else MODAL_TAGS
    manifest = {
        "mode": mode,
        "run_name": run_name,
        "source_commit": commit,
        "data_volume": "jeet-mvtracker-data-v2",
        "data_layout_version": DATA_LAYOUT_VERSION,
        "data_inventory": {
            "train_scene_count": data_inventory["train_scene_count"],
            "validation_scene_count": data_inventory["validation_scene_count"],
            "validation_scene_ids": data_inventory["validation_scene_ids"],
        },
        "phases": [dict(phase) for phase in EXPERIMENT_PHASES[mode]],
        "gpu": GPU_REQUEST,
        "max_containers": MAX_CONTAINERS,
        "master_seed": seed,
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "wandb_group": WANDB_GROUP,
        "wandb_run_id": wandb_run_id,
        "materialize_whole_step": materialize_whole_step,
        "modal_tags": modal_tags,
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise RuntimeError("existing run manifest does not match this launch")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        run_volume.commit()

    environment = os.environ.copy()
    environment.update(
        {
            "MVTRACKER_TRAINING_RUN_DIR": str(run_dir),
            "MVTRACKER_TRAINING_CHECKPOINT": str(
                Path(DATA_VOLUME_ROOT) / "checkpoints/mvtracker_200000_june2025.pth"
            ),
            "MVTRACKER_DATA_ROOT": DATA_VOLUME_ROOT,
            "MVTRACKER_MVKUBRIC_INDEX_ROOT": str(
                Path(DATA_VOLUME_ROOT)
                / "datasets/kubric-multiview/train/MVTracker_index"
            ),
            "MVTRACKER_TRAINING_SEED": str(seed),
            "MVTRACKER_WANDB_RUN_NAME": run_name,
            "MVTRACKER_WANDB_RUN_ID": wandb_run_id,
            "WANDB_ENTITY": WANDB_ENTITY,
            "WANDB_PROJECT": WANDB_PROJECT,
            "WANDB_RUN_GROUP": WANDB_GROUP,
            "WANDB_RUN_ID": wandb_run_id,
            "WANDB_RESUME": "allow",
        }
    )
    log_path = run_dir / "training.log"
    with log_path.open("a", encoding="utf-8") as log:
        for phase in EXPERIMENT_PHASES[mode]:
            log.write(f"\n=== {phase['name']} target={phase['target_completed_steps']} ===\n")
            log.flush()
            command = [
                sys.executable,
                "-m",
                "mvtracker.cli.train",
                f"+experiment={phase['config']}",
                f"datasets.train.materialize_whole_step={str(materialize_whole_step).lower()}",
            ]
            completed = subprocess.run(
                command,
                cwd=SOURCE_ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            run_volume.commit()
            if completed.returncode != 0:
                raise RuntimeError(
                    f"training phase {phase['name']} exited {completed.returncode}; "
                    f"see {log_path}"
                )
    return manifest


def _default_run_name(mode: str, commit: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if mode == "production_smoke10":
        return f"gt-replay-prod-smoke10-{commit[:8]}-{timestamp}"
    return f"gt-depth-replay-{mode}-ddp2-h200-{commit[:8]}-{timestamp}"


def _prepare_launch(mode: str, run_name: str, confirm_main: bool) -> str:
    require_main_confirmation(mode, confirm_main)
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers()
    selected = run_name or _default_run_name(mode, commit)
    validate_run_name(selected)
    app.set_tags({**MODAL_TAGS, "experiment": selected, "gpu": "h200x2"})
    return selected


def _spawn_training(
    mode: str,
    run_name: str,
    confirmation: str = "",
    materialize_whole_step: bool = False,
    seed: int = 0,
) -> None:
    deployed_training = modal.Function.from_name(APP_NAME, "train_remote")
    call = deployed_training.spawn(
        mode,
        run_name,
        confirmation,
        materialize_whole_step,
        seed,
    )
    print(
        json.dumps(
            {"run_name": run_name, "function_call_id": call.object_id},
            indent=2,
        )
    )


@app.local_entrypoint(name="profile-cpu-loader")
def profile_cpu_loader() -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    app.set_tags({**PROFILE_TAGS, "experiment": "volume-v2-cpu", "gpu": "cpu"})
    print(json.dumps(profile_cpu_loader_remote.remote(), indent=2))


@app.local_entrypoint(name="profile-h100-loader")
def profile_h100_loader() -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=1)
    app.set_tags({**PROFILE_TAGS, "experiment": "encoded-loader-h100", "gpu": "h100"})
    print(json.dumps(profile_h100_loader_remote.remote(), indent=2))


@app.local_entrypoint(name="smoke")
def smoke(run_name: str = "") -> None:
    selected = _prepare_launch("smoke", run_name, confirm_main=False)
    _spawn_training("smoke", selected)


@app.local_entrypoint(name="smoke10")
def smoke10(
    run_name: str = "",
    materialize_whole_step: bool = False,
    seed: int = 0,
) -> None:
    selected = _prepare_launch("smoke10", run_name, confirm_main=False)
    _spawn_training("smoke10", selected, materialize_whole_step=materialize_whole_step, seed=seed)


@app.local_entrypoint(name="production_smoke10")
def production_smoke10(
    run_name: str = "",
    materialize_whole_step: bool = False,
    seed: int = 0,
) -> None:
    selected = _prepare_launch("production_smoke10", run_name, confirm_main=False)
    _spawn_training(
        "production_smoke10",
        selected,
        materialize_whole_step=materialize_whole_step,
        seed=seed,
    )


@app.local_entrypoint(name="memory-profile")
def memory_profile(run_name: str = "", seed: int = 0) -> None:
    selected = _prepare_launch("memory_profile", run_name, confirm_main=False)
    app.set_tags(
        {**PROFILE_TAGS, "experiment": selected, "gpu": "h200x2"}
    )
    _spawn_training("memory_profile", selected, seed=seed)


@app.local_entrypoint(name="train")
def train(run_name: str = "", confirm_main: bool = False) -> None:
    selected = _prepare_launch("main", run_name, confirm_main)
    _spawn_training("main", selected, MAIN_CONFIRMATION)
