"""Modal setup and two-H100 launcher for the continual-training experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import modal

from modal_dataset_image_build import (
    TRAIN_ARCHIVE_STAGES,
    TRAIN_SCENES,
    install_dataset_base,
    install_mvkubric_archive,
    install_mvkubric_validation_and_index,
)

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
    DATASET_IMAGE_ROOT,
    DATASET_IMAGE_VERSION,
    EPHEMERAL_DISK_MIB,
    GPU_REQUEST,
    MAIN_CONFIRMATION,
    MAX_CONTAINERS,
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
from mvtracker.profiling.modal_mvkubric2000 import TRAIN_ARCHIVES as PINNED_MVKUBRIC_ARCHIVES


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
    "main": (
        {
            "name": "main",
            "config": "diegesis_mvkubric_gt_ddp",
            "target_completed_steps": 1000,
        },
    ),
}

app = modal.App(APP_NAME, tags={**MODAL_TAGS, "experiment": "unclassified"})


canonical_index_path = Path(__file__).resolve().parents[1] / (
    "mvtracker/datasets/kubric_metadata_index.py"
)
dataset_image = _dependency_image().add_local_file(
    str(canonical_index_path),
    remote_path="/opt/mvtracker/mvtracker/datasets/kubric_metadata_index.py",
    copy=True,
)
_dataset_stage_options = {
    "volumes": {str(DATA_ROOT): data_volume.with_mount_options(read_only=True)},
    "cpu": 4,
    "memory": 8192,
    "timeout": 12 * 60 * 60,
}
dataset_image = dataset_image.run_function(
    install_dataset_base,
    **_dataset_stage_options,
    kwargs={"dataset_version": DATASET_IMAGE_VERSION},
)
for _archive_name, _scene_start, _scene_end in TRAIN_ARCHIVE_STAGES:
    _archive_record = next(
        item for item in PINNED_MVKUBRIC_ARCHIVES if item["filename"] == _archive_name
    )
    dataset_image = dataset_image.run_function(
        install_mvkubric_archive,
        **_dataset_stage_options,
        kwargs={
            "dataset_version": DATASET_IMAGE_VERSION,
            "archive_name": _archive_name,
            "scene_start": _scene_start,
            "scene_end": _scene_end,
            "archive_size_bytes": _archive_record["size_bytes"],
            "archive_sha256": _archive_record["sha256"],
        },
    )
dataset_image = dataset_image.run_function(
    install_mvkubric_validation_and_index,
    **_dataset_stage_options,
    kwargs={"dataset_version": DATASET_IMAGE_VERSION},
)
training_image = _source_image(dataset_image)


def _run_identity(run_name: str, commit: str) -> tuple[int, str]:
    digest = hashlib.sha256(f"{commit}:{run_name}".encode()).digest()
    return int.from_bytes(digest[:4], "big"), digest.hex()[:12]


def _profile_dataset_image_cpu() -> dict:
    from mvtracker.profiling.modal_continual_data import profile_encoded_loader

    profiles = {}
    for source in ("diegesis", "mvkubric"):
        profiles[source] = {
            "cold": profile_encoded_loader(
                Path(DATASET_IMAGE_ROOT) / "datasets",
                source=source,
                warmup=0,
                measured=4,
                workers=0,
                use_cuda=False,
            ),
            "warm": profile_encoded_loader(
                Path(DATASET_IMAGE_ROOT) / "datasets",
                source=source,
                warmup=LOADER_PROFILE_WARMUP,
                measured=LOADER_PROFILE_MEASURED,
                workers=0,
                use_cuda=False,
            ),
        }
    return {
        "dataset_image_root": DATASET_IMAGE_ROOT,
        "dataset_image_version": DATASET_IMAGE_VERSION,
        "profiles": profiles,
    }


def _profile_h100_mvkubric_loader() -> dict:
    from mvtracker.profiling.modal_continual_data import profile_encoded_loader

    scene_ids = list(TRAIN_SCENES[:25])
    profile = profile_encoded_loader(
        Path(DATASET_IMAGE_ROOT) / "datasets",
        source="mvkubric",
        warmup=H100_LOADER_PROFILE_WARMUP,
        measured=H100_LOADER_PROFILE_MEASURED,
        workers=8,
        use_cuda=True,
        mvkubric_scene_ids=scene_ids,
    )
    return {
        "profiles": {"mvkubric": profile},
        "staging": {"scene_ids": scene_ids, "local_data_root": DATASET_IMAGE_ROOT},
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
        job_type="dataset-image-cpu-validation",
        tags=["modal", "dataset-image", "cpu", "cold-warm-loader"],
        config={
            "source_commit": _source_commit(),
            "dataset_image_version": DATASET_IMAGE_VERSION,
            **PROFILE_TAGS,
        },
    )
    result = _profile_dataset_image_cpu()
    run.summary.update(_profile_summary(result))
    run.finish()
    return result


@app.function(
    image=training_image,
    secrets=[wandb_secret],
    gpu="H100!",
    cpu=32,
    memory=65536,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
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
        str(RUN_ROOT): run_volume,
    },
    gpu=GPU_REQUEST,
    cpu=32,
    memory=65536,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    timeout=6 * 60 * 60,
    max_containers=MAX_CONTAINERS,
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
    manifest = {
        "mode": mode,
        "run_name": run_name,
        "source_commit": commit,
        "dataset_image_version": DATASET_IMAGE_VERSION,
        "phases": [dict(phase) for phase in EXPERIMENT_PHASES[mode]],
        "gpu": GPU_REQUEST,
        "max_containers": MAX_CONTAINERS,
        "master_seed": seed,
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "wandb_group": WANDB_GROUP,
        "wandb_run_id": wandb_run_id,
        "materialize_whole_step": materialize_whole_step,
        "modal_tags": MODAL_TAGS,
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
                Path(DATASET_IMAGE_ROOT) / "checkpoints/mvtracker_200000_june2025.pth"
            ),
            "MVTRACKER_DATA_ROOT": DATASET_IMAGE_ROOT,
            "MVTRACKER_MVKUBRIC_INDEX_ROOT": str(
                Path(DATASET_IMAGE_ROOT)
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
    return f"gt-depth-replay-{mode}-ddp2-h100-{commit[:8]}-{timestamp}"


def _prepare_launch(mode: str, run_name: str, confirm_main: bool) -> str:
    require_main_confirmation(mode, confirm_main)
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers()
    selected = run_name or _default_run_name(mode, commit)
    validate_run_name(selected)
    app.set_tags({**MODAL_TAGS, "experiment": selected, "gpu": "h100x2"})
    return selected


@app.local_entrypoint(name="build-dataset-image")
def build_dataset_image() -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    app.set_tags({**PROFILE_TAGS, "experiment": "dataset-image-v2", "gpu": "cpu"})
    with modal.enable_output():
        dataset_image.build(app)
    print(f"DATASET_IMAGE {dataset_image.object_id}")


@app.local_entrypoint(name="profile-cpu-loader")
def profile_cpu_loader() -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    app.set_tags({**PROFILE_TAGS, "experiment": "dataset-image-cpu", "gpu": "cpu"})
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
    print(json.dumps(train_remote.remote("smoke", selected, ""), indent=2))


@app.local_entrypoint(name="smoke10")
def smoke10(
    run_name: str = "",
    materialize_whole_step: bool = False,
    seed: int = 0,
) -> None:
    selected = _prepare_launch("smoke10", run_name, confirm_main=False)
    print(
        json.dumps(
            train_remote.remote(
                "smoke10", selected, "", materialize_whole_step, seed
            ),
            indent=2,
        )
    )


@app.local_entrypoint(name="train")
def train(run_name: str = "", confirm_main: bool = False) -> None:
    selected = _prepare_launch("main", run_name, confirm_main)
    print(json.dumps(train_remote.remote("main", selected, MAIN_CONFIRMATION), indent=2))
