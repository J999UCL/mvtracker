"""Modal setup and two-H100 launcher for the continual-training experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import modal

from modal_training_profile import (
    DATA_ROOT,
    RUN_ROOT,
    _runtime_image,
    _source_commit,
    data_volume,
    hf_secret,
    run_volume,
    wandb_secret,
)

from mvtracker.profiling.modal_continual_training import (
    CONTINUAL_RUN_SUBDIR,
    EPHEMERAL_DISK_MIB,
    GPU_REQUEST,
    LOCAL_DATA_ROOT,
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


APP_NAME = "jeet-mvtracker-continual-training"
SOURCE_ROOT = Path("/opt/mvtracker")
CHECKPOINT = DATA_ROOT / "checkpoints/mvtracker_200000_june2025.pth"
BUNDLE_MANIFEST = DATA_ROOT / "bundles/continual-training-data-manifest.json"
LOADER_PROFILE_WARMUP = 4
LOADER_PROFILE_MEASURED = 32
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
    "main": (
        {
            "name": "main",
            "config": "diegesis_mvkubric_gt_ddp",
            "target_completed_steps": 1000,
        },
    ),
}

app = modal.App(APP_NAME, tags={**MODAL_TAGS, "experiment": "unclassified"})
image = _runtime_image()


def _run_identity(run_name: str, commit: str) -> tuple[int, str]:
    digest = hashlib.sha256(f"{commit}:{run_name}".encode()).digest()
    return int.from_bytes(digest[:4], "big"), digest.hex()[:12]


@app.function(
    image=image,
    secrets=[hf_secret, wandb_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=16,
    memory=32768,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    timeout=24 * 60 * 60,
    max_containers=MAX_CONTAINERS,
    include_source=False,
)
def setup_training_data_remote() -> dict:
    import wandb

    from mvtracker.profiling.modal_continual_data import (
        materialize_continual_training_data,
    )

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        job_type="data-setup",
        tags=["modal", "data-setup", "gt-depth-replay-v1"],
        config={"source_commit": _source_commit(), **MODAL_TAGS},
    )
    manifest = materialize_continual_training_data(DATA_ROOT)
    data_volume.commit()
    run.summary.update(
        {
            "mvkubric_scenes": manifest["mvkubric"]["scene_count"],
            "checkpoint_sha256": manifest["checkpoint"]["sha256"],
        }
    )
    run.finish()
    return manifest


@app.function(
    image=image,
    secrets=[hf_secret, wandb_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=16,
    memory=32768,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    timeout=24 * 60 * 60,
    max_containers=MAX_CONTAINERS,
    include_source=False,
)
def prepare_training_bundle_remote() -> dict:
    """Package already-materialized data for local-SSD extraction."""
    import wandb

    from mvtracker.profiling.modal_continual_data import (
        prepare_continual_training_bundle,
    )

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        job_type="data-bundle",
        tags=["modal", "data-bundle", "local-ssd", "gt-depth-replay-v1"],
        config={"source_commit": _source_commit(), **PROFILE_TAGS},
    )
    local_archive = Path("/tmp/continual-training-data.tar")
    destination = DATA_ROOT / "bundles/continual-training-data.tar"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        manifest = prepare_continual_training_bundle(
            DATA_ROOT, bundle_path=destination, reuse_only=True
        )
    except RuntimeError as error:
        if str(error) != "no compatible continual-training bundle is available":
            raise
        if local_archive.exists():
            local_archive.unlink()
        manifest = prepare_continual_training_bundle(DATA_ROOT, bundle_path=local_archive)
        copy_started = time.perf_counter()
        shutil.copyfile(local_archive, destination)
        copy_seconds = time.perf_counter() - copy_started
        manifest["archive"]["relative_path"] = "bundles/continual-training-data.tar"
        manifest.setdefault("timing", {})["copy_seconds"] = copy_seconds
        manifest["timing"]["total_seconds"] = (
            manifest["timing"].get("build_seconds", 0.0) + copy_seconds
        )
        manifest["elapsed_seconds"] = manifest["timing"]["total_seconds"]
    (DATA_ROOT / "bundles/continual-training-data-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    data_volume.commit()
    archive = manifest["archive"]
    run.summary.update(
        {
            "bundle/archive_bytes": archive["size_bytes"],
            "bundle/archive_sha256": archive["sha256"],
            "bundle/elapsed_seconds": manifest["elapsed_seconds"],
            "bundle/extracted_source_bytes": manifest["mvkubric"]["extracted"].get("size_bytes", 0),
            "bundle/index_bytes": manifest["mvkubric_index"]["size_bytes"],
        }
    )
    run.finish()
    return manifest


def _profile_loader(use_cuda: bool) -> dict:
    from mvtracker.profiling.modal_continual_data import (
        extract_continual_training_bundle,
        profile_encoded_loader,
    )

    staging = extract_continual_training_bundle(
        DATA_ROOT,
        local_data_root=Path(LOCAL_DATA_ROOT),
    )
    profiles = {}
    for source in ("diegesis",) if use_cuda else ("diegesis", "mvkubric"):
        profiles[source] = profile_encoded_loader(
            Path(LOCAL_DATA_ROOT) / "datasets",
            source=source,
            warmup=LOADER_PROFILE_WARMUP,
            measured=LOADER_PROFILE_MEASURED,
            workers=8 if use_cuda else 0,
            use_cuda=use_cuda,
        )
    return {"profiles": profiles, "staging": staging}


def _profile_summary(result: dict) -> dict:
    summary = {
        "staging/elapsed_seconds": result["staging"]["elapsed_seconds"],
        "staging/archive_bytes": result["staging"]["archive_size_bytes"],
        "staging/extracted_bytes": result["staging"]["extracted_size_bytes"],
    }
    for source, profile in result["profiles"].items():
        prefix = f"loader/{source}"
        summary.update(
            {
                f"{prefix}/warmup": profile["warmup"],
                f"{prefix}/measured": profile["measured"],
                f"{prefix}/samples_per_second": profile["samples_per_second"],
                f"{prefix}/sample_seconds_median": profile["sample_seconds_median"],
                f"{prefix}/sample_seconds_p95": profile["sample_seconds_p95"],
            }
        )
    return summary


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume.with_mount_options(read_only=True)},
    cpu=32,
    memory=65536,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
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
        job_type="encoded-loader-cpu",
        tags=["modal", "encoded-loader", "local-ssd", "cpu"],
        config={"source_commit": _source_commit(), **PROFILE_TAGS},
    )
    result = _profile_loader(use_cuda=False)
    run.summary.update(_profile_summary(result))
    run.finish()
    return result


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume.with_mount_options(read_only=True)},
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
    result = _profile_loader(use_cuda=True)
    run.summary.update(_profile_summary(result))
    run.finish()
    return result


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
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
def train_remote(mode: str, run_name: str, confirmation: str = "") -> dict:
    if mode not in EXPERIMENT_PHASES:
        raise ValueError("mode must be smoke or main")
    require_remote_main_confirmation(mode, confirmation)
    validate_run_name(run_name)
    commit = _source_commit()
    seed, wandb_run_id = _run_identity(run_name, commit)
    run_dir = RUN_ROOT / CONTINUAL_RUN_SUBDIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "modal-run-manifest.json"
    manifest = {
        "mode": mode,
        "run_name": run_name,
        "source_commit": commit,
        "phases": [dict(phase) for phase in EXPERIMENT_PHASES[mode]],
        "gpu": GPU_REQUEST,
        "max_containers": MAX_CONTAINERS,
        "master_seed": seed,
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "wandb_group": WANDB_GROUP,
        "wandb_run_id": wandb_run_id,
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
    import wandb

    from mvtracker.profiling.modal_continual_data import extract_continual_training_bundle

    staging_run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        job_type="local-ssd-staging",
        tags=["modal", "local-ssd", "staging", "gt-depth-replay-v1"],
        config={"source_commit": commit, "run_name": run_name, **MODAL_TAGS},
    )
    try:
        staging = extract_continual_training_bundle(
            DATA_ROOT,
            local_data_root=Path(LOCAL_DATA_ROOT),
        )
        staging_run.summary.update(
            {
                "staging/archive_bytes": staging["archive_size_bytes"],
                "staging/extracted_bytes": staging["extracted_size_bytes"],
                "staging/elapsed_seconds": staging["elapsed_seconds"],
            }
        )
    finally:
        staging_run.finish()
    stage_manifest_path = run_dir / "local-ssd-staging-manifest.json"
    stage_manifest = {
        "bundle_manifest": str(BUNDLE_MANIFEST),
        "local_data_root": staging["local_data_root"],
        "archive_size_bytes": staging["archive_size_bytes"],
        "extracted_size_bytes": staging["extracted_size_bytes"],
        "elapsed_seconds": staging["elapsed_seconds"],
        "mvkubric_index": staging["mvkubric_index"],
    }
    stage_manifest_path.write_text(
        json.dumps(stage_manifest, indent=2) + "\n", encoding="utf-8"
    )
    run_volume.commit()
    environment.update(
        {
            "MVTRACKER_TRAINING_RUN_DIR": str(run_dir),
            "MVTRACKER_TRAINING_CHECKPOINT": str(
                Path(LOCAL_DATA_ROOT) / "checkpoints/mvtracker_200000_june2025.pth"
            ),
            "MVTRACKER_DATA_ROOT": LOCAL_DATA_ROOT,
            "MVTRACKER_MVKUBRIC_INDEX_ROOT": str(
                Path(LOCAL_DATA_ROOT)
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


@app.local_entrypoint(name="setup-data")
def setup_data() -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    app.set_tags({**MODAL_TAGS, "experiment": "data-setup", "gpu": "cpu"})
    print(json.dumps(setup_training_data_remote.remote(), indent=2))


@app.local_entrypoint(name="prepare-bundle")
def prepare_bundle() -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    app.set_tags({**PROFILE_TAGS, "experiment": "data-bundle", "gpu": "cpu"})
    print(json.dumps(prepare_training_bundle_remote.remote(), indent=2))


@app.local_entrypoint(name="profile-cpu-loader")
def profile_cpu_loader() -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    app.set_tags({**PROFILE_TAGS, "experiment": "encoded-loader-cpu", "gpu": "cpu"})
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


@app.local_entrypoint(name="train")
def train(run_name: str = "", confirm_main: bool = False) -> None:
    selected = _prepare_launch("main", run_name, confirm_main)
    print(json.dumps(train_remote.remote("main", selected, MAIN_CONFIRMATION), indent=2))
