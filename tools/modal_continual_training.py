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
    GPU_REQUEST,
    MAIN_CONFIRMATION,
    MAX_CONTAINERS,
    MODAL_TAGS,
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
    ephemeral_disk=64 * 1024,
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
    secrets=[wandb_secret],
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
        str(RUN_ROOT): run_volume,
    },
    gpu=GPU_REQUEST,
    cpu=32,
    memory=65536,
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
    run_dir = RUN_ROOT / "continual-training" / run_name
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
    environment.update(
        {
            "MVTRACKER_TRAINING_RUN_DIR": str(run_dir),
            "MVTRACKER_TRAINING_CHECKPOINT": str(CHECKPOINT),
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


@app.local_entrypoint(name="smoke")
def smoke(run_name: str = "") -> None:
    selected = _prepare_launch("smoke", run_name, confirm_main=False)
    print(json.dumps(train_remote.remote("smoke", selected, ""), indent=2))


@app.local_entrypoint(name="train")
def train(run_name: str = "", confirm_main: bool = False) -> None:
    selected = _prepare_launch("main", run_name, confirm_main)
    print(json.dumps(train_remote.remote("main", selected, MAIN_CONFIRMATION), indent=2))
