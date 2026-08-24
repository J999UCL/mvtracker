"""Modal setup and two-H200 launcher for the continual-training experiment."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

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
    SYN4D_MAIN_CONFIRMATION,
    preflight_active_containers,
    require_main_confirmation,
    require_pushed_main_commit,
    require_remote_main_confirmation,
    validate_run_name,
)
from mvtracker.profiling.modal_syn4d_split import (
    TRAIN_ENVIRONMENTS,
    VALIDATION_ENVIRONMENTS,
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
    "syn4d_smoke1": (
        {
            "name": "syn4d_smoke1",
            "config": "diegesis_syn4d_mvkubric_gt_ddp_smoke1",
            "target_completed_steps": 1,
        },
    ),
    "syn4d_main": (
        {
            "name": "syn4d_main",
            "config": "diegesis_syn4d_mvkubric_gt_ddp",
            "target_completed_steps": 2000,
        },
    ),
}
RECIPE_ROOT = RUN_ROOT / "training-recipes"
RECIPE_SMOKE_GPU_REQUEST = "H100:2"


def _run_logged_command(command, *, cwd, environment, log_path, label):
    """Stream subprocess output and emit a heartbeat while a phase is quiet."""
    started = time.monotonic()
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(10):
            print(
                f"[{label}] heartbeat elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    monitor = threading.Thread(target=heartbeat, daemon=True)
    monitor.start()
    try:
        with Path(log_path).open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
    finally:
        stopped.set()
        monitor.join()
    print(
        f"[{label}] complete return_code={return_code} "
        f"elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )
    return return_code

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
    cpu=16,
    memory=65536,
    timeout=6 * 60 * 60,
    max_containers=1,
    retries=0,
    include_source=False,
)
def plan_recipe_remote(recipe_name: str, step_count: int = 2000) -> dict:
    """Plan the mixed-source recipe using metadata only on sixteen CPU cores."""
    from types import SimpleNamespace

    import wandb
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from mvtracker.cli.train import _build_training_dataset
    from mvtracker.datasets.kubric_dali_dataset import DaliKubricRecipePlanner
    from mvtracker.datasets.kubric_multiview_dataset import KubricMultiViewDataset
    from mvtracker.datasets.mixed_source_schedule import BalancedMixedSourceSchedule
    from mvtracker.datasets.training_recipe import plan_training_recipe_parallel

    validate_run_name(recipe_name)
    seed = 72
    output_dir = RECIPE_ROOT / recipe_name
    local_output_dir = Path("/tmp/mvtracker-training-recipes") / recipe_name
    print(
        "recipe startup "
        f"commit={_source_commit()} cpus=16 steps={step_count} "
        f"data_root={DATA_VOLUME_ROOT} local_output={local_output_dir} "
        f"volume_output={output_dir}",
        flush=True,
    )
    os.environ.update(
        {
            "MVTRACKER_TRAINING_RUN_DIR": str(output_dir),
            "MVTRACKER_TRAINING_CHECKPOINT": str(
                Path(DATA_VOLUME_ROOT) / "checkpoints/mvtracker_200000_june2025.pth"
            ),
            "MVTRACKER_TRAINING_SEED": str(seed),
            "MVTRACKER_WANDB_RUN_NAME": recipe_name,
            "MVTRACKER_WANDB_RUN_ID": recipe_name,
        }
    )
    phase = {"name": "config"}
    heartbeat_stop = threading.Event()

    def heartbeat():
        started = time.monotonic()
        while not heartbeat_stop.wait(10):
            print(
                f"recipe heartbeat phase={phase['name']} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )

    heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        with initialize_config_dir(config_dir=str(SOURCE_ROOT / "configs"), version_base="1.3"):
            cfg = compose(
                config_name="train.yaml",
                overrides=[
                    "+experiment=diegesis_syn4d_mvkubric_gt_ddp",
                    f"trainer.num_steps={int(step_count)}",
                    "datasets.train.recipe_path=null",
                    "datasets.train.force_gt_depth=false",
                    "augmentations.variable_depth_type=true",
                ],
            )
        phase["name"] = "scene_inventory"
        diegesis_source = cfg.datasets.train.sources.diegesis
        diegesis_cache = (
            Path(diegesis_source.root) / "TAPVid3D_MVTracker_cache" / "train"
        )
        held_out_diegesis = set(cfg.datasets.eval.sources.diegesis.include_scene_ids)
        diegesis_source.include_scene_ids = sorted(
            path.name
            for path in diegesis_cache.iterdir()
            if path.is_dir() and path.name not in held_out_diegesis
        )
        print(
            "recipe phase=scene_inventory "
            f"diegesis_train={len(diegesis_source.include_scene_ids)} "
            f"diegesis_held_out={sorted(held_out_diegesis)}",
            flush=True,
        )
        phase["name"] = "dataset_construction"
        fabric = SimpleNamespace(world_size=2, global_rank=0)
        source_pattern = tuple(cfg.datasets.train.source_schedule)
        datasets = {}
        for source, source_cfg in cfg.datasets.train.sources.items():
            print(f"recipe phase=dataset_construction source={source}", flush=True)
            if source != "mvkubric":
                datasets[source] = _build_training_dataset(
                    source_cfg.name,
                    source_cfg.root,
                    cfg,
                    fabric,
                    source_cfg,
                )
                continue
            kwargs = KubricMultiViewDataset.from_name(
                source_cfg.name,
                source_cfg.root,
                cfg,
                fabric,
                just_return_kwargs=True,
                include_scene_ids=source_cfg.get("include_scene_ids"),
                exclude_scene_ids=source_cfg.get("exclude_scene_ids", ()),
            )
            kwargs.pop("data_root")
            kwargs.pop("metadata_index_root")
            datasets[source] = DaliKubricRecipePlanner(
                **kwargs,
                webdataset_root=cfg.datasets.train.mvkubric_webdataset_root,
                webdataset_split="train",
                stream_world_size=2,
                stream_seed=seed,
                stream_include_scene_ids=source_cfg.get("include_scene_ids"),
            )
            phase["name"] = "mvkubric_metadata_preload"
            datasets[source].preload_recipe_metadata(workers=16)
        schedule = BalancedMixedSourceSchedule(
            {source: dataset.real_len for source, dataset in datasets.items()},
            source_pattern,
            world_size=2,
            master_seed=seed,
        )
        phase["name"] = "sample_planning"
        heartbeat_stop.set()
        heartbeat_thread.join()
        summary = plan_training_recipe_parallel(
            local_output_dir,
            datasets=datasets,
            schedule=schedule,
            step_count=int(step_count),
            manifest={
                "source_commit": _source_commit(),
                "seed": seed,
                "config": OmegaConf.to_container(cfg, resolve=True),
                "scene_lists": {
                    source: list(dataset.seq_names)
                    for source, dataset in datasets.items()
                },
            },
            worker_count=16,
            block_steps=25,
            heartbeat_seconds=10,
        )
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join()
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group="training-recipe-planning",
        job_type="recipe-planning",
        name=recipe_name,
        tags=["modal", "recipe", "cpu16", "training"],
        config={
            "source_commit": _source_commit(),
            "step_count": int(step_count),
            "cpu_cores": 16,
            **MODAL_TAGS,
        },
    )
    run.summary.update(summary)
    run.finish()
    print(
        f"recipe phase=publish source={local_output_dir} destination={output_dir}",
        flush=True,
    )
    shutil.copytree(local_output_dir, output_dir)
    run_volume.commit()
    print(f"recipe volume commit complete output={output_dir}", flush=True)
    return summary


@app.function(
    image=training_image,
    secrets=[wandb_secret],
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
        str(RUN_ROOT): run_volume,
    },
    gpu=RECIPE_SMOKE_GPU_REQUEST,
    cpu=32,
    memory=(
        TRAIN_MEMORY_REQUEST_MIB,
        TRAIN_MEMORY_LIMIT_MIB,
    ),
    timeout=6 * 60 * 60,
    max_containers=1,
    retries=0,
    include_source=False,
)
def recipe_smoke20_remote(run_name: str, recipe_name: str) -> dict:
    """Consume the first twenty recipe steps with GT depth on two H100s."""
    validate_run_name(run_name)
    validate_run_name(recipe_name)
    commit = _source_commit()
    recipe_path = RECIPE_ROOT / recipe_name
    run_dir = RUN_ROOT / CONTINUAL_RUN_SUBDIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    _, wandb_run_id = _run_identity(run_name, commit)
    recipe_manifest = json.loads(
        (recipe_path / "manifest.json").read_text(encoding="utf-8")
    )
    seed = int(recipe_manifest["seed"])
    environment = os.environ.copy()
    environment.update(
        {
            "MVTRACKER_TRAINING_RUN_DIR": str(run_dir),
            "MVTRACKER_TRAINING_RECIPE": str(recipe_path),
            "MVTRACKER_TRAINING_CHECKPOINT": str(
                Path(DATA_VOLUME_ROOT) / "checkpoints/mvtracker_200000_june2025.pth"
            ),
            "MVTRACKER_DATA_ROOT": DATA_VOLUME_ROOT,
            "MVTRACKER_TRAINING_SEED": str(seed),
            "MVTRACKER_WANDB_RUN_NAME": run_name,
            "MVTRACKER_WANDB_RUN_ID": wandb_run_id,
            "WANDB_ENTITY": WANDB_ENTITY,
            "WANDB_PROJECT": WANDB_PROJECT,
            "WANDB_RUN_GROUP": "recipe-gt-smoke20",
            "WANDB_RUN_ID": wandb_run_id,
            "WANDB_RESUME": "allow",
            "PYTHONUNBUFFERED": "1",
        }
    )
    manifest = {
        "mode": "recipe_smoke20",
        "run_name": run_name,
        "recipe": recipe_name,
        "source_commit": commit,
        "gpu": RECIPE_SMOKE_GPU_REQUEST,
        "steps": 20,
        "force_gt_depth": True,
        "validation": False,
        "modal_tags": MODAL_TAGS,
    }
    (run_dir / "modal-run-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    run_volume.commit()
    print(
        "recipe smoke20 startup "
        f"commit={commit} gpu={RECIPE_SMOKE_GPU_REQUEST} run={run_name} "
        f"recipe={recipe_path} steps=20 force_gt_depth=true validation=false",
        flush=True,
    )
    command = [
        sys.executable,
        "-m",
        "mvtracker.cli.train",
        "+experiment=diegesis_syn4d_mvkubric_recipe_gt_ddp_smoke20",
    ]
    return_code = _run_logged_command(
        command,
        cwd=SOURCE_ROOT,
        environment=environment,
        log_path=run_dir / "training.log",
        label="recipe-smoke20",
    )
    run_volume.commit()
    if return_code != 0:
        raise RuntimeError(
            f"recipe smoke20 exited {return_code}; see {run_dir / 'training.log'}"
        )
    return manifest


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
    resume_existing: bool = False,
) -> dict:
    if mode not in EXPERIMENT_PHASES:
        raise ValueError("unsupported training mode")
    require_remote_main_confirmation(mode, confirmation)
    validate_run_name(run_name)
    commit = _source_commit()
    run_dir = RUN_ROOT / CONTINUAL_RUN_SUBDIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "modal-run-manifest.json"
    existing_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )
    if resume_existing:
        if existing_manifest is None:
            raise RuntimeError("resume requires an existing run manifest")
        seed = int(existing_manifest["master_seed"])
        wandb_run_id = str(existing_manifest["wandb_run_id"])
        if seed_override and int(seed_override) != seed:
            raise RuntimeError("resume seed does not match the existing run")
    else:
        derived_seed, wandb_run_id = _run_identity(run_name, commit)
        seed = seed_override or derived_seed
    data_inventory = json.loads(
        (Path(DATA_VOLUME_ROOT) / "direct-volume-data-manifest.json").read_text()
    )
    syn4d_inventory = None
    if mode.startswith("syn4d_"):
        syn4d_root = Path(DATA_VOLUME_ROOT) / "datasets/syn4d-mvtracker"
        syn4d_inventory = {
            "train": [
                f"{environment}__{sequence}"
                for environment, sequence in TRAIN_ENVIRONMENTS
            ],
            "validation": [
                f"{environment}__{sequence}"
                for environment, sequence in VALIDATION_ENVIRONMENTS
            ],
        }
        for split, sequences in syn4d_inventory.items():
            missing = [
                sequence
                for sequence in sequences
                if not (syn4d_root / split / sequence).is_dir()
            ]
            if missing:
                raise RuntimeError(
                    f"Syn4D {split} cache is missing: {', '.join(missing)}"
                )
    wandb_group = (
        "gt-depth-replay-syn4d-v2"
        if mode.startswith("syn4d_")
        else WANDB_GROUP
    )
    run_data_inventory = {
        "train_scene_count": data_inventory["train_scene_count"],
        "validation_scene_count": data_inventory["validation_scene_count"],
        "validation_scene_ids": data_inventory["validation_scene_ids"],
    }
    if syn4d_inventory is not None:
        run_data_inventory["syn4d_sequences"] = syn4d_inventory
    modal_tags = PROFILE_TAGS if mode == "memory_profile" else MODAL_TAGS
    manifest = {
        "mode": mode,
        "run_name": run_name,
        "source_commit": commit,
        "data_volume": "jeet-mvtracker-data-v2",
        "data_layout_version": DATA_LAYOUT_VERSION,
        "data_inventory": run_data_inventory,
        "phases": [dict(phase) for phase in EXPERIMENT_PHASES[mode]],
        "gpu": GPU_REQUEST,
        "max_containers": MAX_CONTAINERS,
        "master_seed": seed,
        "wandb_entity": WANDB_ENTITY,
        "wandb_project": WANDB_PROJECT,
        "wandb_group": wandb_group,
        "wandb_run_id": wandb_run_id,
        "materialize_whole_step": materialize_whole_step,
        "modal_tags": modal_tags,
    }
    if existing_manifest is not None:
        if resume_existing:
            ignored = {"source_commit", "resume_source_commits"}
            existing_contract = {
                key: value
                for key, value in existing_manifest.items()
                if key not in ignored
            }
            current_contract = {
                key: value for key, value in manifest.items() if key not in ignored
            }
            if existing_contract != current_contract:
                raise RuntimeError(
                    "existing run contract does not match this resume"
                )
            source_commits = list(
                existing_manifest.get(
                    "resume_source_commits",
                    [existing_manifest["source_commit"]],
                )
            )
            if commit not in source_commits:
                source_commits.append(commit)
            existing_manifest["resume_source_commits"] = source_commits
            manifest = existing_manifest
            manifest_path.write_text(
                json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
            )
            run_volume.commit()
        elif existing_manifest != manifest:
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
            "WANDB_RUN_GROUP": wandb_group,
            "WANDB_RUN_ID": wandb_run_id,
            "WANDB_RESUME": "allow",
        }
    )
    log_path = run_dir / "training.log"
    for phase in EXPERIMENT_PHASES[mode]:
        print(
            f"training phase start name={phase['name']} "
            f"target={phase['target_completed_steps']}",
            flush=True,
        )
        command = [
            sys.executable,
            "-m",
            "mvtracker.cli.train",
            f"+experiment={phase['config']}",
            f"datasets.train.materialize_whole_step={str(materialize_whole_step).lower()}",
        ]
        if resume_existing:
            command.append("modes.validate_at_start=false")
        return_code = _run_logged_command(
            command,
            cwd=SOURCE_ROOT,
            environment=environment,
            log_path=log_path,
            label=f"training-{phase['name']}",
        )
        run_volume.commit()
        if return_code != 0:
            raise RuntimeError(
                f"training phase {phase['name']} exited {return_code}; "
                f"see {log_path}"
            )
    return manifest


def _default_run_name(mode: str, commit: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if mode == "production_smoke10":
        return f"gt-replay-prod-smoke10-{commit[:8]}-{timestamp}"
    if mode.startswith("syn4d_"):
        return f"gt-replay-syn4d-{mode}-ddp2-h200-{commit[:8]}-{timestamp}"
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
    resume_existing: bool = False,
) -> None:
    deployed_training = modal.Function.from_name(APP_NAME, "train_remote")
    call = deployed_training.spawn(
        mode,
        run_name,
        confirmation,
        materialize_whole_step,
        seed,
        resume_existing,
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


@app.local_entrypoint(name="plan-recipe")
def plan_recipe(recipe_name: str, step_count: int = 2000) -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=1)
    validate_run_name(recipe_name)
    app.set_tags(
        {**MODAL_TAGS, "experiment": recipe_name, "gpu": "cpu", "cpu": "16"}
    )
    deployed = modal.Function.from_name(APP_NAME, "plan_recipe_remote")
    call = deployed.spawn(recipe_name, step_count)
    print(json.dumps({"recipe_name": recipe_name, "function_call_id": call.object_id}, indent=2))


@app.local_entrypoint(name="recipe-smoke20")
def recipe_smoke20(run_name: str, recipe_name: str) -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers()
    validate_run_name(run_name)
    validate_run_name(recipe_name)
    app.set_tags(
        {**MODAL_TAGS, "experiment": run_name, "gpu": "h100x2"}
    )
    deployed = modal.Function.from_name(APP_NAME, "recipe_smoke20_remote")
    call = deployed.spawn(run_name, recipe_name)
    print(json.dumps({"run_name": run_name, "function_call_id": call.object_id}, indent=2))


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
def train(
    run_name: str = "",
    confirm_main: bool = False,
    resume_existing: bool = False,
) -> None:
    if resume_existing and not run_name:
        raise RuntimeError("--resume-existing requires --run-name")
    selected = _prepare_launch("main", run_name, confirm_main)
    _spawn_training(
        "main",
        selected,
        MAIN_CONFIRMATION,
        resume_existing=resume_existing,
    )


@app.local_entrypoint(name="syn4d-smoke1")
def syn4d_smoke1(run_name: str = "", seed: int = 0) -> None:
    selected = _prepare_launch("syn4d_smoke1", run_name, confirm_main=False)
    _spawn_training("syn4d_smoke1", selected, seed=seed)


@app.local_entrypoint(name="syn4d-train")
def syn4d_train(run_name: str = "", confirm_main: bool = False) -> None:
    selected = _prepare_launch("syn4d_main", run_name, confirm_main)
    _spawn_training(
        "syn4d_main",
        selected,
        SYN4D_MAIN_CONFIRMATION,
    )
