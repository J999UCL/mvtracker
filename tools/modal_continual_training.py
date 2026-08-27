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
    hf_secret,
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
RECIPE_PLANNER_CPUS = 32
RECIPE_PLANNER_DISK_MIB = 512 * 1024
RECIPE_METADATA_WORKERS = 16
RECIPE_METADATA_COPY_WORKERS = 8
MVKUBRIC_METADATA_SIDECAR = (
    Path(DATA_VOLUME_ROOT)
    / "datasets/kubric-multiview-webdataset/train/recipe-metadata"
)
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
RECIPE_SMOKE_GPU_REQUEST = GPU_REQUEST


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
    process = None
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
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        stopped.set()
        monitor.join()
    print(
        f"[{label}] complete return_code={return_code} "
        f"elapsed={time.monotonic() - started:.1f}s",
        flush=True,
    )
    return return_code


def _start_logged_process(command, *, cwd, environment, log_path, label):
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def pump():
        with Path(log_path).open("a", encoding="utf-8") as log:
            for line in process.stdout:
                print(f"[{label}] {line}", end="", flush=True)
                log.write(line)
                log.flush()

    thread = threading.Thread(target=pump, daemon=True)
    thread.start()
    return process, thread


def _terminate_logged_processes(processes) -> None:
    for process, _ in processes:
        if process.poll() is None:
            process.terminate()
    for process, thread in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        thread.join()

app = modal.App(
    APP_NAME,
    tags={**MODAL_TAGS, "experiment": "continual-training-worker"},
)


training_image = _source_image(_dependency_image())

DA3_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
da3_training_image = _source_image(
    _dependency_image()
    .pip_install(
        "xformers==0.0.31.post1",
        "addict",
        "omegaconf",
        "safetensors",
        "trimesh",
        "pycolmap",
        "imageio",
        "pillow-heif",
        "plyfile",
        "numpy==1.24.3",
        "matplotlib==3.8.3",
    )
    .run_commands(
        "python -m pip install --no-deps evo==1.37.0",
        "python -m pip install --no-deps "
        f"git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@{DA3_REVISION}"
    )
)


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
    image=da3_training_image,
    secrets=[hf_secret, wandb_secret],
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
        str(RUN_ROOT): run_volume,
    },
    gpu="H200",
    cpu=32,
    memory=(TRAIN_MEMORY_REQUEST_MIB, TRAIN_MEMORY_LIMIT_MIB),
    ephemeral_disk=512 * 1024,
    timeout=2 * 60 * 60,
    max_containers=1,
    retries=0,
    include_source=False,
)
def audit_recipe_gradients_remote(
    audit_name: str,
    source_run_name: str,
    recipe_name: str,
    optimizer_steps_csv: str = "625,688",
) -> dict:
    """Replay two anomalous optimizer steps without updating the model."""
    from types import SimpleNamespace

    import wandb

    from tools.audit_recipe_optimizer_steps import run as run_audit

    validate_run_name(audit_name)
    validate_run_name(source_run_name)
    validate_run_name(recipe_name)
    optimizer_steps = tuple(
        int(value) for value in optimizer_steps_csv.split(",") if value.strip()
    )
    if not optimizer_steps or any(step < 1 for step in optimizer_steps):
        raise ValueError("optimizer_steps_csv must contain positive integers")
    source_run = RUN_ROOT / CONTINUAL_RUN_SUBDIR / source_run_name
    output_dir = RUN_ROOT / "gradient-audits" / audit_name
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / "report.json"
    runtime_depth_root = Path("/tmp/mvtracker-gradient-audit-depth")
    evaluation_tags = {
        "owner": "jeet",
        "project": "mvtracker",
        "purpose": "evaluation",
    }
    wandb_run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group="gradient-corruption-audit",
        job_type="gradient-audit",
        name=audit_name,
        tags=["modal", "h200", "gradient-audit", "no-optimizer-step"],
        config={
            "source_commit": _source_commit(),
            "source_run": source_run_name,
            "recipe": recipe_name,
            "checkpoint_step": 500,
            "optimizer_steps": list(optimizer_steps),
            **evaluation_tags,
        },
    )
    print(
        "AUDIT event=modal_start "
        f"audit={audit_name} source_run={source_run_name} recipe={recipe_name}",
        flush=True,
    )
    try:
        report = run_audit(
            SimpleNamespace(
                run_dir=source_run,
                data_root=Path(DATA_VOLUME_ROOT),
                checkpoint=source_run / "model_000500.pth",
                recipe=RECIPE_ROOT / recipe_name,
                diegesis_root=Path(DATA_VOLUME_ROOT)
                / "datasets/diegesis-mvtracker",
                syn4d_root=Path(DATA_VOLUME_ROOT) / "datasets/syn4d-mvtracker",
                mvkubric_root=Path(DATA_VOLUME_ROOT) / "datasets",
                runtime_depth_root=runtime_depth_root,
                depth_mode="runtime",
                optimizer_steps=optimizer_steps,
                sketch_size=2048,
                sketch_seed=0,
                device="cuda",
                output=output_path,
            )
        )
    except BaseException:
        wandb_run.summary["status"] = "failed"
        wandb_run.finish(exit_code=1)
        raise
    summary = {"status": "complete", "report": str(output_path)}
    for step in report["steps"]:
        optimizer_step = int(step["optimizer_step"])
        summary[f"step_{optimizer_step}/accumulated_gradient_norm"] = float(
            step["accumulated_gradient_norm"]
        )
        summary[f"step_{optimizer_step}/would_clip"] = bool(
            step["would_clip_at_global_norm_1"]
        )
        for sample in step["samples"]:
            if sample["source"] == "diegesis" and sample["scene"] == "kitchen03":
                prefix = f"step_{optimizer_step}/kitchen03"
                summary[f"{prefix}/loss"] = float(sample["scene_losses"]["total"])
                summary[f"{prefix}/gradient_norm"] = float(sample["gradient_norm"])
                counterfactual = sample.get("counterfactual_radius_30m")
                if counterfactual is not None:
                    summary[f"{prefix}/filtered_loss"] = float(
                        counterfactual["scene_losses"]["total"]
                    )
                    summary[f"{prefix}/filtered_gradient_norm"] = float(
                        counterfactual["gradient_norm"]
                    )
                    summary[f"{prefix}/removed_tracks"] = int(
                        counterfactual["removed_tracks"]
                    )
    wandb_run.summary.update(summary)
    wandb_run.finish()
    run_volume.commit()
    print(f"AUDIT event=modal_complete report={output_path}", flush=True)
    return summary


@app.function(
    image=training_image,
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
        str(RUN_ROOT): run_volume,
    },
    gpu="T4",
    cpu=8,
    memory=32768,
    timeout=30 * 60,
    max_containers=1,
    retries=0,
    include_source=False,
)
def visualize_recipe_augmentations_remote(audit_name: str, recipe_name: str) -> dict:
    from types import SimpleNamespace

    from tools.visualize_recipe_augmentations import run

    validate_run_name(audit_name)
    validate_run_name(recipe_name)
    output = RUN_ROOT / "augmentation-audits" / audit_name
    print(
        f"AUGMENTATION_VIS event=start audit={audit_name} recipe={recipe_name}",
        flush=True,
    )
    result = run(
        SimpleNamespace(
            data_root=Path(DATA_VOLUME_ROOT),
            recipe=RECIPE_ROOT / recipe_name,
            diegesis_root=Path(DATA_VOLUME_ROOT) / "datasets/diegesis-mvtracker",
            syn4d_root=Path(DATA_VOLUME_ROOT) / "datasets/syn4d-mvtracker",
            mvkubric_root=Path(DATA_VOLUME_ROOT) / "datasets",
            output=output,
        )
    )
    run_volume.commit()
    print(f"AUGMENTATION_VIS event=complete output={output}", flush=True)
    return result


@app.function(
    image=training_image,
    secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=16,
    memory=65536,
    timeout=6 * 60 * 60,
    max_containers=1,
    retries=0,
    include_source=False,
)
def build_mvkubric_recipe_metadata_remote() -> dict:
    """Build the reusable metadata-only MV-Kubric planning sidecar."""
    import wandb

    from mvtracker.preprocessing.mvkubric_metadata_sidecar import (
        build_metadata_sidecar,
    )

    source_root = (
        Path(DATA_VOLUME_ROOT) / "datasets/kubric-multiview-webdataset/train"
    )
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group="training-recipe-planning",
        job_type="recipe-metadata-build",
        name="mvkubric-recipe-metadata",
        tags=["modal", "recipe", "metadata-sidecar", "cpu"],
        config={"source_commit": _source_commit(), **PROFILE_TAGS},
    )
    started = time.perf_counter()
    print(
        "MVKUBRIC_METADATA event=build_start "
        f"source={source_root} output={MVKUBRIC_METADATA_SIDECAR}",
        flush=True,
    )
    try:
        manifest = build_metadata_sidecar(
            source_root,
            MVKUBRIC_METADATA_SIDECAR,
            shard_count=16,
        )
        data_volume.commit()
    except BaseException:
        run.summary["status"] = "failed"
        run.finish(exit_code=1)
        raise
    elapsed = time.perf_counter() - started
    total_bytes = sum(int(shard["bytes"]) for shard in manifest["shards"])
    run.summary.update(
        {
            "elapsed_seconds": elapsed,
            "scene_count": len(manifest["scene_ids"]),
            "shard_count": len(manifest["shards"]),
            "bytes": total_bytes,
        }
    )
    run.finish()
    print(
        "MVKUBRIC_METADATA event=build_complete "
        f"scenes={len(manifest['scene_ids'])} bytes={total_bytes} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return {
        "elapsed_seconds": elapsed,
        "scene_count": len(manifest["scene_ids"]),
        "shard_count": len(manifest["shards"]),
        "bytes": total_bytes,
        "output": str(MVKUBRIC_METADATA_SIDECAR),
    }


@app.function(
    image=training_image,
    secrets=[wandb_secret],
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
        str(RUN_ROOT): run_volume,
    },
    cpu=RECIPE_PLANNER_CPUS,
    memory=65536,
    ephemeral_disk=RECIPE_PLANNER_DISK_MIB,
    timeout=6 * 60 * 60,
    max_containers=1,
    retries=0,
    include_source=False,
)
def plan_recipe_remote(
    recipe_name: str,
    step_count: int = 2000,
    world_size: int = 2,
    global_batch_size: int = 8,
) -> dict:
    """Plan the mixed-source recipe using metadata only."""
    from functools import partial
    from types import SimpleNamespace

    import wandb
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from mvtracker.cli.train import _build_training_dataset, _physical_batch_capacity
    from mvtracker.datasets.kubric_dali_dataset import DaliKubricRecipePlanner
    from mvtracker.datasets.kubric_multiview_dataset import KubricMultiViewDataset
    from mvtracker.datasets.mixed_source_schedule import BalancedMixedSourceSchedule
    from mvtracker.datasets.physical_batch_scheduler import (
        schedule_physical_batch,
        schedule_singleton_batch,
    )
    from mvtracker.datasets.training_recipe import plan_training_recipe_parallel
    from mvtracker.preprocessing.mvkubric_metadata_sidecar import (
        KubricMetadataSidecar,
    )

    validate_run_name(recipe_name)
    seed = 72
    output_dir = RECIPE_ROOT / recipe_name
    local_output_dir = Path("/tmp/mvtracker-training-recipes") / recipe_name
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group="training-recipe-planning",
        job_type="recipe-planning",
        name=recipe_name,
        tags=["modal", "recipe", f"cpu{RECIPE_PLANNER_CPUS}", "training"],
        config={
            "source_commit": _source_commit(),
            "step_count": int(step_count),
            "world_size": int(world_size),
            "global_batch_size": int(global_batch_size),
            "cpu_cores": RECIPE_PLANNER_CPUS,
            **MODAL_TAGS,
        },
    )
    print(
        "recipe startup "
        f"commit={_source_commit()} cpus={RECIPE_PLANNER_CPUS} steps={step_count} "
        f"world_size={world_size} global_batch_size={global_batch_size} "
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
                    f"trainer.lr_schedule_steps={int(step_count)}",
                    "datasets.train.recipe_path=null",
                    "datasets.train.force_gt_depth=false",
                    "datasets.train.physical_batching.max_scenes=1",
                    "datasets.train.physical_batching.rank_local=false",
                    "augmentations.variable_depth_type=true",
                ],
            )
        source_pattern = tuple(cfg.datasets.train.source_schedule)
        if global_batch_size % len(source_pattern):
            raise ValueError("global batch must divide the source schedule evenly")
        if global_batch_size % world_size:
            raise ValueError("global batch must divide evenly across DDP ranks")
        planning_world_size = global_batch_size // len(source_pattern)
        cfg.trainer.gradient_accumulation_steps = global_batch_size // world_size
        cfg.datasets.train.physical_batching.rank_count = int(world_size)
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
        fabric = SimpleNamespace(world_size=planning_world_size, global_rank=0)
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
                stream_world_size=planning_world_size,
                stream_seed=seed,
                stream_include_scene_ids=source_cfg.get("include_scene_ids"),
            )
            phase["name"] = "mvkubric_metadata_stage"
            metadata_sidecar = KubricMetadataSidecar(MVKUBRIC_METADATA_SIDECAR)
            local_metadata = Path("/tmp/mvkubric-recipe-metadata")
            metadata_stage_started = time.perf_counter()
            metadata_sidecar.stage(
                datasets[source].seq_names,
                local_metadata,
                workers=RECIPE_METADATA_COPY_WORKERS,
            )
            metadata_stage_seconds = time.perf_counter() - metadata_stage_started
            phase["name"] = "mvkubric_metadata_decode"
            metadata_decode_started = time.perf_counter()
            datasets[source]._recipe_metadata = metadata_sidecar.load_many(
                datasets[source].seq_names,
                staged_root=local_metadata,
                workers=RECIPE_METADATA_WORKERS,
            )
            metadata_decode_seconds = time.perf_counter() - metadata_decode_started
            print(
                "recipe phase=mvkubric_metadata_ready "
                f"scenes={len(datasets[source]._recipe_metadata)} "
                f"stage_seconds={metadata_stage_seconds:.1f} "
                f"decode_seconds={metadata_decode_seconds:.1f}",
                flush=True,
            )
        schedule = BalancedMixedSourceSchedule(
            {source: dataset.real_len for source, dataset in datasets.items()},
            source_pattern,
            world_size=planning_world_size,
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
            worker_count=RECIPE_PLANNER_CPUS,
            block_steps=25,
            heartbeat_seconds=10,
            physical_scheduler=partial(
                (
                    schedule_singleton_batch
                    if int(world_size) != planning_world_size
                    else schedule_physical_batch
                ),
                capacity=_physical_batch_capacity(cfg),
            ),
            output_world_size=int(world_size),
        )
        summary.update(
            mvkubric_metadata_stage_seconds=metadata_stage_seconds,
            mvkubric_metadata_decode_seconds=metadata_decode_seconds,
        )
        (local_output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except BaseException:
        run.summary["status"] = "failed"
        run.finish(exit_code=1)
        raise
    finally:
        heartbeat_stop.set()
        heartbeat_thread.join()
    try:
        print(
            f"recipe phase=publish source={local_output_dir} destination={output_dir}",
            flush=True,
        )
        shutil.copytree(local_output_dir, output_dir)
        run_volume.commit()
    except BaseException:
        run.summary["status"] = "failed"
        run.finish(exit_code=1)
        raise
    run.summary.update(summary)
    run.summary["status"] = "complete"
    run.finish()
    print(f"recipe volume commit complete output={output_dir}", flush=True)
    return summary


@app.function(
    image=training_image,
    secrets=[wandb_secret],
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
        str(RUN_ROOT): run_volume,
    },
    cpu=32,
    memory=65536,
    timeout=3 * 60 * 60,
    max_containers=1,
    retries=0,
    include_source=False,
)
def replan_syn4d_recipe_remote(source_name: str, recipe_name: str) -> dict:
    """Regenerate Syn4D logical samples without loading unchanged sources."""
    from types import SimpleNamespace

    import wandb
    from omegaconf import OmegaConf

    from mvtracker.cli.train import _build_training_dataset
    from mvtracker.datasets.mixed_source_schedule import ScheduledSampleRequest
    from mvtracker.datasets.training_recipe import replan_recipe_source

    validate_run_name(source_name)
    validate_run_name(recipe_name)
    source_dir = RECIPE_ROOT / source_name
    output_dir = RECIPE_ROOT / recipe_name
    local_output_dir = Path("/tmp/mvtracker-training-recipes") / recipe_name
    print(
        "syn4d source-replan startup "
        f"source={source_dir} output={output_dir} workers=32",
        flush=True,
    )
    source_manifest = json.loads(
        (source_dir / "manifest.json").read_text(encoding="utf-8")
    )
    cfg = OmegaConf.create(source_manifest["config"])
    cfg.datasets.syn4d_max_track_radius = 30.0
    cfg.datasets.syn4d_mmap_cache_sequences = 32
    source_cfg = cfg.datasets.train.sources.syn4d
    dataset = _build_training_dataset(
        source_cfg.name,
        source_cfg.root,
        cfg,
        SimpleNamespace(world_size=2, global_rank=0),
        source_cfg,
    )
    summary = replan_recipe_source(
        source_dir,
        local_output_dir,
        source="syn4d",
        dataset=dataset,
        request_factory=ScheduledSampleRequest,
        worker_count=32,
        heartbeat_seconds=10,
    )
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group="training-recipe-planning",
        job_type="syn4d-recipe-replan",
        name=recipe_name,
        tags=["modal", "recipe", "syn4d", "source-replan", "cpu32"],
        config={
            "source_commit": _source_commit(),
            "source_recipe": source_name,
            "recipe": recipe_name,
            **MODAL_TAGS,
        },
    )
    run.summary.update(summary)
    run.finish()
    print(
        f"syn4d source-replan publish source={local_output_dir} output={output_dir}",
        flush=True,
    )
    shutil.copytree(local_output_dir, output_dir)
    run_volume.commit()
    print(f"syn4d source-replan committed output={output_dir}", flush=True)
    return summary


@app.function(
    image=training_image,
    secrets=[wandb_secret],
    volumes={str(RUN_ROOT): run_volume},
    cpu=2,
    memory=4096,
    timeout=15 * 60,
    max_containers=1,
    retries=0,
    include_source=False,
)
def fabricate_depth_recipe_remote(source_name: str, recipe_name: str) -> dict:
    import wandb

    from mvtracker.datasets.training_recipe import derive_mixed_depth_smoke_recipe

    validate_run_name(source_name)
    validate_run_name(recipe_name)
    source = RECIPE_ROOT / source_name
    output = RECIPE_ROOT / recipe_name
    print(
        f"depth recipe fabrication source={source} output={output}",
        flush=True,
    )
    counts = derive_mixed_depth_smoke_recipe(source, output)
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group="training-recipe-planning",
        job_type="depth-recipe-fabrication",
        name=recipe_name,
        tags=["modal", "recipe", "da3", "mixed-depth-70-20-10"],
        config={"source_recipe": source_name, "recipe": recipe_name, **MODAL_TAGS},
    )
    run.summary.update({f"depth/{name}": count for name, count in counts.items()})
    run.finish()
    run_volume.commit()
    print(f"depth recipe ready counts={counts}", flush=True)
    return counts


@app.function(
    image=training_image,
    secrets=[wandb_secret],
    volumes={str(RUN_ROOT): run_volume},
    cpu=2,
    memory=4096,
    timeout=15 * 60,
    max_containers=1,
    retries=0,
    include_source=False,
)
def fabricate_singleton_recipe_remote(source_name: str, recipe_name: str) -> dict:
    import wandb

    from mvtracker.datasets.training_recipe import derive_singleton_recipe

    validate_run_name(source_name)
    validate_run_name(recipe_name)
    source = RECIPE_ROOT / source_name
    output = RECIPE_ROOT / recipe_name
    print(f"singleton recipe source={source} output={output}", flush=True)
    counts = derive_singleton_recipe(source, output)
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group="training-recipe-planning",
        job_type="singleton-recipe-fabrication",
        name=recipe_name,
        tags=["modal", "recipe", "singleton", "no-scene-batching"],
        config={"source_recipe": source_name, "recipe": recipe_name, **MODAL_TAGS},
    )
    run.summary.update({f"rank/{rank}/records": count for rank, count in counts.items()})
    run.finish()
    run_volume.commit()
    print(f"singleton recipe ready rank_counts={counts}", flush=True)
    return counts


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


def _coordinate_recipe_da3(
    *,
    workers,
    prefill_devices: tuple[int, ...],
    depth_device: int,
    runtime_root: Path,
    recipe_path: Path,
    prefill_samples: int,
    run_dir: Path,
    base_environment: dict,
    training_devices: tuple[int, ...],
    experiment_name: str,
    training_log: Path,
    manifest: dict,
) -> dict:
    prefill = runtime_root / "prefill.ready"
    prefill_started = time.monotonic()
    next_prefill_log = prefill_started + 10
    shard_markers = [
        runtime_root / f"worker-{worker_id}.prefill.ready"
        for worker_id in range(len(workers))
    ]
    while not all(marker.is_file() for marker in shard_markers):
        failed_workers = [
            worker_id
            for worker_id, (process, _) in enumerate(workers)
            if process.poll() is not None and not shard_markers[worker_id].is_file()
        ]
        if failed_workers:
            raise RuntimeError(
                f"DA3 prefill workers failed: {failed_workers}; see depth-producer-*.log"
            )
        now = time.monotonic()
        if now >= next_prefill_log:
            print(
                "DA3 prefill waiting "
                f"elapsed={now - prefill_started:.1f}s "
                f"ready_shards={sum(marker.is_file() for marker in shard_markers)}/"
                f"{len(shard_markers)}",
                flush=True,
            )
            next_prefill_log = now + 10
        time.sleep(1)

    from mvtracker.preprocessing.runtime_da3 import prefill_ready_paths

    expected_ready_samples = set(
        prefill_ready_paths(recipe_path, runtime_root, prefill_samples)
    )
    ready_samples = set(runtime_root.glob("step-*/sample-*/ready"))
    if ready_samples != expected_ready_samples:
        missing = len(expected_ready_samples - ready_samples)
        unexpected = len(ready_samples - expected_ready_samples)
        raise RuntimeError(
            f"DA3 prefill does not match the first {prefill_samples} non-GT records: "
            f"missing={missing} unexpected={unexpected}"
        )

    steady_worker_id = prefill_devices.index(depth_device)
    for worker_id, (process, thread) in enumerate(workers):
        if worker_id == steady_worker_id:
            continue
        try:
            return_code = process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"DA3 prefill worker {worker_id} did not release GPU {prefill_devices[worker_id]}"
            ) from None
        thread.join()
        if return_code != 0:
            raise RuntimeError(
                f"DA3 prefill worker {worker_id} exited {return_code}; "
                f"see {run_dir / f'depth-producer-{worker_id}.log'}"
            )
    steady_return_code = workers[steady_worker_id][0].poll()
    if steady_return_code is not None:
        raise RuntimeError(
            f"steady DA3 worker exited {steady_return_code} before handoff; "
            f"see {run_dir / f'depth-producer-{steady_worker_id}.log'}"
        )

    prefill_worker_metrics = [
        {
            **json.loads(
                (runtime_root / f"latest-metrics-worker-{worker_id}.json").read_text(
                    encoding="utf-8"
                )
            ),
            "gpu_device": device,
        }
        for worker_id, device in enumerate(prefill_devices)
    ]
    prefill_model_seconds = sum(
        float(metrics["model_seconds"]) for metrics in prefill_worker_metrics
    )
    prefill_generated_images = sum(
        int(metrics["generated_images"]) for metrics in prefill_worker_metrics
    )
    prefill_aggregate = {
        "worker_count": len(prefill_worker_metrics),
        "generated_samples": sum(
            int(metrics["generated_samples"]) for metrics in prefill_worker_metrics
        ),
        "generated_images": prefill_generated_images,
        "model_seconds": prefill_model_seconds,
        "model_images_per_second": prefill_generated_images
        / max(prefill_model_seconds, 1e-9),
    }
    if prefill_aggregate["generated_samples"] != prefill_samples:
        raise RuntimeError(
            "DA3 prefill worker metrics do not account for exactly "
            f"{prefill_samples} samples: {prefill_aggregate}"
        )

    prefill.touch()
    prefill_seconds = time.monotonic() - prefill_started
    prefill_aggregate["wall_seconds"] = prefill_seconds
    prefill_aggregate["wall_images_per_second"] = (
        prefill_generated_images / max(prefill_seconds, 1e-9)
    )
    prefill_event = {
        "event": "da3_prefill_complete",
        "elapsed_seconds": prefill_seconds,
        "aggregate": prefill_aggregate,
        "workers": prefill_worker_metrics,
    }
    encoded_prefill_event = json.dumps(prefill_event, sort_keys=True)
    print(encoded_prefill_event, flush=True)
    with training_log.open("a", encoding="utf-8") as handle:
        handle.write(encoded_prefill_event + "\n")

    training_environment = {
        **base_environment,
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, training_devices)),
    }
    return_code = _run_logged_command(
        [
            sys.executable,
            "-m",
            "mvtracker.cli.train",
            f"+experiment={experiment_name}",
        ],
        cwd=SOURCE_ROOT,
        environment=training_environment,
        log_path=training_log,
        label="recipe-da3-training",
    )
    if return_code != 0:
        raise RuntimeError(
            f"DA3 recipe training exited {return_code}; see {training_log}"
        )

    producer, producer_thread = workers[steady_worker_id]
    producer_return_code = producer.wait(timeout=30 * 60)
    producer_thread.join()
    if producer_return_code != 0:
        raise RuntimeError(
            f"depth producer exited {producer_return_code}; see "
            f"{run_dir / f'depth-producer-{steady_worker_id}.log'}"
        )
    final_producer_metrics = json.loads(
        (runtime_root / f"latest-metrics-worker-{steady_worker_id}.json").read_text(
            encoding="utf-8"
        )
    )
    result = {
        **manifest,
        "prefill_seconds": prefill_seconds,
        "prefill": prefill_aggregate,
        "prefill_workers": prefill_worker_metrics,
        "producer": final_producer_metrics,
        "checkpoint": str(run_dir / "model_final.pth"),
    }
    (run_dir / "integration-summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"DA3 recipe training complete result={result}", flush=True)
    return result


def _run_recipe_da3(
    run_name: str,
    recipe_name: str,
    experiment_name: str,
    expected_steps: int,
    wandb_group: str,
    prefill_samples: int = 64,
    *,
    gpu_label: str,
    training_devices: tuple[int, ...],
    depth_device: int,
    prefill_devices: tuple[int, ...],
    da3_image_capacity: int,
    max_pending_depth_samples: int,
) -> dict:
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
    recipe_summary = json.loads(
        (recipe_path / "summary.json").read_text(encoding="utf-8")
    )
    if int(recipe_manifest["step_count"]) != int(expected_steps):
        raise ValueError(
            f"recipe has {recipe_manifest['step_count']} steps, expected {expected_steps}"
        )
    seed = int(recipe_manifest["seed"])
    runtime_root = Path("/tmp/mvtracker-da3-depth")
    training_log = run_dir / "training.log"
    manifest = {
        "mode": "recipe_da3",
        "run_name": run_name,
        "recipe": recipe_name,
        "source_commit": commit,
        "gpu": gpu_label,
        "training_devices": list(training_devices),
        "depth_device": int(depth_device),
        "prefill_devices": list(prefill_devices),
        "prefill_samples": int(prefill_samples),
        "da3_image_capacity": int(da3_image_capacity),
        "max_pending_depth_samples": int(max_pending_depth_samples),
        "steps": int(expected_steps),
        "depth_counts": recipe_summary["planned_depth_counts"],
        "validation": "smoke20" not in experiment_name,
        "modal_tags": MODAL_TAGS,
    }
    (run_dir / "modal-run-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    run_volume.commit()
    print(
        "DA3 recipe training startup "
        f"commit={commit} run={run_name} recipe={recipe_path} "
        f"depth_counts={recipe_summary['planned_depth_counts']}",
        flush=True,
    )

    base_environment = os.environ.copy()
    base_environment.update(
        {
            "MVTRACKER_DATA_ROOT": DATA_VOLUME_ROOT,
            "MVTRACKER_TRAINING_RECIPE": str(recipe_path),
            "MVTRACKER_TRAINING_CHECKPOINT": str(
                Path(DATA_VOLUME_ROOT) / "checkpoints/mvtracker_200000_june2025.pth"
            ),
            "MVTRACKER_TRAINING_RUN_DIR": str(run_dir),
            "MVTRACKER_TRAINING_SEED": str(seed),
            "MVTRACKER_WANDB_RUN_NAME": run_name,
            "MVTRACKER_WANDB_RUN_ID": wandb_run_id,
            "MVTRACKER_RUNTIME_DEPTH_ROOT": str(runtime_root),
            "MVTRACKER_RUNTIME_DEPTH_METRICS": str(
                runtime_root
                / f"latest-metrics-worker-{prefill_devices.index(depth_device)}.json"
            ),
            "WANDB_ENTITY": WANDB_ENTITY,
            "WANDB_PROJECT": WANDB_PROJECT,
            "WANDB_RUN_GROUP": wandb_group,
            "WANDB_RUN_ID": wandb_run_id,
            "WANDB_RESUME": "allow",
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    cache_environment = {
        **base_environment,
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HOME": "/tmp/huggingface-da3",
        "MVTRACKER_DA3_IMAGE_CAPACITY": str(da3_image_capacity),
    }
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True)
    cache_return_code = _run_logged_command(
        [
            sys.executable,
            "-m",
            "mvtracker.preprocessing.runtime_da3",
            "--download-only",
        ],
        cwd=SOURCE_ROOT,
        environment=cache_environment,
        log_path=run_dir / "depth-model-cache.log",
        label="depth-model-cache",
    )
    if cache_return_code != 0:
        raise RuntimeError(
            "DA3 model cache warmup failed; see "
            f"{run_dir / 'depth-model-cache.log'}"
        )

    producer_command = [
        sys.executable,
        "-m",
        "mvtracker.preprocessing.runtime_da3",
        "--recipe",
        str(recipe_path),
        "--data-root",
        DATA_VOLUME_ROOT,
        "--output-root",
        str(runtime_root),
        "--prefill-samples",
        str(prefill_samples),
        "--max-pending-samples",
        str(max_pending_depth_samples),
    ]
    if depth_device not in prefill_devices:
        raise ValueError("depth_device must be included in prefill_devices")
    if len(set(prefill_devices)) != len(prefill_devices):
        raise ValueError("prefill_devices must be unique")
    workers = []
    try:
        for worker_id, device in enumerate(prefill_devices):
            environment = {
                **base_environment,
                "CUDA_VISIBLE_DEVICES": str(device),
                "HF_HOME": "/tmp/huggingface-da3",
                "MVTRACKER_DA3_IMAGE_CAPACITY": str(da3_image_capacity),
            }
            command = [
                *producer_command,
                "--worker-id",
                str(worker_id),
                "--worker-count",
                str(len(prefill_devices)),
            ]
            if device != depth_device:
                command.append("--prefill-only")
            process, thread = _start_logged_process(
                command,
                cwd=SOURCE_ROOT,
                environment=environment,
                log_path=run_dir / f"depth-producer-{worker_id}.log",
                label=f"depth-producer-{worker_id}",
            )
            workers.append((process, thread))

        return _coordinate_recipe_da3(
            workers=workers,
            prefill_devices=prefill_devices,
            depth_device=depth_device,
            runtime_root=runtime_root,
            recipe_path=recipe_path,
            prefill_samples=prefill_samples,
            run_dir=run_dir,
            base_environment=base_environment,
            training_devices=training_devices,
            experiment_name=experiment_name,
            training_log=training_log,
            manifest=manifest,
        )
    finally:
        _terminate_logged_processes(workers)
        run_volume.commit()


@app.function(
    image=da3_training_image,
    secrets=[hf_secret, wandb_secret],
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
        str(RUN_ROOT): run_volume,
    },
    gpu="H200:3",
    cpu=32,
    memory=(TRAIN_MEMORY_REQUEST_MIB, TRAIN_MEMORY_LIMIT_MIB),
    timeout=12 * 60 * 60,
    max_containers=1,
    retries=0,
    include_source=False,
)
def recipe_da3_remote(
    run_name: str,
    recipe_name: str,
    experiment_name: str,
    expected_steps: int,
    wandb_group: str,
    prefill_samples: int = 64,
) -> dict:
    return _run_recipe_da3(
        run_name,
        recipe_name,
        experiment_name,
        expected_steps,
        wandb_group,
        prefill_samples,
        gpu_label="H200:3",
        training_devices=(0, 1),
        depth_device=2,
        prefill_devices=(0, 1, 2),
        da3_image_capacity=80,
        max_pending_depth_samples=32,
    )


@app.function(
    image=da3_training_image,
    secrets=[hf_secret, wandb_secret],
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
        str(RUN_ROOT): run_volume,
    },
    gpu="H100!:5",
    cpu=32,
    memory=(TRAIN_MEMORY_REQUEST_MIB, TRAIN_MEMORY_LIMIT_MIB),
    timeout=24 * 60 * 60,
    max_containers=1,
    retries=0,
    include_source=False,
)
def recipe_da3_h100x5_remote(
    run_name: str,
    recipe_name: str,
    prefill_samples: int = 64,
) -> dict:
    return _run_recipe_da3(
        run_name,
        recipe_name,
        "diegesis_syn4d_mvkubric_recipe_da3_ddp_5000",
        5000,
        "diegesis351-syn4d-mvkubric-da3-5000",
        prefill_samples,
        gpu_label="H100!:5",
        training_devices=(0, 1, 2, 3),
        depth_device=4,
        prefill_devices=(0, 1, 2, 3, 4),
        da3_image_capacity=64,
        max_pending_depth_samples=64,
    )


@app.function(
    image=da3_training_image,
    secrets=[hf_secret, wandb_secret],
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
        str(RUN_ROOT): run_volume,
    },
    gpu="H200:5",
    cpu=32,
    memory=(TRAIN_MEMORY_REQUEST_MIB, TRAIN_MEMORY_LIMIT_MIB),
    timeout=24 * 60 * 60,
    max_containers=1,
    retries=0,
    include_source=False,
)
def recipe_da3_h200x5_remote(
    run_name: str,
    recipe_name: str,
    prefill_samples: int = 64,
) -> dict:
    return _run_recipe_da3(
        run_name,
        recipe_name,
        "diegesis_syn4d_mvkubric_recipe_da3_h200_ddp_5000",
        5000,
        "diegesis351-syn4d-mvkubric-da3-h200-5000",
        prefill_samples,
        gpu_label="H200:5",
        training_devices=(0, 1, 2, 3),
        depth_device=4,
        prefill_devices=(0, 1, 2, 3, 4),
        da3_image_capacity=64,
        max_pending_depth_samples=64,
    )


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


@app.local_entrypoint(name="audit-recipe-gradients")
def audit_recipe_gradients(
    audit_name: str,
    source_run_name: str,
    recipe_name: str,
    optimizer_steps: str = "625,688",
) -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=1)
    validate_run_name(audit_name)
    validate_run_name(source_run_name)
    validate_run_name(recipe_name)
    app.set_tags(
        {
            "owner": "jeet",
            "project": "mvtracker",
            "purpose": "evaluation",
            "experiment": audit_name,
            "gpu": "h200x1",
        }
    )
    deployed = modal.Function.from_name(APP_NAME, "audit_recipe_gradients_remote")
    call = deployed.spawn(
        audit_name,
        source_run_name,
        recipe_name,
        optimizer_steps,
    )
    print(
        json.dumps(
            {"audit_name": audit_name, "function_call_id": call.object_id},
            indent=2,
        )
    )


@app.local_entrypoint(name="visualize-recipe-augmentations")
def visualize_recipe_augmentations(
    audit_name: str,
    recipe_name: str,
) -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=1)
    validate_run_name(audit_name)
    validate_run_name(recipe_name)
    app.set_tags(
        {
            "owner": "jeet",
            "project": "mvtracker",
            "purpose": "evaluation",
            "experiment": audit_name,
            "gpu": "t4x1",
        }
    )
    deployed = modal.Function.from_name(
        APP_NAME, "visualize_recipe_augmentations_remote"
    )
    result = deployed.remote(audit_name, recipe_name)
    print(json.dumps(result, indent=2))


@app.local_entrypoint(name="plan-recipe")
def plan_recipe(
    recipe_name: str,
    step_count: int = 2000,
    world_size: int = 2,
    global_batch_size: int = 8,
) -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=1)
    validate_run_name(recipe_name)
    app.set_tags(
        {**MODAL_TAGS, "experiment": recipe_name, "gpu": "cpu", "cpu": "32"}
    )
    deployed = modal.Function.from_name(APP_NAME, "plan_recipe_remote")
    call = deployed.spawn(recipe_name, step_count, world_size, global_batch_size)
    print(json.dumps({"recipe_name": recipe_name, "function_call_id": call.object_id}, indent=2))


@app.local_entrypoint(name="build-mvkubric-recipe-metadata")
def build_mvkubric_recipe_metadata() -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=1)
    app.set_tags(
        {
            **PROFILE_TAGS,
            "experiment": "mvkubric-recipe-metadata",
            "gpu": "cpu",
            "cpu": "16",
        }
    )
    deployed = modal.Function.from_name(
        APP_NAME, "build_mvkubric_recipe_metadata_remote"
    )
    call = deployed.spawn()
    print(json.dumps({"function_call_id": call.object_id}, indent=2))


@app.local_entrypoint(name="replan-syn4d-recipe")
def replan_syn4d_recipe(source_name: str, recipe_name: str) -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=1)
    validate_run_name(source_name)
    validate_run_name(recipe_name)
    app.set_tags(
        {**MODAL_TAGS, "experiment": recipe_name, "gpu": "cpu", "cpu": "16"}
    )
    deployed = modal.Function.from_name(APP_NAME, "replan_syn4d_recipe_remote")
    call = deployed.spawn(source_name, recipe_name)
    print(
        json.dumps(
            {"recipe_name": recipe_name, "function_call_id": call.object_id},
            indent=2,
        )
    )


@app.local_entrypoint(name="recipe-smoke20")
def recipe_smoke20(run_name: str, recipe_name: str) -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers()
    validate_run_name(run_name)
    validate_run_name(recipe_name)
    app.set_tags(
        {**MODAL_TAGS, "experiment": run_name, "gpu": "h200x2"}
    )
    deployed = modal.Function.from_name(APP_NAME, "recipe_smoke20_remote")
    call = deployed.spawn(run_name, recipe_name)
    print(json.dumps({"run_name": run_name, "function_call_id": call.object_id}, indent=2))


@app.local_entrypoint(name="fabricate-depth-recipe")
def fabricate_depth_recipe(source_name: str, recipe_name: str) -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=1)
    validate_run_name(source_name)
    validate_run_name(recipe_name)
    app.set_tags(
        {**MODAL_TAGS, "experiment": recipe_name, "gpu": "cpu"}
    )
    deployed = modal.Function.from_name(APP_NAME, "fabricate_depth_recipe_remote")
    call = deployed.spawn(source_name, recipe_name)
    print(
        json.dumps(
            {"recipe_name": recipe_name, "function_call_id": call.object_id},
            indent=2,
        )
    )


@app.local_entrypoint(name="fabricate-singleton-recipe")
def fabricate_singleton_recipe(source_name: str, recipe_name: str) -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=1)
    validate_run_name(source_name)
    validate_run_name(recipe_name)
    app.set_tags(
        {**MODAL_TAGS, "experiment": recipe_name, "gpu": "cpu", "cpu": "2"}
    )
    deployed = modal.Function.from_name(APP_NAME, "fabricate_singleton_recipe_remote")
    call = deployed.spawn(source_name, recipe_name)
    print(
        json.dumps(
            {"recipe_name": recipe_name, "function_call_id": call.object_id},
            indent=2,
        )
    )


@app.local_entrypoint(name="recipe-da3-smoke20")
def recipe_da3_smoke20(run_name: str, recipe_name: str) -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=3)
    validate_run_name(run_name)
    validate_run_name(recipe_name)
    app.set_tags(
        {**MODAL_TAGS, "experiment": run_name, "gpu": "h200x3"}
    )
    deployed = modal.Function.from_name(APP_NAME, "recipe_da3_remote")
    call = deployed.spawn(
        run_name,
        recipe_name,
        "diegesis_syn4d_mvkubric_recipe_da3_ddp_smoke20",
        20,
        "da3-runtime-depth-smoke20",
        64,
    )
    print(json.dumps({"run_name": run_name, "function_call_id": call.object_id}, indent=2))


@app.local_entrypoint(name="recipe-da3-train1000")
def recipe_da3_train1000(run_name: str, recipe_name: str) -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=3)
    validate_run_name(run_name)
    validate_run_name(recipe_name)
    app.set_tags(
        {**MODAL_TAGS, "experiment": run_name, "gpu": "h200x3"}
    )
    deployed = modal.Function.from_name(APP_NAME, "recipe_da3_remote")
    call = deployed.spawn(
        run_name,
        recipe_name,
        "diegesis_syn4d_mvkubric_recipe_da3_ddp_1000",
        1000,
        "expanded-syn4d-da3-1000",
        64,
    )
    print(json.dumps({"run_name": run_name, "function_call_id": call.object_id}, indent=2))


@app.local_entrypoint(name="recipe-da3-h100x5-train5000")
def recipe_da3_h100x5_train5000(run_name: str, recipe_name: str) -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=5)
    validate_run_name(run_name)
    validate_run_name(recipe_name)
    app.set_tags(
        {**MODAL_TAGS, "experiment": run_name, "gpu": "h100x5"}
    )
    deployed = modal.Function.from_name(APP_NAME, "recipe_da3_h100x5_remote")
    call = deployed.spawn(run_name, recipe_name, 64)
    print(
        json.dumps(
            {"run_name": run_name, "function_call_id": call.object_id},
            indent=2,
        )
    )


@app.local_entrypoint(name="recipe-da3-h200x5-train5000")
def recipe_da3_h200x5_train5000(run_name: str, recipe_name: str) -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=5)
    validate_run_name(run_name)
    validate_run_name(recipe_name)
    app.set_tags(
        {**MODAL_TAGS, "experiment": run_name, "gpu": "h200x5"}
    )
    deployed = modal.Function.from_name(APP_NAME, "recipe_da3_h200x5_remote")
    call = deployed.spawn(run_name, recipe_name, 64)
    print(
        json.dumps(
            {"run_name": run_name, "function_call_id": call.object_id},
            indent=2,
        )
    )


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
