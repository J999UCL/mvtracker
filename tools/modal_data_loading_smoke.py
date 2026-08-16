"""Measure archive-to-local-SSD staging and one CPU sample per dataset."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import modal

from modal_training_profile import (
    DATA_ROOT,
    _runtime_image,
    _source_commit,
    data_volume,
    wandb_secret,
)
from mvtracker.profiling.modal_continual_training import (
    EPHEMERAL_DISK_MIB,
    PROFILE_TAGS,
    WANDB_ENTITY,
    WANDB_GROUP,
    WANDB_PROJECT,
    preflight_active_containers,
    require_pushed_main_commit,
)


APP_NAME = "jeet-mvtracker-data-loading-smoke"
SOURCE_ROOT = Path("/opt/mvtracker")
LOCAL_ROOT = Path("/tmp/mvtracker-data")
DIEGESIS_ARCHIVE = DATA_ROOT / (
    "archives/diegesis/"
    "diegesis-81389015a6d713a848a120e34850f360621bcdce.tar.zst"
)
MVKUBRIC_ARCHIVE = (
    DATA_ROOT
    / "archives/mvkubric/kubric-multiview--train.micro.0900-0999.tar.gz"
)
DIEGESIS_CACHE = (
    DATA_ROOT
    / "datasets/diegesis-mvtracker/TAPVid3D_MVTracker_cache"
)
MVKUBRIC_INDEX = DATA_ROOT / "datasets/kubric-multiview/train/MVTracker_index"

app = modal.App(
    APP_NAME,
    tags={**PROFILE_TAGS, "experiment": "archive-staging-cpu", "gpu": "cpu"},
)
image = _runtime_image().apt_install("zstd")


def _copy_file(source: Path, destination: Path) -> dict[str, float | int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    shutil.copyfile(source, destination)
    elapsed = time.perf_counter() - started
    size = source.stat().st_size
    return {
        "seconds": elapsed,
        "bytes": size,
        "gb_per_second": size / 1_000_000_000 / elapsed,
    }


def _copy_tree(source: Path, destination: Path) -> dict[str, float | int]:
    started = time.perf_counter()
    shutil.copytree(source, destination)
    elapsed = time.perf_counter() - started
    size = sum(path.stat().st_size for path in destination.rglob("*") if path.is_file())
    return {
        "seconds": elapsed,
        "bytes": size,
        "gb_per_second": size / 1_000_000_000 / elapsed,
    }


def _extract(archive: Path, destination: Path, *, zstd: bool) -> float:
    destination.mkdir(parents=True, exist_ok=True)
    command = ["tar", "--extract", "--file", str(archive), "--directory", str(destination)]
    command.insert(2, "--zstd" if zstd else "--gzip")
    started = time.perf_counter()
    subprocess.run(command, check=True)
    return time.perf_counter() - started


def _create_diegesis_splits() -> None:
    split_document = json.loads(
        (SOURCE_ROOT / "configs/diegesis_split_v1.json").read_text(encoding="utf-8")
    )
    raw_root = LOCAL_ROOT / "datasets/diegesis-mvtracker/TAPVid3D_raw"
    for split, scenes in split_document["splits"].items():
        split_root = raw_root / split
        split_root.mkdir(parents=True)
        for scene in scenes:
            source = LOCAL_ROOT / "source/diegesis/scenes" / scene / "tracking/sequence"
            if not source.is_dir():
                raise FileNotFoundError(source)
            (split_root / scene).symlink_to(
                os.path.relpath(source, split_root), target_is_directory=True
            )


def _stage() -> dict:
    required = (DIEGESIS_ARCHIVE, MVKUBRIC_ARCHIVE, DIEGESIS_CACHE, MVKUBRIC_INDEX)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing staging inputs: {', '.join(missing)}")
    if LOCAL_ROOT.exists():
        shutil.rmtree(LOCAL_ROOT)
    LOCAL_ROOT.mkdir()

    total_started = time.perf_counter()
    local_diegesis_archive = LOCAL_ROOT / "archives" / DIEGESIS_ARCHIVE.name
    local_mvkubric_archive = LOCAL_ROOT / "archives" / MVKUBRIC_ARCHIVE.name
    metrics = {
        "diegesis_archive_copy": _copy_file(DIEGESIS_ARCHIVE, local_diegesis_archive),
        "mvkubric_archive_copy": _copy_file(MVKUBRIC_ARCHIVE, local_mvkubric_archive),
    }
    metrics["diegesis_extract_seconds"] = _extract(
        local_diegesis_archive,
        LOCAL_ROOT / "source/diegesis",
        zstd=True,
    )
    metrics["mvkubric_extract_seconds"] = _extract(
        local_mvkubric_archive,
        LOCAL_ROOT / "datasets",
        zstd=False,
    )
    cache_destination = (
        LOCAL_ROOT / "datasets/diegesis-mvtracker/TAPVid3D_MVTracker_cache"
    )
    cache_destination.parent.mkdir(parents=True)
    metrics["diegesis_cache_copy"] = _copy_tree(DIEGESIS_CACHE, cache_destination)
    metrics["mvkubric_index_copy"] = _copy_tree(
        MVKUBRIC_INDEX,
        LOCAL_ROOT / "datasets/kubric-multiview/train/MVTracker_index",
    )
    _create_diegesis_splits()
    metrics["total_staging_seconds"] = time.perf_counter() - total_started
    metrics["diegesis_scene_count"] = len(
        list((LOCAL_ROOT / "source/diegesis/scenes").iterdir())
    )
    metrics["mvkubric_scene_count"] = len(
        [
            path
            for path in (LOCAL_ROOT / "datasets/kubric-multiview/train").iterdir()
            if path.is_dir() and path.name.isdigit()
        ]
    )
    if metrics["diegesis_scene_count"] != 21 or metrics["mvkubric_scene_count"] != 100:
        raise RuntimeError("staged dataset scene counts are incomplete")
    return metrics


def _load_one_sample(source: str) -> dict:
    from mvtracker.profiling.modal_continual_data import profile_encoded_loader

    started = time.perf_counter()
    result = profile_encoded_loader(
        LOCAL_ROOT / "datasets",
        source=source,
        warmup=0,
        measured=1,
        workers=0,
        use_cuda=False,
    )
    total = time.perf_counter() - started
    return {
        "total_seconds": total,
        "first_sample_seconds": result["elapsed_seconds"],
        "initialization_seconds": total - result["elapsed_seconds"],
        "encoded_frames": result["encoded_frames"],
    }


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume.with_mount_options(read_only=True)},
    cpu=16,
    memory=32768,
    ephemeral_disk=EPHEMERAL_DISK_MIB,
    timeout=4 * 60 * 60,
    max_containers=1,
    include_source=False,
)
def run_smoke() -> dict:
    import wandb

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group=WANDB_GROUP,
        job_type="data-loading-smoke",
        tags=["modal", "cpu", "archive-staging"],
        config={"source_commit": _source_commit(), **PROFILE_TAGS},
    )
    result = {
        "staging": _stage(),
        "diegesis_loader": _load_one_sample("diegesis"),
        "mvkubric_loader": _load_one_sample("mvkubric"),
    }
    run.summary.update(result)
    run.finish()
    return result


@app.local_entrypoint()
def main() -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    preflight_active_containers(required_free_slots=1)
    print(json.dumps(run_smoke.remote(), indent=2))
