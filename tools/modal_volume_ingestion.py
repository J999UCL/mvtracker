"""Materialize continual-training datasets directly into Modal Volume v2."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import shutil
import subprocess
import time


DATA_ROOT = Path("/mnt/mvtracker-data")
ARCHIVE_ROOT = DATA_ROOT / "archives/mvkubric/2000-scenes-v1"
TRAIN_ROOT = DATA_ROOT / "datasets/kubric-multiview/train"
VALIDATION_ROOT = DATA_ROOT / "datasets/kubric-multiview/2000-scenes-v1/validation"
DIEGESIS_ARCHIVE = DATA_ROOT / (
    "archives/diegesis/"
    "diegesis-81389015a6d713a848a120e34850f360621bcdce.tar.zst"
)
DIEGESIS_SOURCE_ROOT = DATA_ROOT / "source/diegesis"
DIEGESIS_DATASET_ROOT = DATA_ROOT / "datasets/diegesis-mvtracker"
TRAIN_ARCHIVES = (
    ("kubric-multiview--train.full.1001-2000.tar.gz", 1001, 2000),
    ("kubric-multiview--train.full.2001-3000.tar.gz", 2001, 3000),
)
VALIDATION_SCENES = tuple(str(scene) for scene in range(101, 128))
MANIFEST_PATH = DATA_ROOT / "direct-volume-data-manifest.json"


def _extract_mvkubric_archive(archive_name: str, destination: Path) -> float:
    source = ARCHIVE_ROOT / archive_name
    local_archive = Path("/tmp") / archive_name
    started = time.perf_counter()
    shutil.copyfile(source, local_archive)
    decompressor = subprocess.Popen(
        ["rapidgzip", "-d", "-c", "-P", "16", str(local_archive)],
        stdout=subprocess.PIPE,
    )
    assert decompressor.stdout is not None
    extraction = subprocess.run(
        [
            "tar",
            "--extract",
            "--strip-components=2",
            "--file",
            "-",
            "--directory",
            str(destination),
        ],
        stdin=decompressor.stdout,
        check=False,
    )
    decompressor.stdout.close()
    decompression_returncode = decompressor.wait()
    local_archive.unlink()
    if decompression_returncode:
        raise subprocess.CalledProcessError(decompression_returncode, decompressor.args)
    if extraction.returncode:
        raise subprocess.CalledProcessError(extraction.returncode, extraction.args)
    return time.perf_counter() - started


def _materialize_diegesis() -> None:
    marker = DATA_ROOT / "direct-volume-diegesis.json"
    if marker.is_file():
        print("INGEST phase=diegesis event=skip-complete", flush=True)
        return
    print("INGEST phase=diegesis event=extract-start", flush=True)
    staging = DATA_ROOT / "source/.diegesis-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    subprocess.run(
        [
            "tar",
            "--extract",
            "--zstd",
            "--file",
            str(DIEGESIS_ARCHIVE),
            "--directory",
            str(staging),
        ],
        check=True,
    )
    if DIEGESIS_SOURCE_ROOT.exists():
        shutil.rmtree(DIEGESIS_SOURCE_ROOT)
    staging.rename(DIEGESIS_SOURCE_ROOT)

    splits = {
        "train": (
            "bathroom01", "bathroom02", "bathroom03", "bedroom02", "bedroom03",
            "bedroom04", "diningroom01", "diningroom03", "diningroom04", "kitchen01",
            "kitchen02", "kitchen03", "kitchen04", "livingroom01", "livingroom03",
            "livingroom04", "livingroom05",
        ),
        "validation": ("bedroom01", "diningroom02"),
        "test": ("bathroom04", "livingroom02"),
    }
    raw_root = DIEGESIS_DATASET_ROOT / "TAPVid3D_raw"
    if raw_root.exists():
        shutil.rmtree(raw_root)
    for split, scenes in splits.items():
        split_root = raw_root / split
        split_root.mkdir(parents=True)
        for scene in scenes:
            sequence = DIEGESIS_SOURCE_ROOT / "scenes" / scene / "tracking/sequence"
            split_root.joinpath(scene).symlink_to(
                os.path.relpath(sequence, split_root), target_is_directory=True
            )
    marker.write_text('{"status":"complete"}\n')
    print("INGEST phase=diegesis event=complete", flush=True)


def materialize_direct_volume_data(commit) -> dict:
    """Expand both datasets once and publish the observed inventory."""

    from mvtracker.datasets.kubric_metadata_index import build_kubric_metadata_index

    if MANIFEST_PATH.is_file():
        return json.loads(MANIFEST_PATH.read_text())

    _materialize_diegesis()
    commit()

    staging = DATA_ROOT / "datasets/kubric-multiview/.train-staging-v2"
    staging.mkdir(parents=True, exist_ok=True)
    archive_seconds = {}
    for archive_name, scene_start, scene_end in TRAIN_ARCHIVES:
        marker = staging / f".complete-{scene_start}-{scene_end}"
        if marker.is_file():
            archive_seconds[archive_name] = 0.0
            print(f"INGEST archive={archive_name} event=skip-complete", flush=True)
            continue
        for path in staging.iterdir():
            if path.is_dir() and path.name.isdigit() and scene_start <= int(path.name) <= scene_end:
                shutil.rmtree(path)
        print(f"INGEST archive={archive_name} event=copy-and-extract-start", flush=True)
        archive_seconds[archive_name] = _extract_mvkubric_archive(
            archive_name, staging
        )
        marker.write_text("complete\n")
        print(
            f"INGEST archive={archive_name} event=complete "
            f"seconds={archive_seconds[archive_name]:.1f}",
            flush=True,
        )
        commit()

    print(
        f"INGEST phase=validation-copy event=start scenes={len(VALIDATION_SCENES)}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {
            executor.submit(
                shutil.copytree,
                VALIDATION_ROOT / scene_id,
                staging / scene_id,
                dirs_exist_ok=True,
            ): scene_id
            for scene_id in VALIDATION_SCENES
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            future.result()
            print(
                f"INGEST phase=validation-copy event=progress "
                f"completed={completed}/{len(VALIDATION_SCENES)}",
                flush=True,
            )

    index_root = staging / "MVTracker_index"
    print("INGEST phase=index event=start", flush=True)
    build_kubric_metadata_index(staging, index_root=index_root, overwrite=True)
    observed = sorted(
        (
            path.name
            for path in staging.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        key=int,
    )
    train_scenes = [scene for scene in observed if scene not in VALIDATION_SCENES]

    if TRAIN_ROOT.exists():
        print("INGEST phase=publish event=remove-old-train", flush=True)
        shutil.rmtree(TRAIN_ROOT)
    print("INGEST phase=publish event=rename-staging", flush=True)
    staging.rename(TRAIN_ROOT)
    manifest = {
        "schema_version": 1,
        "layout": "direct-volume-v2",
        "train_scene_ids": train_scenes,
        "validation_scene_ids": list(VALIDATION_SCENES),
        "train_scene_count": len(train_scenes),
        "validation_scene_count": len(VALIDATION_SCENES),
        "archive_seconds": archive_seconds,
        "index": "datasets/kubric-multiview/train/MVTracker_index",
    }
    temporary = MANIFEST_PATH.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(MANIFEST_PATH)
    commit()
    print(
        f"INGEST phase=publish event=complete train_scenes={len(train_scenes)} "
        f"validation_scenes={len(VALIDATION_SCENES)}",
        flush=True,
    )
    return manifest
