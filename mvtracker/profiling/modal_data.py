"""Materialize the immutable Modal dataset used by MVTracker profiles."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

from huggingface_hub import hf_hub_download, snapshot_download


DIEGESIS_REPO = "j99999/diegesis"
DIEGESIS_REVISION = "81389015a6d713a848a120e34850f360621bcdce"
MVKUBRIC_REPO = "ethz-vlg/mv3dpt-datasets"
MVKUBRIC_REVISION = "cccb9128fb95d302c662151e65a09377175c2a3a"
MVKUBRIC_ARCHIVE = "kubric-multiview--train.micro.0900-0999.tar.gz"
CHECKPOINT_REPO = "ethz-vlg/mvtracker"
CHECKPOINT_REVISION = "010d5d114e860aae6b2568104927b636cdca01bc"
CHECKPOINT_FILE = "mvtracker_200000_june2025_cleandepth.pth"
MANIFEST_VERSION = 1


def _tree_stats(root: Path) -> dict[str, int]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "file_count": len(files),
        "size_bytes": sum(path.stat().st_size for path in files),
    }


def _validate_materialized_data(data_root: Path, manifest: dict) -> None:
    split_counts = manifest["diegesis"]["splits"]
    raw_root = data_root / "datasets/diegesis-mvtracker/TAPVid3D_raw"
    cache_root = data_root / "datasets/diegesis-mvtracker/TAPVid3D_MVTracker_cache"
    for split, expected_count in split_counts.items():
        raw_scenes = [path for path in (raw_root / split).iterdir() if path.is_dir()]
        cached_scenes = [path for path in (cache_root / split).iterdir() if path.is_dir()]
        if len(raw_scenes) != expected_count or len(cached_scenes) != expected_count:
            raise RuntimeError(f"DIEGESIS {split} split is incomplete")
    kubric_train = data_root / "datasets/kubric-multiview/train"
    if len([path for path in kubric_train.iterdir() if path.is_dir()]) != 100:
        raise RuntimeError("MV-Kubric micro is incomplete")
    checkpoint = data_root / "checkpoints" / CHECKPOINT_FILE
    if checkpoint.stat().st_size != manifest["checkpoint"]["size_bytes"]:
        raise RuntimeError("clean-depth checkpoint size differs from its manifest")


def _materialize_diegesis(data_root: Path, source_root: Path, token: str) -> dict:
    snapshot_download(
        repo_id=DIEGESIS_REPO,
        repo_type="dataset",
        revision=DIEGESIS_REVISION,
        token=token,
        local_dir=source_root,
    )
    split_document = json.loads(
        (Path(__file__).resolve().parents[2] / "configs/diegesis_split_v1.json").read_text(
            encoding="utf-8"
        )
    )
    raw_root = data_root / "datasets/diegesis-mvtracker/TAPVid3D_raw"
    if raw_root.exists():
        shutil.rmtree(raw_root)
    for split, scenes in split_document["splits"].items():
        split_root = raw_root / split
        split_root.mkdir(parents=True)
        for scene in scenes:
            source = source_root / "scenes" / scene / "tracking/sequence"
            if not source.is_dir():
                raise FileNotFoundError(source)
            relative = os.path.relpath(source, split_root)
            (split_root / scene).symlink_to(relative, target_is_directory=True)

    cache_root = data_root / "datasets/diegesis-mvtracker/TAPVid3D_MVTracker_cache"
    if cache_root.exists():
        shutil.rmtree(cache_root)
    subprocess.run(
        [
            "python",
            str(Path(__file__).resolve().parents[2] / "scripts/prepare_tapvid3d_mvtracker.py"),
            "--raw-root",
            str(raw_root),
            "--cache-root",
            str(cache_root),
            "--workers",
            "8",
        ],
        check=True,
    )
    return {
        "repo_id": DIEGESIS_REPO,
        "revision": DIEGESIS_REVISION,
        "splits": {
            split: len(scenes) for split, scenes in split_document["splits"].items()
        },
        "source": _tree_stats(source_root),
        "cache": _tree_stats(cache_root),
    }


def _materialize_mvkubric(data_root: Path, token: str) -> dict:
    archive = Path(
        hf_hub_download(
            repo_id=MVKUBRIC_REPO,
            repo_type="dataset",
            revision=MVKUBRIC_REVISION,
            filename=MVKUBRIC_ARCHIVE,
            token=token,
            cache_dir="/tmp/mvtracker-hf",
        )
    )
    staging = data_root / ".staging/mvkubric"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    subprocess.run(
        [
            "tar",
            "--extract",
            "--gzip",
            "--file",
            str(archive),
            "--directory",
            str(staging),
            "--no-same-owner",
        ],
        check=True,
    )
    extracted = staging / "kubric-multiview"
    train = extracted / "train"
    if not train.is_dir():
        raise FileNotFoundError(
            f"{MVKUBRIC_ARCHIVE} did not contain kubric-multiview/train"
        )
    sequences = sorted(path.name for path in train.iterdir() if path.is_dir())
    if len(sequences) != 100:
        raise RuntimeError(f"MV-Kubric micro contained {len(sequences)} scenes, expected 100")
    destination = data_root / "datasets/kubric-multiview"
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    extracted.rename(destination)
    shutil.rmtree(staging)
    return {
        "repo_id": MVKUBRIC_REPO,
        "revision": MVKUBRIC_REVISION,
        "archive": MVKUBRIC_ARCHIVE,
        "archive_size_bytes": archive.stat().st_size,
        "scene_count": len(sequences),
        "extracted": _tree_stats(destination),
    }


def _materialize_checkpoint(data_root: Path, token: str) -> dict:
    checkpoint_root = data_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=CHECKPOINT_REPO,
            revision=CHECKPOINT_REVISION,
            filename=CHECKPOINT_FILE,
            token=token,
            local_dir=checkpoint_root,
        )
    )
    return {
        "repo_id": CHECKPOINT_REPO,
        "revision": CHECKPOINT_REVISION,
        "filename": CHECKPOINT_FILE,
        "size_bytes": downloaded.stat().st_size,
    }


def materialize_profile_data(data_root: Path) -> dict:
    token = os.environ["HF_TOKEN"]
    manifest_path = data_root / "profile-data-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": MANIFEST_VERSION,
            "diegesis_revision": DIEGESIS_REVISION,
            "mvkubric_revision": MVKUBRIC_REVISION,
            "checkpoint_revision": CHECKPOINT_REVISION,
        }
        observed = {name: manifest.get(name) for name in expected}
        if observed != expected:
            raise RuntimeError("existing Modal profile data has a different manifest")
        _validate_materialized_data(data_root, manifest)
        return manifest

    data_root.mkdir(parents=True, exist_ok=True)
    source_root = data_root / "source/diegesis"
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "diegesis_revision": DIEGESIS_REVISION,
        "mvkubric_revision": MVKUBRIC_REVISION,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "diegesis": _materialize_diegesis(data_root, source_root, token),
        "mvkubric": _materialize_mvkubric(data_root, token),
        "checkpoint": _materialize_checkpoint(data_root, token),
    }
    _validate_materialized_data(data_root, manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
