"""Immutable Modal data materialization for continual training."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from huggingface_hub import hf_hub_download

CHECKPOINT_REPO = "ethz-vlg/mvtracker"
CHECKPOINT_REVISION = "010d5d114e860aae6b2568104927b636cdca01bc"
CHECKPOINT_FILE = "mvtracker_200000_june2025.pth"
CHECKPOINT_SHA256 = "a7fa86f2a7223e3e0aa4c1d3eff0dec5fe8a9227a48572ce943b8e49d8a4f8e6"
MANIFEST_VERSION = 1
EXPECTED_DIEGESIS_SPLITS = {"train": 17, "validation": 2, "test": 2}
EXPECTED_MVKUBRIC_SCENES = {str(scene) for scene in range(900, 1000)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _materialize_checkpoint(data_root: Path, token: str) -> dict:
    checkpoint_root = data_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    path = Path(
        hf_hub_download(
            repo_id=CHECKPOINT_REPO,
            revision=CHECKPOINT_REVISION,
            filename=CHECKPOINT_FILE,
            token=token,
            local_dir=checkpoint_root,
        )
    )
    observed = sha256(path)
    if observed != CHECKPOINT_SHA256:
        raise RuntimeError(f"mixed-depth checkpoint checksum mismatch: {observed}")
    return {
        "repo_id": CHECKPOINT_REPO,
        "revision": CHECKPOINT_REVISION,
        "filename": CHECKPOINT_FILE,
        "sha256": observed,
        "size_bytes": path.stat().st_size,
    }


def _require_existing_profile_data(data_root: Path) -> dict:
    manifest_path = data_root / "profile-data-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            "profile-data-manifest.json is required; continual setup does not "
            "download or rebuild dataset inputs"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("diegesis", {}).get("splits") != EXPECTED_DIEGESIS_SPLITS:
        raise RuntimeError("existing DIEGESIS split manifest is incompatible")

    raw_root = data_root / "datasets/diegesis-mvtracker/TAPVid3D_raw"
    cache_root = data_root / "datasets/diegesis-mvtracker/TAPVid3D_MVTracker_cache"
    for split, expected_count in EXPECTED_DIEGESIS_SPLITS.items():
        raw = {path.name for path in (raw_root / split).iterdir() if path.is_dir()}
        cached = {path.name for path in (cache_root / split).iterdir() if path.is_dir()}
        if len(raw) != expected_count or raw != cached:
            raise RuntimeError(f"existing DIEGESIS {split} data is incomplete")

    mvkubric_root = data_root / "datasets/kubric-multiview/train"
    mvkubric = {path.name for path in mvkubric_root.iterdir() if path.is_dir()}
    if mvkubric != EXPECTED_MVKUBRIC_SCENES:
        raise RuntimeError("existing MV-Kubric micro pool must be exactly scenes 900..999")
    return manifest


def materialize_continual_training_data(data_root: Path) -> dict:
    token = os.environ["HF_TOKEN"]
    profile_manifest = _require_existing_profile_data(data_root)
    manifest_path = data_root / "continual-training-data-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != MANIFEST_VERSION
            or manifest.get("mvkubric_revision") != profile_manifest["mvkubric_revision"]
            or manifest.get("checkpoint", {}).get("sha256") != CHECKPOINT_SHA256
        ):
            raise RuntimeError("existing continual-training data manifest is incompatible")
        _materialize_checkpoint(data_root, token)
        return manifest
    manifest = {
        "schema_version": MANIFEST_VERSION,
        "mvkubric_revision": profile_manifest["mvkubric_revision"],
        "diegesis": profile_manifest["diegesis"],
        "mvkubric": profile_manifest["mvkubric"],
        "checkpoint": _materialize_checkpoint(data_root, token),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
