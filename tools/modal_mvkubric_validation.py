"""Materialize the two MV-Kubric validation scenes into the shared Modal volume."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

import modal


DATA_ROOT = Path("/mnt/mvtracker-data")
MVKUBRIC_REPO = "ethz-vlg/mv3dpt-datasets"
MVKUBRIC_REVISION = "cccb9128fb95d302c662151e65a09377175c2a3a"
SOURCE_ARCHIVE = "kubric-multiview--train.full.0031-1000.tar.gz"
VALIDATION_SCENES = ("101", "102")
TRAIN_ROOT = DATA_ROOT / "datasets/kubric-multiview/train"
INDEX_ROOT = TRAIN_ROOT / "MVTracker_index"

app = modal.App(
    "jeet-mvkubric-validation",
    tags={"owner": "jeet", "project": "mvtracker", "purpose": "profiling"},
)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub", "numpy", "torch", "wandb"
)
volume = modal.Volume.from_name("jeet-mvtracker-data-v2", version=2)
hf_secret = modal.Secret.from_name(
    "jeet-mvtracker-huggingface", required_keys=["HF_TOKEN"]
)
wandb_secret = modal.Secret.from_name(
    "jeet-mvtracker-wandb", required_keys=["WANDB_API_KEY"]
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scene_inventory(root: Path) -> list[str]:
    return sorted(
        (path.name for path in root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=int,
    )


def _validate_scene(root: Path, scene_id: str) -> dict[str, object]:
    scene = root / scene_id
    if not scene.is_dir():
        raise FileNotFoundError(scene)
    if not (scene / "tracks_3d.npz").is_file() or not (scene / "scene.json").is_file():
        raise RuntimeError(f"scene {scene_id} is missing core metadata")
    views = sorted(path.parent.name for path in scene.glob("view_*/metadata.json"))
    if views != ["view_0", "view_1", "view_2", "view_3"]:
        raise RuntimeError(f"scene {scene_id} has unexpected views: {views}")
    for view in views:
        view_root = scene / view
        if not list(view_root.glob("rgba_*.png")) or not list(view_root.glob("depth_*.tiff")):
            raise RuntimeError(f"scene {scene_id}/{view} has no readable RGB/depth frames")
    return {
        "scene_id": scene_id,
        "views": views,
        "files": sum(1 for path in scene.rglob("*") if path.is_file()),
        "bytes": sum(path.stat().st_size for path in scene.rglob("*") if path.is_file()),
    }


@app.function(
    image=image,
    secrets=[hf_secret, wandb_secret],
    volumes={str(DATA_ROOT): volume},
    cpu=16,
    memory=65536,
    ephemeral_disk=512 * 1024,
    timeout=6 * 60 * 60,
    max_containers=1,
)
def materialize_validation() -> dict[str, object]:
    import wandb

    from mvtracker.datasets.kubric_metadata_index import build_kubric_metadata_index, KubricMetadataIndex

    started = time.perf_counter()
    run = wandb.init(
        entity="jeetucl-ucl",
        project="mvtracker-modal-profiling",
        job_type="mvkubric-validation-setup",
        tags=["modal", "mv-kubric", "validation", "archive"],
        config={
            "repo_id": MVKUBRIC_REPO,
            "revision": MVKUBRIC_REVISION,
            "archive": SOURCE_ARCHIVE,
            "validation_scenes": list(VALIDATION_SCENES),
            "owner": "jeet",
            "project": "mvtracker",
            "purpose": "profiling",
        },
    )
    try:
        staging = Path("/tmp/mvkubric-validation")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        from huggingface_hub import hf_hub_url
        import requests

        url = hf_hub_url(
            repo_id=MVKUBRIC_REPO,
            filename=SOURCE_ARCHIVE,
            repo_type="dataset",
            revision=MVKUBRIC_REVISION,
        )
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
            stream=True,
            timeout=120,
        )
        response.raise_for_status()
        source_root = staging / "kubric-multiview/train"
        source_root.mkdir(parents=True)
        seen = set()
        with tarfile.open(fileobj=response.raw, mode="r|gz") as archive:
            for member in archive:
                parts = Path(member.name).parts
                if len(parts) < 3 or parts[:2] != ("kubric-multiview", "train"):
                    continue
                scene_id = parts[2]
                if scene_id not in VALIDATION_SCENES:
                    if seen == set(VALIDATION_SCENES):
                        break
                    continue
                archive.extract(member, staging)
                seen.add(scene_id)
        response.close()
        if seen != set(VALIDATION_SCENES):
            raise RuntimeError(
                f"{SOURCE_ARCHIVE} at {MVKUBRIC_REVISION} lacks validation scenes "
                f"{sorted(set(VALIDATION_SCENES) - seen)}"
            )
        reports = [_validate_scene(source_root, scene) for scene in VALIDATION_SCENES]
        existing = set(_scene_inventory(TRAIN_ROOT))
        expected_train = {str(scene) for scene in range(900, 1000)}
        if existing != expected_train:
            raise RuntimeError(
                f"refusing to modify volume: expected train scenes 900..999, observed "
                f"{len(existing)} scenes"
            )
        for scene in VALIDATION_SCENES:
            destination = TRAIN_ROOT / scene
            if destination.exists():
                shutil.rmtree(destination)
            shutil.copytree(source_root / scene, destination)
        all_scenes = expected_train | set(VALIDATION_SCENES)
        manifest_path = build_kubric_metadata_index(
            TRAIN_ROOT, index_root=INDEX_ROOT, overwrite=True
        )
        index = KubricMetadataIndex(INDEX_ROOT)
        if set(index.scenes) != all_scenes:
            raise RuntimeError("combined MV-Kubric index scene allowlist is incomplete")
        index.validate_source(TRAIN_ROOT)
        volume.commit()
        result = {
            "repo_id": MVKUBRIC_REPO,
            "revision": MVKUBRIC_REVISION,
            "archive": SOURCE_ARCHIVE,
            "validation_scenes": reports,
            "train_scene_count": len(expected_train),
            "combined_index_scene_count": len(index.scenes),
            "index_manifest": str(manifest_path),
            "index_manifest_sha256": _sha256(manifest_path),
            "elapsed_seconds": time.perf_counter() - started,
        }
        run.summary.update(result)
        return result
    finally:
        run.finish()


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(materialize_validation.remote(), indent=2, sort_keys=True))
