"""Split the pinned MV-Kubric archive into four concurrent tar.zst shards."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

import modal


DATA_ROOT = Path("/mnt/mvtracker-data")
SOURCE_ARCHIVE = DATA_ROOT / "archives/mvkubric/kubric-multiview--train.micro.0900-0999.tar.gz"
OUTPUT_ROOT = DATA_ROOT / "archives/mvkubric/zstd"
LOCAL_ROOT = Path("/tmp/mvkubric-shard-work")
SCENES = tuple(str(scene) for scene in range(900, 1000))
GROUPS = tuple(tuple(SCENES[index : index + 25]) for index in range(0, 100, 25))

app = modal.App(
    "jeet-mvkubric-shard",
    tags={"owner": "jeet", "project": "mvtracker", "purpose": "profiling"},
)
image = modal.Image.debian_slim(python_version="3.11").apt_install("zstd").pip_install("wandb")
volume = modal.Volume.from_name("jeet-mvtracker-data-v2", version=2)
wandb_secret = modal.Secret.from_name("jeet-mvtracker-wandb", required_keys=["WANDB_API_KEY"])


def _log(event: str, **fields: object) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True)
    subprocess.run(
        ["tar", "--extract", "--gzip", "--file", str(source), "--directory", str(destination)],
        check=True,
    )
    candidates = [destination / "datasets/kubric-multiview/train", destination / "kubric-multiview/train"]
    roots = [root for root in candidates if root.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"could not identify unique MV-Kubric train root: {candidates}")
    return roots[0]


def _scene_inventory(train_root: Path) -> list[str]:
    observed = sorted(path.name for path in train_root.iterdir() if path.is_dir() and path.name.isdigit())
    if observed != list(SCENES):
        raise RuntimeError(f"expected scenes 900..999 exactly, observed {observed[:3]}..{observed[-3:]}")
    return observed


def _create_shard(train_root: Path, scenes: tuple[str, ...], output: Path) -> dict[str, object]:
    partial = output.with_suffix(output.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    tar = subprocess.Popen(
        [
            "tar",
            "--create",
            "--file",
            "-",
            "--directory",
            str(train_root),
            "--transform",
            "s|^|datasets/kubric-multiview/train/|",
            *scenes,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    zstd = subprocess.Popen(["zstd", "-T4", "-3", "--quiet", "--stdout"], stdin=tar.stdout, stdout=partial.open("wb"), stderr=subprocess.PIPE)
    assert tar.stdout is not None
    tar.stdout.close()
    zstd_stderr = zstd.communicate()[1]
    tar_stderr = tar.stderr.read() if tar.stderr is not None else b""
    tar_code = tar.wait()
    if tar_code or zstd.returncode:
        raise RuntimeError(f"shard failed tar={tar_code} zstd={zstd.returncode}: {(tar_stderr + zstd_stderr).decode(errors='replace')}")
    partial.replace(output)
    return {"scenes": list(scenes), "bytes": output.stat().st_size, "sha256": _sha256(output)}


def _verify_listing(path: Path, expected: tuple[str, ...]) -> None:
    listing = subprocess.run(["tar", "--list", "--zstd", "--file", str(path)], check=True, capture_output=True, text=True).stdout.splitlines()
    found = {scene for scene in expected if any(item.startswith(f"datasets/kubric-multiview/train/{scene}/") or item == f"datasets/kubric-multiview/train/{scene}" for item in listing)}
    if found != set(expected):
        raise RuntimeError(f"{path.name}: listing does not cover exactly its assigned scenes")
    extras = {item.split("/")[3] for item in listing if item.startswith("datasets/kubric-multiview/train/") and len(item.split("/")) > 3 and item.split("/")[3].isdigit()} - set(expected)
    if extras:
        raise RuntimeError(f"{path.name}: unexpected scenes {sorted(extras)}")


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={str(DATA_ROOT): volume},
    cpu=16,
    memory=32768,
    ephemeral_disk=512 * 1024,
    timeout=4 * 60 * 60,
    max_containers=1,
)
def shard_archive() -> dict[str, object]:
    import wandb

    run = wandb.init(entity="jeetucl-ucl", project="mvtracker-modal-profiling", job_type="mvkubric-shard", tags=["modal", "cpu", "mv-kubric", "archive"], config={"owner": "jeet", "project": "mvtracker", "purpose": "profiling"})
    if not SOURCE_ARCHIVE.is_file():
        raise FileNotFoundError(SOURCE_ARCHIVE)
    if LOCAL_ROOT.exists():
        shutil.rmtree(LOCAL_ROOT)
    LOCAL_ROOT.mkdir()
    _log("staging_started", source=str(SOURCE_ARCHIVE))
    started = time.perf_counter()
    train_root = _extract(SOURCE_ARCHIVE, LOCAL_ROOT / "expanded")
    scenes = _scene_inventory(train_root)
    _log("extraction_complete", seconds=time.perf_counter() - started, scene_count=len(scenes))
    outputs = [LOCAL_ROOT / f"mvkubric-train-{index:02d}.tar.zst" for index in range(4)]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as executor:
        shard_results = list(executor.map(lambda args: _create_shard(train_root, *args), zip(GROUPS, outputs)))
    _log("shards_created", seconds=time.perf_counter() - started, shards=shard_results)
    for path, group in zip(outputs, GROUPS):
        _verify_listing(path, group)
    if sorted(scene for result in shard_results for scene in result["scenes"]) != list(SCENES):
        raise RuntimeError("shard scene coverage is not exactly 900..999")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {"format": "mvkubric_tar_zstd_shards", "source": str(SOURCE_ARCHIVE.relative_to(DATA_ROOT)), "shards": []}
    for path, result in zip(outputs, shard_results):
        destination = OUTPUT_ROOT / path.name
        partial = destination.with_suffix(destination.suffix + ".partial")
        shutil.copyfile(path, partial)
        if _sha256(partial) != result["sha256"]:
            raise RuntimeError(f"destination hash mismatch: {destination}")
        partial.replace(destination)
        manifest["shards"].append({**result, "path": str(destination.relative_to(DATA_ROOT))})
    manifest_path = OUTPUT_ROOT / "manifest.json"
    manifest_partial = manifest_path.with_suffix(".json.partial")
    manifest_partial.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_partial.replace(manifest_path)
    volume.commit()
    _log("volume_commit_complete", manifest=str(manifest_path.relative_to(DATA_ROOT)))
    run.summary.update(manifest)
    run.finish()
    return manifest


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(shard_archive.remote(), indent=2, sort_keys=True))
