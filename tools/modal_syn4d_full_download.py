"""Download every Syn4D stride-1 environment archive to the Modal Volume."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import threading
import time

import modal

from mvtracker.profiling.modal_syn4d_split import (
    ARCHIVE_BYTES,
    LEGACY_ARCHIVES,
    SYN4D_REPO_ID,
    SYN4D_REVISION,
    SYN4D_SUBSET,
)


APP_NAME = "jeet-mvtracker-syn4d-full-download"
DATA_VOLUME_NAME = "jeet-mvtracker-data-v2"
DATA_ROOT = Path("/mnt/mvtracker-data")
OUTPUT_ROOT = DATA_ROOT / "datasets/syn4d/full-stride1"
ARCHIVE_ROOT = OUTPUT_ROOT / "source"
MANIFEST_PATH = OUTPUT_ROOT / "download-manifest.json"
MAPPING_PATH = OUTPUT_ROOT / "sequence_to_asset_mapping.csv"
MAPPING_SOURCE = f"{SYN4D_SUBSET}/sequence_to_asset_mapping.csv"
BASE_TAGS = {"owner": "jeet", "project": "mvtracker", "purpose": "profiling"}
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _source_commit() -> str:
    commit = os.environ.get("MVTRACKER_MODAL_COMMIT", "")
    if COMMIT_RE.fullmatch(commit) is None:
        raise RuntimeError("MVTRACKER_MODAL_COMMIT must be a full lowercase Git commit")
    return commit


def _image() -> modal.Image:
    commit = _source_commit()
    clone = (
        "git init /opt/mvtracker && "
        "git -C /opt/mvtracker remote add origin https://github.com/J999UCL/mvtracker.git && "
        f"git -C /opt/mvtracker fetch --depth=1 origin {commit} && "
        "git -C /opt/mvtracker checkout --detach FETCH_HEAD && "
        f'test "$(git -C /opt/mvtracker rev-parse HEAD)" = "{commit}"'
    )
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ca-certificates", "git", "zstd")
        .pip_install("hf-xet==1.6.0", "huggingface-hub==1.28.0", "wandb==0.19.9")
        .run_commands(clone)
        .env({"MVTRACKER_MODAL_COMMIT": commit, "PYTHONPATH": "/opt/mvtracker:/opt/mvtracker/tools"})
    )


app = modal.App(APP_NAME, tags={**BASE_TAGS, "experiment": "syn4d-full-stride1-download", "gpu": "cpu"})
image = _image()
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True, version=2)
hf_secret = modal.Secret.from_name("jeet-mvtracker-huggingface", required_keys=["HF_TOKEN"])
wandb_secret = modal.Secret.from_name("jeet-mvtracker-wandb", required_keys=["WANDB_API_KEY"])


def _emit(run, event: str, **fields: object) -> None:
    payload = {"event": event, **fields}
    print(json.dumps(payload, sort_keys=True), flush=True)
    if run is not None:
        run.log(
            {
                f"progress/{key}": value
                for key, value in payload.items()
                if key != "event" and isinstance(value, (int, float))
            },
            commit=True,
        )


def _heartbeat(stop: threading.Event, run, phase: dict[str, object], started: float) -> None:
    while not stop.wait(30.0):
        target = Path(str(phase.get("target", "/tmp")))
        try:
            usage = shutil.disk_usage(target if target.exists() else Path("/tmp"))
            free_gib = usage.free / (1 << 30)
        except OSError:
            free_gib = None
        _emit(
            run,
            "heartbeat",
            phase=phase.get("name"),
            environment=phase.get("environment"),
            elapsed_seconds=round(time.perf_counter() - started, 1),
            local_disk_free_gib=None if free_gib is None else round(free_gib, 2),
        )


def _existing_archive(environment: str, expected_bytes: int) -> tuple[Path, str] | None:
    legacy = LEGACY_ARCHIVES.get(environment)
    if legacy is not None:
        path = DATA_ROOT / legacy
        if path.is_file() and path.stat().st_size == expected_bytes:
            return path, "existing_legacy"
    target = ARCHIVE_ROOT / f"{environment}.tar.zst"
    if target.is_file() and target.stat().st_size == expected_bytes:
        return target, "existing_full_root"
    return None


def _download_one(environment: str, expected_bytes: int, token: str, run) -> dict[str, object]:
    from huggingface_hub import hf_hub_download

    existing = _existing_archive(environment, expected_bytes)
    if existing is not None:
        path, status = existing
        _emit(run, "archive_reused", environment=environment, path=str(path), bytes=expected_bytes, source=status)
        return {"environment": environment, "path": str(path), "bytes": expected_bytes, "status": status}

    target = ARCHIVE_ROOT / f"{environment}.tar.zst"
    partial = target.with_suffix(target.suffix + ".partial")
    if partial.exists():
        partial.unlink()
    work = Path("/tmp/syn4d-full-download") / environment
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    phase = {"name": "download", "environment": environment, "target": str(work)}
    started = time.perf_counter()
    stop = threading.Event()
    watcher = threading.Thread(target=_heartbeat, args=(stop, run, phase, started), daemon=True)
    watcher.start()
    _emit(
        run,
        "archive_download_start",
        environment=environment,
        expected_bytes=expected_bytes,
        expected_gib=round(expected_bytes / (1 << 30), 2),
        filename=f"{SYN4D_SUBSET}/{environment}.tar.zst",
    )
    try:
        source = Path(
            hf_hub_download(
                repo_id=SYN4D_REPO_ID,
                repo_type="dataset",
                revision=SYN4D_REVISION,
                filename=f"{SYN4D_SUBSET}/{environment}.tar.zst",
                token=token,
                local_dir=str(work),
                cache_dir=str(work / "hf-cache"),
            )
        )
        observed = source.stat().st_size
        if observed != expected_bytes:
            raise RuntimeError(f"{environment}: expected {expected_bytes} bytes, downloaded {observed}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, partial)
        if partial.stat().st_size != expected_bytes:
            raise RuntimeError(f"{environment}: Volume copy size mismatch")
        partial.replace(target)
        elapsed = time.perf_counter() - started
        _emit(run, "archive_download_complete", environment=environment, path=str(target), bytes=observed, elapsed_seconds=round(elapsed, 2))
        return {"environment": environment, "path": str(target), "bytes": observed, "status": "downloaded", "elapsed_seconds": elapsed}
    finally:
        stop.set()
        watcher.join(timeout=2)
        shutil.rmtree(work, ignore_errors=True)


@app.function(
    image=image,
    secrets=[hf_secret, wandb_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=16,
    memory=32 * 1024,
    ephemeral_disk=1024 * 1024,
    timeout=24 * 60 * 60,
    retries=0,
    max_containers=1,
    include_source=False,
)
def download_all() -> dict[str, object]:
    import wandb
    from huggingface_hub import hf_hub_download

    run = wandb.init(
        project="mvtracker-modal-profiling",
        job_type="syn4d-full-stride1-download",
        name="syn4d-full-stride1-download",
        tags=["modal", "cpu", "syn4d", "stride1", "download"],
        config={**BASE_TAGS, "source_revision": SYN4D_REVISION, "archive_count": len(ARCHIVE_BYTES)},
    )
    started = time.perf_counter()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    _emit(run, "startup", source_revision=SYN4D_REVISION, archive_count=len(ARCHIVE_BYTES), expected_bytes=sum(ARCHIVE_BYTES.values()))

    if not MAPPING_PATH.is_file():
        _emit(run, "mapping_download_start", filename=MAPPING_SOURCE)
        mapping = Path(
            hf_hub_download(
                repo_id=SYN4D_REPO_ID,
                repo_type="dataset",
                revision=SYN4D_REVISION,
                filename=MAPPING_SOURCE,
                token=os.environ["HF_TOKEN"],
                local_dir="/tmp/syn4d-mapping",
            )
        )
        shutil.copyfile(mapping, MAPPING_PATH)
        _emit(run, "mapping_download_complete", path=str(MAPPING_PATH), bytes=MAPPING_PATH.stat().st_size)
    else:
        _emit(run, "mapping_reused", path=str(MAPPING_PATH), bytes=MAPPING_PATH.stat().st_size)

    results: list[dict[str, object]] = []
    for environment in sorted(ARCHIVE_BYTES):
        try:
            result = _download_one(environment, ARCHIVE_BYTES[environment], os.environ["HF_TOKEN"], run)
            results.append(result)
            manifest = {
                "format": "mvtracker_syn4d_full_stride1_download",
                "source_revision": SYN4D_REVISION,
                "subset": SYN4D_SUBSET,
                "mapping": str(MAPPING_PATH),
                "archives": results,
                "complete_count": len(results),
                "expected_count": len(ARCHIVE_BYTES),
            }
            temporary = MANIFEST_PATH.with_suffix(".json.partial")
            temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(MANIFEST_PATH)
            data_volume.commit()
            _emit(run, "archive_committed", environment=environment, complete_count=len(results), expected_count=len(ARCHIVE_BYTES))
        except Exception as error:
            _emit(run, "archive_failed", environment=environment, error_type=type(error).__name__, error=str(error))
            run.summary["failure"] = f"{environment}: {type(error).__name__}: {error}"
            run.finish(exit_code=1)
            raise

    result = {
        "format": "mvtracker_syn4d_full_stride1_download",
        "source_revision": SYN4D_REVISION,
        "archive_count": len(results),
        "archive_bytes": sum(int(item["bytes"]) for item in results),
        "elapsed_seconds": time.perf_counter() - started,
        "manifest": str(MANIFEST_PATH),
        "archives": results,
    }
    run.summary.update({key: value for key, value in result.items() if key != "archives"})
    run.finish()
    _emit(run, "complete", archive_count=len(results), elapsed_seconds=round(result["elapsed_seconds"], 2))
    return result


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(download_all.remote(), indent=2, sort_keys=True))


if __name__ == "__main__":
    print("Use: modal run --timestamps tools/modal_syn4d_full_download.py")
