"""Download, index, and publish the official DIEGESIS training set."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time

import modal

from modal_training_profile import (
    DATA_ROOT,
    _dependency_image,
    _source_commit,
    _source_image,
    data_volume,
    wandb_secret,
)
from mvtracker.profiling.modal_continual_training import (
    MODAL_TAGS,
    WANDB_ENTITY,
    WANDB_PROJECT,
    require_pushed_main_commit,
)


APP_NAME = "jeet-mvtracker-diegesis-ingestion"
DOWNLOADER_URL = (
    "https://storage.googleapis.com/dm-tapnet/mv-tap/"
    "diegesis_train/download_diegesis_train.py"
)
DATASET_ROOT = DATA_ROOT / "datasets/diegesis-train"
STAGING_ROOT = DATA_ROOT / "datasets/.diegesis-train-staging"
EXPECTED_SCENES = 351
EXPECTED_TRACKING_BYTES = 490174521611
COPY_WORKERS = 16
DOWNLOAD_WORKERS = 16
CACHE_WORKERS = 16
LOCAL_ROOT = Path("/tmp/diegesis-train")


class _Progress:
    def __init__(self, run):
        self.run = run
        self.phase = "startup"
        self.fields = {}
        self.started = time.perf_counter()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def emit(self, event: str, **fields) -> None:
        payload = {
            "event": event,
            "phase": self.phase,
            "elapsed_seconds": round(time.perf_counter() - self.started, 2),
            **self.fields,
            **fields,
        }
        print("DIEGESIS_INGEST " + json.dumps(payload, sort_keys=True), flush=True)
        self.run.log(
            {
                f"ingestion/{key}": value
                for key, value in payload.items()
                if isinstance(value, (int, float))
            }
        )

    def set_phase(self, phase: str, **fields) -> None:
        self.phase = phase
        self.fields = fields
        self.emit("phase_start")

    def _heartbeat(self) -> None:
        while not self.stop.wait(30):
            usage = shutil.disk_usage("/tmp")
            self.emit(
                "heartbeat",
                local_disk_used_gib=round(usage.used / (1 << 30), 2),
                local_disk_free_gib=round(usage.free / (1 << 30), 2),
            )

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)


def _run_downloader(source_root: Path, progress: _Progress) -> None:
    script = LOCAL_ROOT / "download_diegesis_train.py"
    subprocess.run(["curl", "-fsSLo", str(script), DOWNLOADER_URL], check=True)
    shutil.copyfile(script, source_root / script.name)
    command = [
        "python3",
        str(script),
        "--workers",
        str(DOWNLOAD_WORKERS),
        "--output_dir",
        str(source_root),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    completed = re.compile(r"\[(\d+)/(\d+)\]")
    for line in process.stdout:
        print(line, end="", flush=True)
        match = completed.search(line)
        if match:
            progress.emit(
                "download_progress",
                completed_files=int(match.group(1)),
                total_files=int(match.group(2)),
            )
    return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def _prepare_layout(source_root: Path, progress: _Progress) -> list[dict]:
    index = json.loads((source_root / "diegesis_index.json").read_text())
    scenes = list(index["scenes"])
    if len(scenes) != EXPECTED_SCENES:
        raise RuntimeError(f"expected {EXPECTED_SCENES} scenes, found {len(scenes)}")
    tracking_bytes = sum(int(scene["tracking_bytes"]) for scene in scenes)
    if tracking_bytes != EXPECTED_TRACKING_BYTES:
        raise RuntimeError(
            f"tracking inventory changed: {tracking_bytes} != {EXPECTED_TRACKING_BYTES}"
        )
    raw_root = LOCAL_ROOT / "TAPVid3D_raw/train"
    raw_root.mkdir(parents=True)
    for scene in scenes:
        sequence = source_root / "scenes" / scene["path"] / "tracking/sequence"
        if not sequence.is_dir():
            raise FileNotFoundError(sequence)
        link = raw_root / scene["run_id"]
        link.symlink_to(os.path.relpath(sequence, raw_root), target_is_directory=True)
    progress.emit(
        "layout_ready",
        scenes=len(scenes),
        tracking_gib=round(tracking_bytes / (1 << 30), 3),
    )
    return scenes


def _build_cache(progress: _Progress) -> dict[str, int]:
    from mvtracker.datasets.tapvid3d_multiview_dataset import prepare_tapvid3d_cache

    started = time.perf_counter()

    def report(completed, total, split, scene, result):
        elapsed = max(time.perf_counter() - started, 1e-6)
        progress.emit(
            "cache_progress",
            completed_scenes=completed,
            total_scenes=total,
            scene=scene,
            split=split,
            result=result,
            scenes_per_second=round(completed / elapsed, 3),
            eta_seconds=round((total - completed) / (completed / elapsed), 1),
        )

    return prepare_tapvid3d_cache(
        LOCAL_ROOT / "TAPVid3D_raw",
        LOCAL_ROOT / "TAPVid3D_MVTracker_cache",
        workers=CACHE_WORKERS,
        progress=report,
    )


def _inventory(root: Path):
    files = []
    links = []
    for path in root.rglob("*"):
        if path.is_symlink():
            links.append(path)
        elif path.is_file():
            files.append(path)
    return files, links


def _copy_file(source: Path, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    size = source.stat().st_size
    if destination.is_file() and destination.stat().st_size == size:
        return size
    partial = destination.with_name(destination.name + ".partial")
    with source.open("rb") as input_stream, partial.open("wb") as output_stream:
        while block := input_stream.read(64 << 20):
            output_stream.write(block)
    os.replace(partial, destination)
    return size


def _publish(progress: _Progress) -> dict[str, int]:
    if DATASET_ROOT.exists():
        raise FileExistsError(DATASET_ROOT)
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    files, links = _inventory(LOCAL_ROOT)
    total_bytes = sum(path.stat().st_size for path in files)
    completed_bytes = 0
    completed_files = 0
    started = time.perf_counter()
    lock = threading.Lock()
    next_report = 5 << 30
    with ThreadPoolExecutor(max_workers=COPY_WORKERS) as executor:
        futures = {
            executor.submit(
                _copy_file,
                source,
                STAGING_ROOT / source.relative_to(LOCAL_ROOT),
            ): source
            for source in files
        }
        for future in as_completed(futures):
            copied = future.result()
            with lock:
                completed_bytes += copied
                completed_files += 1
                if completed_bytes >= next_report or completed_files == len(files):
                    elapsed = max(time.perf_counter() - started, 1e-6)
                    rate = completed_bytes / elapsed
                    progress.emit(
                        "publish_progress",
                        completed_files=completed_files,
                        total_files=len(files),
                        completed_gib=round(completed_bytes / (1 << 30), 2),
                        total_gib=round(total_bytes / (1 << 30), 2),
                        gib_per_second=round(rate / (1 << 30), 3),
                        eta_seconds=round((total_bytes - completed_bytes) / rate, 1),
                    )
                    next_report = completed_bytes + (5 << 30)
    for source in links:
        destination = STAGING_ROOT / source.relative_to(LOCAL_ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(os.readlink(source), target_is_directory=True)
    staged_files, staged_links = _inventory(STAGING_ROOT)
    staged_bytes = sum(path.stat().st_size for path in staged_files)
    if len(staged_files) != len(files) or len(staged_links) != len(links):
        raise RuntimeError("published file inventory differs from local inventory")
    if staged_bytes != total_bytes:
        raise RuntimeError("published byte count differs from local inventory")
    STAGING_ROOT.rename(DATASET_ROOT)
    return {
        "files": len(files),
        "links": len(links),
        "bytes": total_bytes,
    }


image = _source_image(_dependency_image().apt_install("curl"))
app = modal.App(APP_NAME, tags={**MODAL_TAGS, "experiment": "diegesis-train-ingestion"})


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=32,
    memory=64 * 1024,
    ephemeral_disk=1024 * 1024,
    timeout=4 * 60 * 60,
    max_containers=1,
    retries=0,
    include_source=False,
)
def ingest_remote() -> dict:
    import wandb

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        group="diegesis-data-ingestion",
        job_type="dataset-ingestion",
        name="diegesis-train-351",
        tags=["modal", "cpu", "diegesis", "351-scenes"],
        config={
            "source_commit": _source_commit(),
            "download_workers": DOWNLOAD_WORKERS,
            "cache_workers": CACHE_WORKERS,
            "copy_workers": COPY_WORKERS,
            "expected_scenes": EXPECTED_SCENES,
            **MODAL_TAGS,
        },
    )
    progress = _Progress(run)
    try:
        LOCAL_ROOT.mkdir(parents=True, exist_ok=False)
        source_root = LOCAL_ROOT / "source"
        source_root.mkdir()
        progress.set_phase("download", expected_gib=round(EXPECTED_TRACKING_BYTES / (1 << 30), 3))
        _run_downloader(source_root, progress)
        progress.set_phase("layout")
        scenes = _prepare_layout(source_root, progress)
        progress.set_phase("jpeg_cache", scenes=len(scenes))
        cache = _build_cache(progress)
        progress.set_phase("volume_publish")
        published = _publish(progress)
        manifest = {
            "format": "diegesis-train-v1",
            "source_url": DOWNLOADER_URL,
            "source_commit": _source_commit(),
            "scene_count": len(scenes),
            "scene_ids": sorted(scene["run_id"] for scene in scenes),
            "tracking_bytes": EXPECTED_TRACKING_BYTES,
            "cache": cache,
            "published": published,
        }
        (DATASET_ROOT / "ingestion_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        data_volume.commit()
        progress.emit("complete", scenes=len(scenes), published_bytes=published["bytes"])
        run.summary.update({"status": "complete", **manifest})
        run.finish()
        return manifest
    except BaseException:
        run.summary["status"] = "failed"
        run.finish(exit_code=1)
        raise
    finally:
        progress.close()


@app.local_entrypoint()
def main() -> None:
    commit = _source_commit()
    require_pushed_main_commit(commit)
    app.set_tags(
        {
            "owner": "jeet",
            "project": "mvtracker",
            "purpose": "training",
            "experiment": "diegesis-train-ingestion",
            "gpu": "cpu",
        }
    )
    deployed = modal.Function.from_name(APP_NAME, "ingest_remote")
    call = deployed.spawn()
    print(json.dumps({"function_call_id": call.object_id}, indent=2))
