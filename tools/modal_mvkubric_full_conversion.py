"""Convert the pinned MV-Kubric archives into indexed scene/view WebDataset TARs.

The job deliberately keeps the orchestration here.  The converter owns shard
formatting; this module only stages one source archive at a time on local SSD,
reports progress, and calls the converter with globally unique shard offsets.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any, Iterable

import modal

from modal_training_profile import (
    BASE_TAGS,
    DATA_ROOT,
    _source_commit,
    data_volume,
    wandb_secret,
)


APP_NAME = "jeet-mvtracker-mvkubric-full-conversion"
WANDB_ENTITY = "jeetucl-ucl"
WANDB_PROJECT = "mvtracker-modal-profiling"
MODAL_TAGS = {**BASE_TAGS, "experiment": "mvkubric-webdataset-full-conversion"}

ARCHIVE_ROOT = DATA_ROOT / "archives/mvkubric/2000-scenes-v1"
TRAIN_OUTPUT = DATA_ROOT / "datasets/kubric-multiview-webdataset/train"
VALIDATION_SOURCE = DATA_ROOT / "datasets/kubric-multiview/2000-scenes-v1/validation"
VALIDATION_OUTPUT = DATA_ROOT / "datasets/kubric-multiview-webdataset/validation"
MANIFEST_PATH = DATA_ROOT / "direct-volume-data-manifest.json"
LOCAL_ROOT = Path("/tmp/mvkubric-full-conversion")
SCENES_PER_SHARD = 4
SHARD_WORKERS = 8
READ_WORKERS = 2
HEARTBEAT_SECONDS = 30.0
COPY_PROGRESS_BYTES = 10 * (1 << 30)

TRAIN_ARCHIVES = (
    ("kubric-multiview--train.full.1001-2000.tar.gz", 1001, 2000),
    ("kubric-multiview--train.full.2001-3000.tar.gz", 2001, 3000),
)
VALIDATION_SCENES = tuple(str(scene) for scene in range(101, 128))


def _conversion_image() -> modal.Image:
    """Use a small CPU image; the full CUDA training image is unnecessary here."""

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
        .apt_install("git", "zstd")
        .pip_install("numpy", "wandb", "wids==0.1.11", "rapidgzip==0.16.0")
        .run_commands(clone)
        .env(
            {
                "MVTRACKER_MODAL_COMMIT": commit,
                "PYTHONPATH": "/opt/mvtracker:/opt/mvtracker/tools",
            }
        )
    )


app = modal.App(APP_NAME, tags=MODAL_TAGS)
image = _conversion_image()


def _read_proc_mem() -> tuple[float | None, float | None]:
    try:
        values = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            name, value, *_ = line.split()
            values[name.rstrip(":")] = int(value) * 1024
        total = values["MemTotal"] / (1 << 30)
        available = values["MemAvailable"] / (1 << 30)
        return total - available, total
    except (FileNotFoundError, KeyError, ValueError):
        return None, None


class _Progress:
    """Flush progress events and heartbeat metrics while work is blocked."""

    def __init__(self, run: Any, root: Path):
        self.run = run
        self.root = root
        self.started = time.perf_counter()
        self._lock = threading.Lock()
        self._phase = "starting"
        self._fields: dict[str, object] = {}
        self._cpu_previous: tuple[int, int] | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()

    def _resource_fields(self) -> dict[str, object]:
        used, total = _read_proc_mem()
        cpu_percent = None
        try:
            values = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
            current = (sum(int(value) for value in values), int(values[3]))
            if self._cpu_previous is not None:
                total_delta = current[0] - self._cpu_previous[0]
                idle_delta = current[1] - self._cpu_previous[1]
                cpu_percent = round(100 * (total_delta - idle_delta) / max(total_delta, 1), 1)
            self._cpu_previous = current
        except (FileNotFoundError, IndexError, ValueError):
            pass
        try:
            usage = shutil.disk_usage(self.root)
            disk = {
                "local_disk_used_gib": round(usage.used / (1 << 30), 2),
                "local_disk_free_gib": round(usage.free / (1 << 30), 2),
            }
        except OSError:
            disk = {"local_disk_used_gib": None, "local_disk_free_gib": None}
        return {
            "elapsed_seconds": round(time.perf_counter() - self.started, 2),
            "cpu_count": os.cpu_count(),
            "cpu_percent": cpu_percent,
            "ram_used_gib": None if used is None else round(used, 2),
            "ram_total_gib": None if total is None else round(total, 2),
            **disk,
        }

    def emit(self, event: str, **fields: object) -> None:
        with self._lock:
            payload = {
                "event": event,
                "phase": self._phase,
                **self._resource_fields(),
                **self._fields,
                **fields,
            }
            print(json.dumps(payload, sort_keys=True), flush=True)
            try:
                self.run.log(
                    {
                        f"progress/{key}": value
                        for key, value in payload.items()
                        if key != "event" and isinstance(value, (int, float))
                    },
                        commit=True,
                )
            except Exception:
                # Progress reporting must not hide the conversion failure.
                pass

    def phase(self, phase: str, **fields: object) -> None:
        with self._lock:
            self._phase = phase
            self._fields = fields
        self.emit("phase_start")

    def update(self, **fields: object) -> None:
        with self._lock:
            self._fields.update(fields)

    def _heartbeat(self) -> None:
        while not self._stop.wait(HEARTBEAT_SECONDS):
            # Do not recursively scan the extracted tree while rapidgzip/tar
            # is running.  That would compete with the extraction itself.
            # Exact file/byte counts are collected once at phase completion.
            self.emit("heartbeat")

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.emit("job_end")


def _directory_stats(root: Path) -> tuple[int | None, int | None]:
    """Best-effort file/byte counts for a heartbeat without failing the job."""

    if not root.exists():
        return 0, 0
    try:
        completed = subprocess.run(
            ["find", str(root), "-type", "f", "-printf", "%s\\n"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if completed.returncode:
            return None, None
        sizes = [int(value) for value in completed.stdout.split()]
        return len(sizes), sum(sizes)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None, None


def _copy_archive(source: Path, destination: Path, progress: _Progress) -> dict[str, object]:
    started = time.perf_counter()
    copied = 0
    next_report = COPY_PROGRESS_BYTES
    total_bytes = source.stat().st_size
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_stream, destination.open("wb") as target_stream:
        while True:
            block = source_stream.read(64 * (1 << 20))
            if not block:
                break
            target_stream.write(block)
            copied += len(block)
            progress.update(
                archive_bytes=copied,
                archive_total_bytes=total_bytes,
                archive_percent=100 * copied / total_bytes,
            )
            if copied >= next_report:
                elapsed = time.perf_counter() - started
                progress.emit(
                    "copy_progress",
                    copied_gib=round(copied / (1 << 30), 2),
                    copy_gib_per_second=round(copied / (1 << 30) / max(elapsed, 1e-6), 3),
                    copy_eta_seconds=round((total_bytes - copied) / max(copied / max(elapsed, 1e-6), 1), 1),
                )
                next_report += COPY_PROGRESS_BYTES
        target_stream.flush()
        os.fsync(target_stream.fileno())
    elapsed = time.perf_counter() - started
    progress.emit(
        "copy_complete",
        copied_gib=round(copied / (1 << 30), 2),
        copy_gib_per_second=round(copied / (1 << 30) / max(elapsed, 1e-6), 3),
        copy_seconds=round(elapsed, 2),
    )
    return {"bytes": copied, "seconds": elapsed}


def _extract_archive(archive: Path, destination: Path, progress: _Progress) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    decompressor = subprocess.Popen(
        ["rapidgzip", "-d", "-c", "-P", "16", str(archive)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    extractor = subprocess.Popen(
        [
            "tar",
            "--extract",
            "--strip-components=2",
            "--file=-",
            "--directory",
            str(destination),
        ],
        stdin=decompressor.stdout,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert decompressor.stdout is not None
    decompressor.stdout.close()
    _, tar_stderr = extractor.communicate()
    gzip_stderr = decompressor.stderr.read() if decompressor.stderr is not None else b""
    gzip_code = decompressor.wait()
    elapsed = time.perf_counter() - started
    if gzip_code or extractor.returncode:
        raise RuntimeError(
            "archive extraction failed "
            f"rapidgzip={gzip_code} tar={extractor.returncode}: "
            f"{(gzip_stderr + (tar_stderr or b''))[-4000:].decode(errors='replace')}"
        )
    files, bytes_ = _directory_stats(destination)
    progress.emit(
        "extract_complete",
        extracted_files=files,
        extracted_bytes=bytes_,
        extraction_seconds=round(elapsed, 2),
        extraction_gib_per_second=None if bytes_ is None else round(bytes_ / (1 << 30) / max(elapsed, 1e-6), 3),
    )
    return {"seconds": elapsed, "files": files, "bytes": bytes_}


def _scene_allowlist() -> tuple[str, ...]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"required scene manifest is missing: {MANIFEST_PATH}")
    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        scenes = payload["train_scene_ids"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"invalid scene manifest: {MANIFEST_PATH}") from error
    if not isinstance(scenes, list) or len(scenes) != 1992:
        raise RuntimeError(f"expected 1,992 train scene IDs in {MANIFEST_PATH}")
    ordered = tuple(sorted((str(scene) for scene in scenes), key=int))
    if len(set(ordered)) != len(ordered):
        raise RuntimeError(f"scene manifest contains duplicate IDs: {MANIFEST_PATH}")
    return ordered


def _scenes_in_range(root: Path, start: int, end: int, allowlist: Iterable[str] | None) -> tuple[str, ...]:
    from mvtracker.preprocessing.mvkubric_webdataset import discover_scene_ids

    allowed = {str(scene) for scene in allowlist} if allowlist is not None else None
    selected = discover_scene_ids(root, allowed)
    return tuple(scene for scene in selected if start <= int(scene) <= end)


def _conversion_progress(
    progress: _Progress,
    archive_name: str,
    total_scenes: int,
    total_shards: int,
    commit_volume,
):
    state_lock = threading.Lock()
    started = time.perf_counter()
    completed_scenes = 0

    def callback(event: object, *values: object) -> None:
        nonlocal completed_scenes
        if event == "shard":
            result, completed, total = values
            commit_volume()
            progress.emit(
                "shard_complete",
                archive=archive_name,
                shard=str(result["name"]),
                shard_completed=int(completed),
                shard_total=int(total),
                shard_bytes=int(result["bytes"]),
                shard_seconds=float(result["seconds"]),
            )
            return
        shard, scene_id, completed, seconds = (event, *values)
        with state_lock:
            completed_scenes += 1
            done = completed_scenes
        rate = done / max(time.perf_counter() - started, 1e-6)
        progress.emit(
            "scene_complete",
            archive=archive_name,
            shard=str(shard.name),
            scene=str(scene_id),
            scene_completed=int(completed),
            scene_total=len(shard.scene_ids),
            archive_scene_total=total_scenes,
            conversion_shards_total=total_shards,
            conversion_scene_completed=done,
            conversion_scene_total=total_scenes,
            conversion_eta_seconds=round((total_scenes - done) / max(rate, 1e-6), 1),
            scene_seconds=float(seconds),
        )

    return callback


def _clear_partials(root: Path) -> int:
    removed = 0
    if root.exists():
        for path in root.rglob("*.partial"):
            if path.is_file():
                path.unlink()
                removed += 1
    return removed


def _read_published_manifest(root: Path) -> dict[str, object]:
    manifest_path = Path(root) / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"canonical output is incomplete: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read canonical manifest: {manifest_path}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("scene_ids"), list):
        raise RuntimeError(f"canonical manifest is invalid: {manifest_path}")
    return payload


def _convert_archive(
    archive_name: str,
    start: int,
    end: int,
    output_root: Path,
    progress: _Progress,
    allowlist: tuple[str, ...],
    shard_offset: int,
    commit_volume,
) -> tuple[tuple[str, ...], dict[str, object]]:
    from mvtracker.preprocessing.mvkubric_webdataset import convert_shards

    expected_ids = tuple(scene for scene in allowlist if start <= int(scene) <= end)
    shard_count = (len(expected_ids) + SCENES_PER_SHARD - 1) // SCENES_PER_SHARD
    expected_names = tuple(
        f"mvkubric-{shard_offset + index:05d}" for index in range(shard_count)
    )
    if all(
        (output_root / f"{name}.tar").is_file()
        and (output_root / f"{name}.inventory.json").is_file()
        for name in expected_names
    ):
        progress.emit(
            "archive_already_converted",
            archive=archive_name,
            scene_count=len(expected_ids),
            shard_count=shard_count,
            shard_offset=shard_offset,
        )
        return expected_ids, {
            "scene_ids": list(expected_ids),
            "shards": [{"name": name} for name in expected_names],
        }

    source = ARCHIVE_ROOT / archive_name
    local_archive = LOCAL_ROOT / archive_name
    local_extract = LOCAL_ROOT / "native"
    if not source.is_file():
        raise FileNotFoundError(source)
    if local_extract.exists():
        shutil.rmtree(local_extract)
    local_extract.mkdir(parents=True)
    progress.phase("copy", archive=archive_name)
    _copy_archive(source, local_archive, progress)
    progress.phase("extract", archive=archive_name, extraction_root=str(local_extract))
    _extract_archive(local_archive, local_extract, progress)
    local_archive.unlink()
    progress.emit("local_archive_deleted", archive=archive_name)
    observed_ids = _scenes_in_range(local_extract, start, end, None)
    if observed_ids != expected_ids:
        raise RuntimeError(
            f"{archive_name}: extracted scene IDs do not match the pinned manifest "
            f"(expected {len(expected_ids)}, observed {len(observed_ids)})"
        )
    scene_ids = expected_ids
    shard_count = (len(scene_ids) + SCENES_PER_SHARD - 1) // SCENES_PER_SHARD
    progress.phase(
        "convert",
        archive=archive_name,
        archive_scene_count=len(scene_ids),
        archive_shard_count=shard_count,
        shard_offset=shard_offset,
    )
    result = convert_shards(
        local_extract,
        output_root,
        scene_ids,
        scenes_per_shard=SCENES_PER_SHARD,
        shard_workers=SHARD_WORKERS,
        read_workers=READ_WORKERS,
        shard_offset=shard_offset,
        finalize=False,
        progress_callback=_conversion_progress(
            progress, archive_name, len(scene_ids), shard_count, commit_volume
        ),
    )
    shutil.rmtree(local_extract)
    progress.emit("local_extraction_deleted", archive=archive_name)
    return scene_ids, result


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=16,
    memory=32768,
    ephemeral_disk=1024 * 1024,
    timeout=24 * 60 * 60,
    retries=2,
    max_containers=1,
    include_source=False,
)
def convert_full(
    *,
    train_output: str = str(TRAIN_OUTPUT),
    validation_output: str = str(VALIDATION_OUTPUT),
    validation_source: str = str(VALIDATION_SOURCE),
) -> dict[str, object]:
    import wandb

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        job_type="mvkubric-webdataset-full-conversion",
        tags=["modal", "cpu", "mv-kubric", "webdataset", "full-conversion"],
        config={"source_commit": _source_commit(), **MODAL_TAGS},
    )
    progress = _Progress(run, LOCAL_ROOT)
    started = time.perf_counter()
    output_root = Path(train_output)
    validation_root = Path(validation_output)
    staging_root = output_root.with_name(output_root.name + ".staging")
    validation_staging = validation_root.with_name(validation_root.name + ".staging")
    try:
        print(
            json.dumps(
                {
                    "event": "startup",
                    "modal_app_id": os.environ.get("MODAL_APP_ID"),
                    "wandb_url": getattr(run, "url", None),
                    "output_root": str(output_root),
                    "validation_output": str(validation_root),
                    "local_root": str(LOCAL_ROOT),
                    "source_commit": _source_commit(),
                    **MODAL_TAGS,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        progress.emit("startup")
        allowlist = _scene_allowlist()
        _clear_partials(staging_root)
        _clear_partials(validation_staging)
        train_scene_ids: list[str] = list(allowlist)
        shard_offset = 0
        archive_results: list[dict[str, object]] = []
        commit_lock = threading.Lock()

        def commit_volume() -> None:
            with commit_lock:
                data_volume.commit()

        if output_root.is_dir():
            train_manifest = _read_published_manifest(output_root)
            shard_offset = len(train_manifest.get("shards", []))
            progress.emit("train_already_published", scene_count=len(train_scene_ids), shard_count=shard_offset)
        else:
            staging_root.mkdir(parents=True, exist_ok=True)

            for archive_name, start, end in TRAIN_ARCHIVES:
                scene_ids, result = _convert_archive(
                    archive_name,
                    start,
                    end,
                    staging_root,
                    progress,
                    allowlist,
                    shard_offset,
                    commit_volume,
                )
                shard_offset += (len(scene_ids) + SCENES_PER_SHARD - 1) // SCENES_PER_SHARD
                archive_results.append(
                    {"archive": archive_name, "scene_count": len(scene_ids), "shard_count": len(result["shards"])}
                )
                commit_volume()

        from mvtracker.preprocessing.mvkubric_webdataset import convert_shards, finalize_shards

        if not output_root.is_dir():
            progress.phase("finalize_train", scene_count=len(train_scene_ids), shard_count=shard_offset)
            train_manifest = finalize_shards(
                staging_root, allowlist, scenes_per_shard=SCENES_PER_SHARD
            )
        progress.phase("convert_validation", scene_count=len(VALIDATION_SCENES))
        if validation_root.is_dir():
            validation_manifest = _read_published_manifest(validation_root)
            validation_ids = tuple(str(scene) for scene in validation_manifest["scene_ids"])
            if validation_ids != VALIDATION_SCENES:
                raise RuntimeError("published validation scene IDs are not 101-127")
            validation_result = {"shards": validation_manifest.get("shards", [])}
            progress.emit("validation_already_published", scene_count=len(validation_ids))
        else:
            validation_source_root = Path(validation_source)
            if not validation_source_root.is_dir():
                raise FileNotFoundError(validation_source_root)
            observed_validation = _scenes_in_range(validation_source_root, 0, 10**9, None)
            if observed_validation != VALIDATION_SCENES:
                raise RuntimeError("validation source does not contain exactly scenes 101-127")
            validation_staging.mkdir(parents=True, exist_ok=True)
            validation_ids = VALIDATION_SCENES
            validation_result = convert_shards(
                validation_source_root,
                validation_staging,
                validation_ids,
                scenes_per_shard=SCENES_PER_SHARD,
                shard_workers=SHARD_WORKERS,
                read_workers=READ_WORKERS,
                finalize=False,
                progress_callback=_conversion_progress(
                    progress,
                    "validation",
                    len(validation_ids),
                    (len(validation_ids) + SCENES_PER_SHARD - 1) // SCENES_PER_SHARD,
                    commit_volume,
                ),
            )
            progress.phase("finalize_validation", scene_count=len(validation_ids))
            validation_manifest = finalize_shards(
                validation_staging, VALIDATION_SCENES, scenes_per_shard=SCENES_PER_SHARD
            )
        if not output_root.is_dir():
            if not staging_root.is_dir():
                raise RuntimeError(f"train staging output is missing: {staging_root}")
            staging_root.replace(output_root)
        if not validation_root.is_dir():
            if not validation_staging.is_dir():
                raise RuntimeError(f"validation staging output is missing: {validation_staging}")
            validation_staging.replace(validation_root)
        data_volume.commit()
        progress.emit("published", train_scene_count=len(train_scene_ids), validation_scene_count=len(validation_ids), train_shard_count=shard_offset, validation_shard_count=len(validation_result["shards"]))
        result = {
            "status": "complete",
            "train_scene_count": len(train_scene_ids),
            "validation_scene_count": len(validation_ids),
            "train_shard_count": shard_offset,
            "validation_shard_count": len(validation_result["shards"]),
            "train_output": str(output_root),
            "validation_output": str(validation_root),
            "archives": archive_results,
            "elapsed_seconds": time.perf_counter() - started,
        }
        run.summary.update(
            {
                "train_scene_count": len(train_scene_ids),
                "validation_scene_count": len(validation_ids),
                "train_shard_count": shard_offset,
                "validation_shard_count": len(validation_result["shards"]),
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
        progress.emit("complete", elapsed_seconds=result["elapsed_seconds"])
        return result
    except Exception as error:
        progress.emit(
            "failed",
            error_type=type(error).__name__,
            error=str(error),
            output_root=str(output_root),
            staging_root=str(staging_root),
        )
        run.summary["failure"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        progress.close()
        run.finish()


@app.local_entrypoint()
def main() -> None:
    result = convert_full.remote()
    print(json.dumps(result, indent=2, sort_keys=True))
