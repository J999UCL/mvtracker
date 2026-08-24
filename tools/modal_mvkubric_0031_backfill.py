"""Download and package only the missing MV-Kubric 0031--1000 source archive."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import modal

from modal_training_profile import BASE_TAGS, DATA_ROOT, _source_commit, data_volume, wandb_secret


APP_NAME = "jeet-mvtracker-mvkubric-0031-backfill"
ARCHIVE_NAME = "kubric-multiview--train.full.0031-1000.tar.gz"
REPO_ID = "ethz-vlg/mv3dpt-datasets"
REVISION = "cccb9128fb95d302c662151e65a09377175c2a3a"
ARCHIVE_PATH = DATA_ROOT / "archives/mvkubric/2000-scenes-v1" / ARCHIVE_NAME
TRAIN_ROOT = DATA_ROOT / "datasets/kubric-multiview-webdataset/train"
REPORT_ROOT = DATA_ROOT / "datasets/materialization-reports/mvkubric"
DEFAULT_MANIFEST = "manifests/mvkubric-0031-1000-backfill.json"
TAGS = {**BASE_TAGS, "experiment": "mvkubric-0031-backfill", "gpu": "cpu"}


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
        .pip_install(
            "hf-xet==1.6.0", "huggingface-hub==1.28.0", "rapidgzip==0.16.0",
            "nvidia-dali-cuda120==1.53.0", "numpy==2.2.4", "wandb==0.19.9",
        )
        .run_commands(clone)
        .env({"MVTRACKER_MODAL_COMMIT": commit, "PYTHONPATH": "/opt/mvtracker:/opt/mvtracker/tools"})
    )


app = modal.App(APP_NAME, tags=TAGS)
image = _image()


def _scenes(payload: dict[str, object]) -> list[str]:
    seen: set[str] = set()
    result = []
    for value in payload.get("train_scene_ids", []):
        scene = str(value)
        if scene.isdigit() and scene not in seen:
            seen.add(scene)
            result.append(scene)
    for item in payload.get("train_ranges", []):
        if not isinstance(item, dict):
            continue
        try:
            start = int(item["start"])
            end = int(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        excluded = {str(value) for value in item.get("exclude", [])}
        for value in range(start, end + 1):
            scene = str(value)
            if scene not in excluded and scene not in seen:
                seen.add(scene)
                result.append(scene)
    return sorted(result, key=int)


class _Progress:
    def __init__(self, run, report: Path, name: str):
        self.run = run
        self.report = report
        self.name = name
        self.started = time.perf_counter()
        self.phase = "startup"
        self.fields: dict[str, object] = {}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    def emit(self, event: str, **fields: object) -> None:
        with self.lock:
            payload = {
                "event": event,
                "manifest": self.name,
                "phase": self.phase,
                "elapsed_seconds": round(time.perf_counter() - self.started, 2),
                **self.fields,
                **fields,
            }
            print(json.dumps(payload, sort_keys=True), flush=True)
            self.report.parent.mkdir(parents=True, exist_ok=True)
            with self.report.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
            self.run.log(
                {
                    f"progress/{key}": value
                    for key, value in payload.items()
                    if isinstance(value, (int, float))
                },
                commit=True,
            )

    def set_phase(self, phase: str, **fields: object) -> None:
        self.phase = phase
        self.fields = fields
        self.emit("phase_start")

    def _heartbeat(self) -> None:
        while not self.stop.wait(15):
            usage = shutil.disk_usage("/tmp")
            self.emit(
                "heartbeat",
                local_disk_free_gib=round(usage.free / (1 << 30), 2),
                local_disk_used_gib=round(usage.used / (1 << 30), 2),
            )

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)
        self.emit("job_complete")


def _copy(source: Path, destination: Path, progress: _Progress, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    copied = 0
    started = time.perf_counter()
    next_report = 1 << 30
    with source.open("rb") as input_stream, destination.open("wb") as output_stream:
        while block := input_stream.read(64 << 20):
            output_stream.write(block)
            copied += len(block)
            elapsed = max(time.perf_counter() - started, 1e-6)
            if copied >= next_report:
                progress.emit(
                    "copy_progress",
                    copy=label,
                    bytes=copied,
                    gib_per_second=round(copied / (1 << 30) / elapsed, 3),
                )
                next_report += 1 << 30
    progress.emit("copy_complete", copy=label, bytes=copied)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("jeet-mvtracker-huggingface", required_keys=["HF_TOKEN"]), wandb_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=16,
    memory=32 * 1024,
    ephemeral_disk=1024 * 1024,
    timeout=24 * 60 * 60,
    max_containers=1,
    include_source=False,
)
def download_remote(manifest: dict[str, object]) -> dict[str, object]:
    import wandb
    from huggingface_hub import hf_hub_download

    name = str(manifest.get("name", "mvkubric-0031-backfill"))
    run = wandb.init(project="mvtracker-modal-profiling", job_type="mvkubric-download", name=name, tags=["modal", "cpu", "mv-kubric", "0031-1000"], config={**TAGS, "source_revision": REVISION})
    progress = _Progress(run, REPORT_ROOT / name / "events.ndjson", name)
    try:
        requested = _scenes(manifest)
        if ARCHIVE_PATH.is_file():
            progress.emit("archive_reused", archive=str(ARCHIVE_PATH), requested_scenes=len(requested))
            result = {"status": "reused", "archive": str(ARCHIVE_PATH), "requested_scenes": requested}
        else:
            work = Path("/tmp/mvkubric-0031-download")
            work.mkdir(parents=True, exist_ok=True)
            progress.set_phase("download", archive=ARCHIVE_NAME, requested_scenes=len(requested))
            source = Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", revision=REVISION, filename=ARCHIVE_NAME, token=os.environ["HF_TOKEN"], local_dir=str(work), cache_dir=str(work / "cache")))
            partial = ARCHIVE_PATH.with_suffix(ARCHIVE_PATH.suffix + ".partial")
            progress.set_phase("volume_copy", archive=ARCHIVE_NAME)
            _copy(source, partial, progress, "archive_to_volume")
            partial.replace(ARCHIVE_PATH)
            progress.emit("archive_downloaded", archive=str(ARCHIVE_PATH), bytes=ARCHIVE_PATH.stat().st_size)
            result = {"status": "downloaded", "archive": str(ARCHIVE_PATH), "requested_scenes": requested}
        progress.set_phase("volume_commit")
        data_volume.commit()
        progress.emit("volume_commit_complete")
        return result
    except Exception as error:
        progress.emit("failed", error_type=type(error).__name__, error=str(error))
        return {"status": "partial", "error": f"{type(error).__name__}: {error}"}
    finally:
        progress.close()
        run.finish()


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=16,
    memory=32 * 1024,
    ephemeral_disk=1024 * 1024,
    timeout=24 * 60 * 60,
    max_containers=1,
    include_source=False,
)
def preprocess_remote(manifest: dict[str, object]) -> dict[str, object]:
    import wandb
    from mvtracker.preprocessing.mvkubric_webdataset import convert_shards, finalize_shards
    from tools.modal_mvkubric_tar_index import _index_one, _wds2idx_command

    name = str(manifest.get("name", "mvkubric-0031-backfill"))
    run = wandb.init(project="mvtracker-modal-profiling", job_type="mvkubric-preprocess", name=name, tags=["modal", "cpu", "mv-kubric", "0031-1000", "webdataset"], config={**TAGS})
    progress = _Progress(run, REPORT_ROOT / name / "events.ndjson", name)
    try:
        if not ARCHIVE_PATH.is_file():
            progress.emit("archive_missing", archive=str(ARCHIVE_PATH))
            return {"status": "partial", "error": "missing source archive"}
        requested = _scenes(manifest)
        existing = json.loads((TRAIN_ROOT / "manifest.json").read_text(encoding="utf-8"))
        existing_ids = {str(scene) for scene in existing["scene_ids"]}
        pending = [scene for scene in requested if scene not in existing_ids]
        if not pending:
            progress.emit("all_scenes_reused", scene_count=len(requested))
            return {"status": "reused", "scene_count": len(requested)}
        work = Path("/tmp/mvkubric-0031-preprocess")
        local_archive = work / ARCHIVE_NAME
        extracted = work / "native"
        progress.set_phase("local_copy", pending_scenes=len(pending))
        _copy(ARCHIVE_PATH, local_archive, progress, "volume_to_local")
        extracted.mkdir(parents=True, exist_ok=True)
        progress.set_phase("extract", pending_scenes=len(pending))
        decompress = subprocess.Popen(["rapidgzip", "-d", "-c", "-P", "16", str(local_archive)], stdout=subprocess.PIPE)
        assert decompress.stdout is not None
        extract = subprocess.run(["tar", "--extract", "--strip-components=2", "--file=-", "--directory", str(extracted)], stdin=decompress.stdout, check=False)
        decompress.stdout.close()
        if decompress.wait() or extract.returncode:
            progress.emit("extract_failed", gzip_returncode=decompress.returncode, tar_returncode=extract.returncode)
            return {"status": "partial", "error": "archive extraction failed"}
        available = [scene for scene in pending if (extracted / scene).is_dir()]
        missing = [scene for scene in pending if scene not in set(available)]
        for scene in missing:
            progress.emit("scene_missing", scene=scene)
        if not available:
            return {"status": "partial", "missing_scenes": missing}
        offset = len(existing["shards"])
        progress.set_phase("package", scene_count=len(available), shard_offset=offset)
        converted = convert_shards(extracted, TRAIN_ROOT, available, shard_offset=offset, shard_workers=8, read_workers=2, finalize=False)
        command = _wds2idx_command()
        for completed, shard in enumerate(converted["shards"], start=1):
            tar = TRAIN_ROOT / str(shard["tar"])
            progress.set_phase("index", completed=completed, total=len(converted["shards"]), shard=str(shard["name"]))
            _index_one(tar, force=False, command=command)
        published = finalize_shards(TRAIN_ROOT, [*existing["scene_ids"], *converted["scene_ids"]])
        progress.set_phase("volume_commit")
        data_volume.commit()
        progress.emit("volume_commit_complete", scene_count=len(converted["scene_ids"]), shard_count=len(converted["shards"]))
        return {"status": "complete" if not missing else "partial", "added_scenes": converted["scene_ids"], "missing_scenes": missing, "total_scenes": len(published["scene_ids"])}
    except Exception as error:
        progress.emit("failed", error_type=type(error).__name__, error=str(error))
        return {"status": "partial", "error": f"{type(error).__name__}: {error}"}
    finally:
        progress.close()
        run.finish()


def _manifest(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@app.local_entrypoint(name="download")
def download(manifest: str = DEFAULT_MANIFEST) -> None:
    print(json.dumps(download_remote.remote(_manifest(manifest)), indent=2, sort_keys=True))


@app.local_entrypoint(name="preprocess")
def preprocess(manifest: str = DEFAULT_MANIFEST) -> None:
    print(json.dumps(preprocess_remote.remote(_manifest(manifest)), indent=2, sort_keys=True))
