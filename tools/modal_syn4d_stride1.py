"""Manifest-selected Syn4D stride-1 download and CPU cache conversion."""

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
from mvtracker.profiling.modal_syn4d_split import ARCHIVE_BYTES, LEGACY_ARCHIVES, SYN4D_REPO_ID, SYN4D_REVISION, SYN4D_SUBSET


APP_NAME = "jeet-mvtracker-syn4d-stride1"
ARCHIVE_ROOT = DATA_ROOT / "datasets/syn4d/full-stride1/source"
MAPPING_PATH = DATA_ROOT / "datasets/syn4d/sequence_to_asset_mapping_stride1.csv"
METADATA_ROOT = DATA_ROOT / "datasets/syn4d/v1-stride1-12train-4validation/metadata"
OUTPUT_ROOTS = {
    "train": DATA_ROOT / "datasets/syn4d-mvtracker/train",
    "validation": DATA_ROOT / "datasets/syn4d-mvtracker/validation",
}
REPORT_ROOT = DATA_ROOT / "datasets/materialization-reports/syn4d"
DEFAULT_MANIFEST = "manifests/syn4d-stride1-backfill.json"
TAGS = {**BASE_TAGS, "experiment": "syn4d-stride1", "gpu": "cpu"}


def _clone(image: modal.Image) -> modal.Image:
    commit = _source_commit()
    clone = (
        "git init /opt/mvtracker && "
        "git -C /opt/mvtracker remote add origin https://github.com/J999UCL/mvtracker.git && "
        f"git -C /opt/mvtracker fetch --depth=1 origin {commit} && "
        "git -C /opt/mvtracker checkout --detach FETCH_HEAD && "
        f'test "$(git -C /opt/mvtracker rev-parse HEAD)" = "{commit}"'
    )
    return image.run_commands(clone).env(
        {"MVTRACKER_MODAL_COMMIT": commit, "PYTHONPATH": "/opt/mvtracker:/opt/mvtracker/tools"}
    )


def _image() -> modal.Image:
    visualizer = f"https://huggingface.co/datasets/Syn4D/Syn4D/resolve/{SYN4D_REVISION}/code/visualizer/"
    return _clone(
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("ca-certificates", "curl", "ffmpeg", "git", "libgl1", "libglib2.0-0", "libopenexr-dev", "openexr", "zstd")
        .pip_install(
            "torch==2.7.1", "torchvision==0.22.1",
            index_url="https://download.pytorch.org/whl/cpu",
        )
        .pip_install(
            "hf-xet==1.6.0", "huggingface-hub==1.28.0", "OpenEXR==3.3.5", "safetensors==0.5.3",
            "opencv-python-headless==4.11.0.86", "numpy==2.2.4", "scipy==1.15.2",
            "pandas==2.2.3", "Pillow==11.1.0", "kornia==0.7.3",
            "mediapy==1.2.0", "pypng==0.20220715.0", "rerun-sdk==0.21.0",
            "wandb==0.19.9",
        )
        .run_commands(
            "mkdir -p /opt/syn4d-visualizer",
            *(
                f"curl -fsSL {visualizer}{name} -o /opt/syn4d-visualizer/{name}"
                for name in ("syn4d_track.py", "base_dataset.py", "utils.py")
            ),
        )
    )


app = modal.App(APP_NAME, tags=TAGS)
image = _image()


def _rows(manifest: dict[str, object]) -> list[dict[str, str]]:
    result = []
    seen: set[tuple[str, str, str]] = set()
    for item in manifest.get("sequences", []):
        if not isinstance(item, dict):
            continue
        environment = str(item.get("environment", ""))
        sequence = str(item.get("sequence", ""))
        split = str(item.get("split", "train"))
        key = environment, sequence, split
        if not environment or not sequence or split not in OUTPUT_ROOTS or key in seen:
            continue
        seen.add(key)
        result.append({"environment": environment, "sequence": sequence, "split": split})
    for item in manifest.get("sequence_ranges", []):
        if not isinstance(item, dict):
            continue
        environment = str(item.get("environment", ""))
        split = str(item.get("split", "train"))
        try:
            start = int(item["start"])
            end = int(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not environment or split not in OUTPUT_ROOTS:
            continue
        for index in range(start, end + 1):
            sequence = f"seq_{index:06d}"
            key = environment, sequence, split
            if key not in seen:
                seen.add(key)
                result.append({"environment": environment, "sequence": sequence, "split": split})
    return result


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
        self.cpu_count = os.cpu_count() or 1
        self.last_cpu_usage_usec = self._cpu_usage_usec()
        self.last_cpu_sample = time.perf_counter()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread.start()

    @staticmethod
    def _cpu_usage_usec() -> int:
        for line in Path("/sys/fs/cgroup/cpu.stat").read_text(encoding="utf-8").splitlines():
            key, value = line.split()
            if key == "usage_usec":
                return int(value)
        raise RuntimeError("cgroup cpu.stat did not report usage_usec")

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
            self.run.log({f"progress/{key}": value for key, value in payload.items() if isinstance(value, (int, float))}, commit=True)

    def set_phase(self, phase: str, **fields: object) -> None:
        self.phase = phase
        self.fields = fields
        self.emit("phase_start")

    def _heartbeat(self) -> None:
        while not self.stop.wait(15):
            usage = shutil.disk_usage("/tmp")
            now = time.perf_counter()
            cpu_usage_usec = self._cpu_usage_usec()
            interval = now - self.last_cpu_sample
            cpu_percent = 100 * (cpu_usage_usec - self.last_cpu_usage_usec) / (1_000_000 * interval * self.cpu_count)
            self.last_cpu_usage_usec = cpu_usage_usec
            self.last_cpu_sample = now
            self.emit(
                "heartbeat",
                allocated_cpus=self.cpu_count,
                cpu_percent=round(cpu_percent, 1),
                local_disk_free_gib=round(usage.free / (1 << 30), 2),
                local_disk_used_gib=round(usage.used / (1 << 30), 2),
            )

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)
        self.emit("job_complete")


def _archive(environment: str) -> Path | None:
    legacy = LEGACY_ARCHIVES.get(environment)
    if legacy is not None and (DATA_ROOT / legacy).is_file():
        return DATA_ROOT / legacy
    cached = ARCHIVE_ROOT / f"{environment}.tar.zst"
    return cached if cached.is_file() else None


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
                progress.emit("copy_progress", copy=label, bytes=copied, gib_per_second=round(copied / (1 << 30) / elapsed, 3))
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

    name = str(manifest.get("name", "syn4d-stride1"))
    run = wandb.init(project="mvtracker-modal-profiling", job_type="syn4d-download", name=name, tags=["modal", "cpu", "syn4d", "stride1"], config={**TAGS, "source_revision": SYN4D_REVISION})
    progress = _Progress(run, REPORT_ROOT / name / "events.ndjson", name)
    results = []
    try:
        rows = _rows(manifest)
        environments = sorted({row["environment"] for row in rows})
        if not MAPPING_PATH.is_file():
            progress.set_phase("mapping_download")
            source = Path(hf_hub_download(repo_id=SYN4D_REPO_ID, repo_type="dataset", revision=SYN4D_REVISION, filename=f"{SYN4D_SUBSET}/sequence_to_asset_mapping.csv", token=os.environ["HF_TOKEN"], local_dir="/tmp/syn4d-mapping"))
            _copy(source, MAPPING_PATH, progress, "mapping_to_volume")
        else:
            progress.emit("mapping_reused", path=str(MAPPING_PATH))
        for environment in environments:
            existing = _archive(environment)
            if existing is not None:
                progress.emit("archive_reused", environment=environment, path=str(existing))
                results.append({"environment": environment, "status": "reused", "path": str(existing)})
                continue
            try:
                progress.set_phase("archive_download", environment=environment, expected_bytes=ARCHIVE_BYTES.get(environment))
                work = Path("/tmp/syn4d-download") / environment
                work.mkdir(parents=True, exist_ok=True)
                source = Path(hf_hub_download(repo_id=SYN4D_REPO_ID, repo_type="dataset", revision=SYN4D_REVISION, filename=f"{SYN4D_SUBSET}/{environment}.tar.zst", token=os.environ["HF_TOKEN"], local_dir=str(work), cache_dir=str(work / "cache")))
                target = ARCHIVE_ROOT / f"{environment}.tar.zst"
                partial = target.with_suffix(target.suffix + ".partial")
                progress.set_phase("volume_copy", environment=environment)
                _copy(source, partial, progress, "archive_to_volume")
                partial.replace(target)
                progress.emit("archive_downloaded", environment=environment, path=str(target), bytes=target.stat().st_size)
                results.append({"environment": environment, "status": "downloaded", "path": str(target)})
            except Exception as error:
                progress.emit("archive_failed", environment=environment, error_type=type(error).__name__, error=str(error))
                results.append({"environment": environment, "status": "failed", "error": f"{type(error).__name__}: {error}"})
        progress.set_phase("volume_commit")
        data_volume.commit()
        progress.emit("volume_commit_complete")
        return {"status": "complete" if all(item["status"] != "failed" for item in results) else "partial", "archives": results}
    finally:
        progress.close()
        run.finish()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("jeet-mvtracker-huggingface", required_keys=["HF_TOKEN"]), wandb_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=16,
    memory=32 * 1024,
    ephemeral_disk=1024 * 1024,
    timeout=24 * 60 * 60,
    max_containers=4,
    include_source=False,
)
def download_environment_remote(environment: str, manifest_name: str) -> dict[str, object]:
    import wandb
    from huggingface_hub import hf_hub_download

    run = wandb.init(
        project="mvtracker-modal-profiling",
        job_type="syn4d-download",
        name=f"{manifest_name}-{environment}",
        tags=["modal", "cpu", "syn4d", "stride1", "archive"],
        config={**TAGS, "source_revision": SYN4D_REVISION, "environment": environment},
    )
    progress = _Progress(run, REPORT_ROOT / manifest_name / f"{environment}.ndjson", manifest_name)
    try:
        existing = _archive(environment)
        if existing is not None:
            progress.emit("archive_reused", environment=environment, path=str(existing))
            return {"environment": environment, "status": "reused", "path": str(existing)}
        progress.set_phase("archive_download", environment=environment, expected_bytes=ARCHIVE_BYTES.get(environment))
        work = Path("/tmp/syn4d-download") / environment
        work.mkdir(parents=True, exist_ok=True)
        source = Path(hf_hub_download(
            repo_id=SYN4D_REPO_ID,
            repo_type="dataset",
            revision=SYN4D_REVISION,
            filename=f"{SYN4D_SUBSET}/{environment}.tar.zst",
            token=os.environ["HF_TOKEN"],
            local_dir=str(work),
            cache_dir=str(work / "cache"),
        ))
        target = ARCHIVE_ROOT / f"{environment}.tar.zst"
        partial = target.with_suffix(target.suffix + ".partial")
        progress.set_phase("volume_copy", environment=environment)
        _copy(source, partial, progress, "archive_to_volume")
        partial.replace(target)
        progress.emit("archive_downloaded", environment=environment, path=str(target), bytes=target.stat().st_size)
        progress.set_phase("volume_commit", environment=environment)
        data_volume.commit()
        progress.emit("volume_commit_complete", environment=environment)
        return {"environment": environment, "status": "downloaded", "path": str(target)}
    except Exception as error:
        progress.emit("archive_failed", environment=environment, error_type=type(error).__name__, error=str(error))
        return {"environment": environment, "status": "failed", "error": f"{type(error).__name__}: {error}"}
    finally:
        progress.close()
        run.finish()


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("jeet-mvtracker-huggingface", required_keys=["HF_TOKEN"]), wandb_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=4,
    memory=8 * 1024,
    ephemeral_disk=512 * 1024,
    timeout=60 * 60,
    max_containers=1,
    include_source=False,
)
def prepare_mapping_remote(manifest_name: str) -> dict[str, object]:
    import wandb
    from huggingface_hub import hf_hub_download

    run = wandb.init(project="mvtracker-modal-profiling", job_type="syn4d-download", name=f"{manifest_name}-mapping", tags=["modal", "cpu", "syn4d", "stride1", "mapping"], config={**TAGS, "source_revision": SYN4D_REVISION})
    progress = _Progress(run, REPORT_ROOT / manifest_name / "mapping.ndjson", manifest_name)
    try:
        if MAPPING_PATH.is_file():
            progress.emit("mapping_reused", path=str(MAPPING_PATH))
            return {"status": "reused", "path": str(MAPPING_PATH)}
        progress.set_phase("mapping_download")
        source = Path(hf_hub_download(repo_id=SYN4D_REPO_ID, repo_type="dataset", revision=SYN4D_REVISION, filename=f"{SYN4D_SUBSET}/sequence_to_asset_mapping.csv", token=os.environ["HF_TOKEN"], local_dir="/tmp/syn4d-mapping"))
        _copy(source, MAPPING_PATH, progress, "mapping_to_volume")
        progress.set_phase("volume_commit")
        data_volume.commit()
        progress.emit("volume_commit_complete")
        return {"status": "downloaded", "path": str(MAPPING_PATH)}
    finally:
        progress.close()
        run.finish()


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=16,
    memory=48 * 1024,
    ephemeral_disk=1024 * 1024,
    timeout=24 * 60 * 60,
    max_containers=4,
    include_source=False,
)
def preprocess_environment_remote(
    environment: str,
    rows: list[dict[str, str]],
    manifest_name: str,
) -> dict[str, object]:
    import wandb
    import torch
    from mvtracker.preprocessing.syn4d import convert_syn4d_sequence

    torch.set_num_threads(os.cpu_count() or 1)
    torch.set_num_interop_threads(1)
    run = wandb.init(
        project="mvtracker-modal-profiling",
        job_type="syn4d-preprocess",
        name=f"{manifest_name}-{environment}",
        tags=["modal", "cpu", "syn4d", "stride1", "environment"],
        config={**TAGS, "environment": environment, "sequences": len(rows), "allocated_cpus": os.cpu_count() or 1},
    )
    progress = _Progress(run, REPORT_ROOT / manifest_name / f"{environment}.ndjson", manifest_name)
    results = []
    try:
        archive = _archive(environment)
        if archive is None:
            progress.emit("archive_missing", environment=environment)
            return {"environment": environment, "status": "blocked", "sequences": [{**row, "status": "blocked", "error": "missing archive"} for row in rows]}
        work = Path("/tmp/syn4d-preprocess") / environment
        extracted = work / "extracted"
        progress.set_phase("extract", environment=environment, sequences=len(rows))
        extracted.mkdir(parents=True, exist_ok=True)
        extraction_started = time.perf_counter()
        extraction = subprocess.run(["tar", "-I", "zstd -T0", "-xf", str(archive), "-C", str(extracted)], check=False)
        progress.emit("extract_complete", environment=environment, seconds=round(time.perf_counter() - extraction_started, 2), returncode=extraction.returncode)
        if extraction.returncode:
            results.extend({**row, "status": "failed", "error": "extract failed"} for row in rows)
        else:
            for sequence_index, row in enumerate(rows, start=1):
                output = OUTPUT_ROOTS[row["split"]] / f"{environment}__{row['sequence']}"
                if (output / "manifest.json").is_file():
                    progress.emit("sequence_reused", **row, sequence_index=sequence_index, output=str(output))
                    results.append({**row, "status": "reused"})
                    continue
                try:
                    progress.set_phase("sequence_convert", **row, sequence_index=sequence_index, sequences=len(rows))
                    started = time.perf_counter()
                    result = convert_syn4d_sequence(extracted / environment, METADATA_ROOT, OUTPUT_ROOTS[row["split"]], official_visualizer_root=Path("/opt/syn4d-visualizer"), sequence=row["sequence"], device="cpu", progress=lambda event: progress.emit("converter", **{**row, **event}))
                    progress.emit("sequence_complete", **row, sequence_index=sequence_index, seconds=round(time.perf_counter() - started, 2), output=result["output_path"])
                    results.append({**row, "status": "complete"})
                    progress.set_phase("volume_commit", **row, sequence_index=sequence_index)
                    commit_started = time.perf_counter()
                    data_volume.commit()
                    progress.emit("volume_commit_complete", **row, sequence_index=sequence_index, seconds=round(time.perf_counter() - commit_started, 2))
                except Exception as error:
                    progress.emit("sequence_failed", **row, sequence_index=sequence_index, error_type=type(error).__name__, error=str(error))
                    results.append({**row, "status": "failed", "error": f"{type(error).__name__}: {error}"})
        return {"environment": environment, "status": "complete" if all(item["status"] != "failed" for item in results) else "partial", "sequences": results}
    finally:
        progress.close()
        run.finish()


def _manifest(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@app.local_entrypoint(name="download")
def download(manifest: str = DEFAULT_MANIFEST) -> None:
    payload = _manifest(manifest)
    name = str(payload.get("name", "syn4d-stride1"))
    rows = _rows(payload)
    mapping = prepare_mapping_remote.remote(name)
    environments = sorted({row["environment"] for row in rows})
    archives = list(download_environment_remote.map(environments, [name] * len(environments)))
    print(json.dumps({"mapping": mapping, "archives": archives}, indent=2, sort_keys=True))


@app.local_entrypoint(name="preprocess")
def preprocess(manifest: str = DEFAULT_MANIFEST) -> None:
    payload = _manifest(manifest)
    name = str(payload.get("name", "syn4d-stride1"))
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in _rows(payload):
        grouped.setdefault(row["environment"], []).append(row)
    environments = sorted(grouped)
    results = list(preprocess_environment_remote.map(environments, [grouped[environment] for environment in environments], [name] * len(environments)))
    print(json.dumps({"environments": results}, indent=2, sort_keys=True))
