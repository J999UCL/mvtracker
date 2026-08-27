"""Download TAPVid-MV and profile five persistent VGGT-Omega cache conversions."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import urllib.request

import modal


APP_NAME = "jeet-mvtracker-tapvidmv-vggt-smoke"
TAPVIDMV_REVISION = "9c7a2a7fd74cbfe8dd0d0ade8fca6f07e86fd091"
VGGT_OMEGA_REVISION = "39a0cb8af88554f15ddcb5354cd52bde588fa014"
VGGT_OMEGA_CHECKPOINT_REVISION = "05654241adc2f218dfb089c373a011f8a7040576"

WORKSPACE = Path("/opt/tapvidmv-workspace")
TAPVIDMV_ROOT = WORKSPACE / "tapvidmv"
EVAL_MOUNT = Path("/mnt/tapvidmv-eval")
DATASET_ROOT = EVAL_MOUNT / "tapvidmv_dataset"
RUNS_ROOT = EVAL_MOUNT / "runs"
BUCKET_ROOT = "https://storage.googleapis.com/dm-tapnet/mv-tap"

PROFILE_SOURCE = "harmony4d"
PROFILE_SEQUENCES = (
    "001_ballroom2_human_cleaned",
    "001_ballroom_human_cleaned",
    "001_hugging_human_cleaned",
    "001_sword3_human_cleaned",
    "001_sword_human_cleaned",
)

SOURCES = (
    ("droid", f"{BUCKET_ROOT}/droid", "tapvidmv/"),
    ("harmony4d", f"{BUCKET_ROOT}/harmony4d", ""),
    ("egoexo4d", f"{BUCKET_ROOT}/egoexo4d", ""),
    ("pace", f"{BUCKET_ROOT}/pace", ""),
    ("waymo", f"{BUCKET_ROOT}/waymo", ""),
    ("diegesis", BUCKET_ROOT, "diegesis/scenes/"),
)

HARMONY_ROOT_FILES = (
    "tracks_xyz.npy",
    "queries_xytv.npy",
    "multiview_confirmed.npy",
    "static_track_index.npy",
    "visualize_tracks.mp4",
)
HARMONY_VIEW_FILES = (
    "depth.npy.gz",
    "extrinsics_w2c.npy",
    "images_jpeg_bytes.npy",
    "intrinsics.npy",
    "visibility.npy",
)

TAGS = {
    "owner": "jeet",
    "project": "mvtracker",
    "purpose": "profiling",
    "experiment": "tapvidmv-vggt-h200-five-sequence-profile",
}

github_secret = modal.Secret.from_name(
    "jeet-mvtracker-github", required_keys=["GITHUB_TOKEN"]
)
hf_secret = modal.Secret.from_name(
    "jeet-mvtracker-huggingface", required_keys=["HF_TOKEN"]
)
wandb_secret = modal.Secret.from_name(
    "jeet-mvtracker-wandb", required_keys=["WANDB_API_KEY"]
)


def _source_commit() -> str:
    commit = os.environ.get("MVTRACKER_MODAL_COMMIT", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError("set MVTRACKER_MODAL_COMMIT to the pushed source commit")
    return commit


def _image() -> modal.Image:
    commit = _source_commit()
    return (
        modal.Image.debian_slim(python_version="3.12")
        .apt_install(
            "build-essential",
            "ca-certificates",
            "ffmpeg",
            "git",
            "libgl1",
            "libglib2.0-0",
        )
        .run_commands(
            "python -m pip install --upgrade pip wheel setuptools",
            (
                "python -m pip install --index-url https://download.pytorch.org/whl/cu129 "
                "torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0"
            ),
        )
        .run_commands(
            (
                "git clone https://github.com/J999UCL/mvtracker.git /opt/mvtracker && "
                f"git -C /opt/mvtracker checkout {commit} && "
                f'test "$(git -C /opt/mvtracker rev-parse HEAD)" = "{commit}"'
            ),
            (
                "git clone \"https://x-access-token:${GITHUB_TOKEN}@github.com/Fxiz0707/tapvidmv.git\" "
                f"{TAPVIDMV_ROOT} && git -C {TAPVIDMV_ROOT} checkout {TAPVIDMV_REVISION} && "
                f"git -C {TAPVIDMV_ROOT} remote set-url origin https://github.com/Fxiz0707/tapvidmv.git"
            ),
            secrets=[github_secret],
        )
        .run_commands(
            f"python -m pip install -r {TAPVIDMV_ROOT / 'requirements.txt'}",
            "python -m pip install wandb",
            f"git clone https://github.com/facebookresearch/vggt-omega.git {WORKSPACE / 'vggt-omega'}",
            f"git -C {WORKSPACE / 'vggt-omega'} checkout {VGGT_OMEGA_REVISION}",
            f"python /opt/mvtracker/tools/patch_tapvidmv_performance.py {TAPVIDMV_ROOT}",
        )
        .run_commands(
            f"mkdir -p {WORKSPACE / 'checkpoints'} && "
            "hf download facebook/VGGT-Omega vggt_omega_1b_512.pt "
            f"--revision {VGGT_OMEGA_CHECKPOINT_REVISION} --local-dir {WORKSPACE / 'checkpoints'}",
            secrets=[hf_secret],
        )
        .env(
            {
                "PYTHONPATH": str(WORKSPACE),
                "MVTRACKER_MODAL_COMMIT": commit,
                "PYTHONUNBUFFERED": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "TAPVIDMV_KEEP_VGGT_MODEL": "1",
                "OMP_NUM_THREADS": "2",
                "MKL_NUM_THREADS": "2",
                "OPENBLAS_NUM_THREADS": "2",
                "NUMEXPR_NUM_THREADS": "2",
            }
        )
    )


app = modal.App(APP_NAME, tags=TAGS)
image = _image()
eval_volume = modal.Volume.from_name(
    "jeet-mvtracker-tapvidmv-eval-v1", create_if_missing=True
)


def _log(event: str, **fields) -> None:
    print(
        json.dumps(
            {
                "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "event": event,
                **fields,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _download_file(url: str, target: Path) -> tuple[int, bool]:
    if target.is_file() and target.stat().st_size > 0:
        return target.stat().st_size, True
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "mvtracker-tapvidmv"})
            with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as output:
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
            partial.replace(target)
            return target.stat().st_size, False
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 3:
                raise
    raise AssertionError


def _manifest_jobs(source: str, base_url: str, strip_prefix: str):
    with urllib.request.urlopen(
        f"{BUCKET_ROOT}/{source}_file_list.txt", timeout=300
    ) as response:
        entries = [line for line in response.read().decode().splitlines() if line]
    for entry in entries:
        relative = entry[len(strip_prefix) :] if entry.startswith(strip_prefix) else entry
        yield f"{base_url}/{entry}", DATASET_ROOT / source / relative


def _harmony_jobs():
    sys.path.insert(0, str(WORKSPACE))
    from tapvidmv.splits import tapvidmv_splits

    names = [
        name
        for name in tapvidmv_splits.get_eval_sequences("harmony4d")
        if name != "010_ballroom2"
    ]
    for sequence in names:
        for filename in HARMONY_ROOT_FILES:
            relative = f"{sequence}/{filename}"
            yield f"{BUCKET_ROOT}/harmony4d/{relative}", DATASET_ROOT / "harmony4d" / relative
        for view in range(4):
            for filename in HARMONY_VIEW_FILES:
                relative = f"{sequence}/{view}/{filename}"
                yield f"{BUCKET_ROOT}/harmony4d/{relative}", DATASET_ROOT / "harmony4d" / relative


@app.function(
    image=image,
    cpu=16,
    memory=32768,
    timeout=24 * 60 * 60,
    volumes={EVAL_MOUNT: eval_volume},
    secrets=[wandb_secret],
)
def download_dataset(run_name: str) -> dict:
    import wandb

    run = wandb.init(
        project="mvtracker-modal-profiling",
        name=f"{run_name}-download",
        group=run_name,
        job_type="tapvidmv-download",
        config={**TAGS, "sources": [source[0] for source in SOURCES]},
    )
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    result = {}
    for source, base_url, strip_prefix in SOURCES:
        jobs = list(_harmony_jobs()) if source == "harmony4d" else list(
            _manifest_jobs(source, base_url, strip_prefix)
        )
        started = time.perf_counter()
        downloaded = reused = bytes_total = 0
        _log("download_source_started", source=source, files=len(jobs))
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(_download_file, url, target) for url, target in jobs]
            for completed, future in enumerate(as_completed(futures), start=1):
                size, was_reused = future.result()
                bytes_total += size
                reused += int(was_reused)
                downloaded += int(not was_reused)
                if completed % 50 == 0 or completed == len(futures):
                    _log(
                        "download_source_progress",
                        source=source,
                        completed=completed,
                        total=len(futures),
                    )
        eval_volume.commit()
        seconds = time.perf_counter() - started
        result[source] = {
            "files": len(jobs),
            "downloaded": downloaded,
            "reused": reused,
            "bytes": bytes_total,
            "seconds": seconds,
        }
        run.log(
            {
                f"download/{source}/files": len(jobs),
                f"download/{source}/bytes": bytes_total,
                f"download/{source}/seconds": seconds,
            }
        )
        _log("download_source_completed", source=source, **result[source])
    run.finish()
    return result


def _gpu_monitor(stop: threading.Event, samples: list[dict]) -> None:
    while not stop.wait(10):
        process = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if process.returncode:
            continue
        values = [float(value.strip()) for value in process.stdout.strip().split(",")]
        sample = {
            "utilization_percent": values[0],
            "memory_used_mib": values[1],
            "memory_total_mib": values[2],
            "power_watts": values[3],
        }
        samples.append(sample)
        _log("gpu_status", **sample)


@app.function(
    image=image,
    gpu="H200",
    cpu=16,
    memory=98304,
    timeout=24 * 60 * 60,
    max_containers=1,
    volumes={EVAL_MOUNT: eval_volume},
    secrets=[wandb_secret],
)
def profile_five(run_name: str) -> dict:
    import wandb

    function_started = time.perf_counter()
    run = wandb.init(
        project="mvtracker-modal-profiling",
        name=f"{run_name}-h200",
        group=run_name,
        job_type="tapvidmv-vggt-omega-five-sequence-profile",
        config={
            **TAGS,
            "gpu": "H200",
            "source": PROFILE_SOURCE,
            "sequences": PROFILE_SEQUENCES,
            "image_backend": "torchvision_nvjpeg_cuda_endpoint_cpu",
            "load_optional_depth": False,
            "keep_vggt_model": True,
            "prefetch_sequences": 1,
            "tapvidmv_revision": TAPVIDMV_REVISION,
            "vggt_omega_revision": VGGT_OMEGA_REVISION,
        },
    )
    prediction_root = RUNS_ROOT / run_name / "reconstruction__copycat__vggt_omega"
    command = [
        sys.executable,
        "-u",
        "-m",
        "tapvidmv.run_predictor",
        "--predictor",
        "reconstruction__copycat__vggt_omega",
        "--tapvidmv_dir",
        str(DATASET_ROOT),
        "--tapvidmv_predictions",
        str(prediction_root),
        "--data_sources_to_predict",
        PROFILE_SOURCE,
        "--sequences",
        ",".join(PROFILE_SEQUENCES),
        "--resolution",
        "512",
        "--skip_optional_depth",
        "--image_backend",
        "cuda",
        "--overwrite_predictions",
    ]
    profile_events: list[dict] = []
    samples: list[dict] = []
    stop = threading.Event()
    monitor = threading.Thread(target=_gpu_monitor, args=(stop, samples), daemon=True)
    monitor.start()
    _log("inference_started", command=command)
    inference_started = time.perf_counter()
    try:
        process = subprocess.Popen(
            command,
            cwd=WORKSPACE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            if text := line.rstrip():
                _log("inference_output", message=text)
                try:
                    event = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and "event" in event:
                    profile_events.append(event)
        returncode = process.wait()
    finally:
        stop.set()
        monitor.join(timeout=5)
    inference_seconds = time.perf_counter() - inference_started
    if returncode:
        run.finish(exit_code=returncode)
        raise subprocess.CalledProcessError(returncode, command)
    eval_volume.commit()
    function_seconds = time.perf_counter() - function_started
    conversion_events = [
        event for event in profile_events if event["event"] == "image_conversion_completed"
    ]
    load_events = [
        event for event in profile_events if event["event"] == "sequence_load_completed"
    ]
    prediction_events = [
        event for event in profile_events if event["event"] == "sequence_prediction_completed"
    ]
    result = {
        "inference_seconds": inference_seconds,
        "h200_seconds": function_seconds,
        "h200_hours": function_seconds / 3600.0,
        "sequence_count": len(PROFILE_SEQUENCES),
        "conversion_seconds_sum": sum(event["seconds"] for event in conversion_events),
        "load_seconds_sum": sum(event["seconds"] for event in load_events),
        "inference_seconds_sum": sum(event["inference_seconds"] for event in prediction_events),
        "load_wait_seconds_sum": sum(event["load_wait_seconds"] for event in prediction_events),
        "gpu_samples": len(samples),
        "mean_gpu_utilization_percent": (
            sum(sample["utilization_percent"] for sample in samples) / len(samples)
            if samples
            else 0.0
        ),
        "peak_memory_used_mib": max(
            (sample["memory_used_mib"] for sample in samples), default=0.0
        ),
        "mean_power_watts": (
            sum(sample["power_watts"] for sample in samples) / len(samples)
            if samples
            else 0.0
        ),
        "profile_events": profile_events,
        "cache_paths": [
            str(RUNS_ROOT / run_name / "_reconstruction_cache/vggt_omega" / PROFILE_SOURCE / f"{sequence}.npz")
            for sequence in PROFILE_SEQUENCES
        ],
    }
    report_path = RUNS_ROOT / run_name / "h200-five-sequence-profile.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    eval_volume.commit()
    scalar_result = {key: value for key, value in result.items() if isinstance(value, int | float)}
    run.log(scalar_result)
    for sample in samples:
        run.log({f"gpu/{key}": value for key, value in sample.items()})
    for event in load_events:
        run.summary[f"sequences/{event['sequence']}/load_seconds"] = event["seconds"]
    for event in conversion_events:
        run.summary[
            f"sequences/{event['sequence']}/views/{event['view']}/conversion_seconds"
        ] = event["seconds"]
    for event in prediction_events:
        run.summary[f"sequences/{event['sequence']}/inference_seconds"] = event["inference_seconds"]
        run.summary[f"sequences/{event['sequence']}/load_wait_seconds"] = event["load_wait_seconds"]
    run.summary.update(scalar_result)
    run.finish()
    _log("inference_completed", report_path=str(report_path), **result)
    return result


@app.local_entrypoint()
def main(stage: str = "full", run_name: str = "") -> None:
    if not run_name:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_name = f"tapvidmv-vggt-h200-five-{timestamp}"
    app.set_tags({**TAGS, "run_name": run_name, "stage": stage})
    if stage in {"full", "download"}:
        print(download_dataset.remote(run_name))
    if stage in {"full", "profile"}:
        print(profile_five.remote(run_name))
    if stage not in {"full", "download", "profile"}:
        raise ValueError(f"unsupported stage: {stage}")
