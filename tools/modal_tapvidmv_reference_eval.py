"""Run the TAPVidMV reference-depth benchmark on one MVTracker checkpoint."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request

import modal


APP_NAME = "jeet-mvtracker-tapvidmv-reference-eval"
TAPVIDMV_REVISION = "0722eb85ad93aa4ef06df2e573273dd023dc673e"
CHECKPOINT_SHA256 = "3185c09b766dd32024e2d866e689fc34054b94f515df31651ddc6aef6a180929"
CHECKPOINT_RELATIVE_PATH = Path(
    "continual-training/gt-replay-centered-syn4d-v5-b1-ddp2-h200-20260822T205118Z/"
    "model_001500.pth"
)

WORKSPACE = Path("/opt/tapvidmv-workspace")
TAPVIDMV_ROOT = WORKSPACE / "tapvidmv"
TAPVIDMV_PERFORMANCE_PATCHER = Path("/tmp/patch_tapvidmv_performance.py")
CHECKPOINT_MOUNT = Path("/mnt/mvtracker-runs")
EVAL_MOUNT = Path("/mnt/tapvidmv-eval")
DATASET_ROOT = EVAL_MOUNT / "tapvidmv_dataset"
RUNS_ROOT = EVAL_MOUNT / "runs"

BUCKET_ROOT = "https://storage.googleapis.com/dm-tapnet/mv-tap"
SOURCES = (
    ("droid", f"{BUCKET_ROOT}/droid", "tapvidmv/", 50),
    ("hi4d", f"{BUCKET_ROOT}/hi4d", "hi4d/", 24),
    ("harmony4d", BUCKET_ROOT, "harmony4d/", 21),
    ("egoexo4d", f"{BUCKET_ROOT}/egoexo4d", "", 18),
    ("pace", f"{BUCKET_ROOT}/pace", "", 69),
    ("waymo", f"{BUCKET_ROOT}/waymo", "", 50),
    ("diegesis", BUCKET_ROOT, "diegesis/scenes/", 21),
)
SOURCE_NAMES = tuple(source[0] for source in SOURCES)
PREDICTION_LANES = 8
SOURCE_PREDICTION_LANES = {"hi4d": 7}
LOADER_WORKERS_PER_LANE = 1

TAGS = {
    "owner": "jeet",
    "project": "mvtracker",
    "purpose": "evaluation",
    "experiment": "tapvidmv-reference-depth-step1500",
}

github_secret = modal.Secret.from_name(
    "jeet-mvtracker-github",
    required_keys=["GITHUB_TOKEN"],
)

tapvidmv_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "build-essential",
        "ca-certificates",
        "curl",
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
        f"mkdir -p {WORKSPACE}",
    )
    .run_commands(
        (
            "git clone \"https://x-access-token:${GITHUB_TOKEN}@github.com/Fxiz0707/tapvidmv.git\" "
            f"{TAPVIDMV_ROOT} && "
            f"git -C {TAPVIDMV_ROOT} checkout {TAPVIDMV_REVISION} && "
            f"git -C {TAPVIDMV_ROOT} remote set-url origin https://github.com/Fxiz0707/tapvidmv.git"
        ),
        secrets=[github_secret],
    )
    .run_commands(
        f"python -m pip install -r {TAPVIDMV_ROOT / 'requirements.txt'}",
        "python -m pip install wandb",
    )
    .add_local_file(
        str(Path(__file__).parent / "patch_tapvidmv_performance.py"),
        str(TAPVIDMV_PERFORMANCE_PATCHER),
        copy=True,
    )
    .run_commands(
        f"python {TAPVIDMV_PERFORMANCE_PATCHER} {TAPVIDMV_ROOT}",
    )
    .env(
        {
            "PYTHONPATH": str(WORKSPACE),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "NUMEXPR_NUM_THREADS": "2",
        }
    )
)

app = modal.App(APP_NAME, tags=TAGS)
run_volume = modal.Volume.from_name("jeet-mvtracker-runs-v2")
eval_volume = modal.Volume.from_name("jeet-mvtracker-tapvidmv-eval-v1", create_if_missing=True)
wandb_secret = modal.Secret.from_name("jeet-mvtracker-wandb")


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


def _wandb_run(run_name: str, stage: str):
    import wandb

    return wandb.init(
        project="mvtracker-external-evaluation",
        name=f"{run_name}-{stage}",
        group=run_name,
        job_type=stage,
        tags=["modal", "tapvidmv", "reference-depth", "step1500", stage],
        config={
            **TAGS,
            "run_name": run_name,
            "stage": stage,
            "tapvidmv_revision": TAPVIDMV_REVISION,
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "sources": list(SOURCE_NAMES),
            "predictor": "reconstruction__mvtracker",
            "resolution": 512,
            "seed": 72,
        },
    )


def _run_logged(command: list[str], *, cwd: Path, stage: str) -> None:
    _log("command_started", stage=stage, command=command, cwd=str(cwd))
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        text = line.rstrip()
        if text:
            _log("command_output", stage=stage, message=text)
    returncode = process.wait()
    elapsed = time.perf_counter() - started
    _log("command_finished", stage=stage, returncode=returncode, seconds=round(elapsed, 3))
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)


def _download_file(source: str, url: str, target: Path) -> tuple[str, int, bool, float]:
    if target.is_file():
        return str(target), target.stat().st_size, True, 0.0
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    _log("download_file_started", source=source, url=url, target=str(target))
    started = time.perf_counter()
    error = None
    for attempt in range(1, 4):
        try:
            bytes_written = 0
            request = urllib.request.Request(url, headers={"User-Agent": "mvtracker-tapvidmv"})
            with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as output:
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
                    bytes_written += len(chunk)
            partial.replace(target)
            elapsed = time.perf_counter() - started
            _log(
                "download_file_completed",
                source=source,
                target=str(target),
                bytes=bytes_written,
                seconds=round(elapsed, 3),
                mib_per_second=round(bytes_written / 2**20 / max(elapsed, 1e-6), 3),
            )
            return str(target), bytes_written, False, elapsed
        except Exception as exc:  # report and retry the exact README URL
            error = exc
            _log(
                "download_file_retry",
                source=source,
                target=str(target),
                attempt=attempt,
                error=repr(exc),
            )
            partial.unlink(missing_ok=True)
    raise RuntimeError(f"failed to download {url} to {target}: {error}")


def _source_download_jobs(source: str, base_url: str, strip_prefix: str):
    manifest_url = f"{BUCKET_ROOT}/{source}_file_list.txt"
    _log("manifest_download_started", source=source, url=manifest_url)
    with urllib.request.urlopen(manifest_url, timeout=300) as response:
        entries = [line.strip() for line in response.read().decode().splitlines() if line.strip()]
    jobs = []
    for entry in entries:
        relative = (
            entry[len(strip_prefix) :]
            if strip_prefix and entry.startswith(strip_prefix)
            else entry
        )
        jobs.append((source, f"{base_url}/{entry}", DATASET_ROOT / source / relative))
    _log("manifest_ready", source=source, files=len(jobs))
    return jobs


def _dataset_inventory() -> dict[str, dict[str, int]]:
    sys.path.insert(0, str(WORKSPACE))
    from tapvidmv.splits import tapvidmv_splits

    inventory = {}
    for source, _base_url, _strip_prefix, expected_sequences in SOURCES:
        sequence_names = tapvidmv_splits.get_eval_sequences(source)
        if len(sequence_names) != expected_sequences:
            raise RuntimeError(
                f"{source}: split has {len(sequence_names)} sequences, expected {expected_sequences}"
            )
        depth_files = 0
        view_count = 0
        for sequence_name in sequence_names:
            sequence_root = DATASET_ROOT / source / sequence_name
            if not (sequence_root / "tracks_xyz.npy").is_file():
                raise FileNotFoundError(sequence_root / "tracks_xyz.npy")
            if not (sequence_root / "queries_xytv.npy").is_file():
                raise FileNotFoundError(sequence_root / "queries_xytv.npy")
            view_dirs = [path for path in sequence_root.iterdir() if path.is_dir() and path.name.isdigit()]
            if not view_dirs:
                raise RuntimeError(f"{sequence_root}: no numeric view directories")
            for view_dir in view_dirs:
                view_count += 1
                available_depths = [
                    view_dir / filename
                    for filename in ("depth.npy", "depth.npy.gz", "depth.npz")
                    if (view_dir / filename).is_file()
                ]
                if len(available_depths) != 1:
                    raise RuntimeError(f"{view_dir}: expected one reference depth, found {available_depths}")
                depth_files += 1
        inventory[source] = {
            "sequences": len(sequence_names),
            "views": view_count,
            "depth_files": depth_files,
        }
    return inventory


@app.function(
    image=tapvidmv_image,
    cpu=16,
    memory=32768,
    timeout=24 * 60 * 60,
    volumes={EVAL_MOUNT: eval_volume},
    secrets=[wandb_secret],
)
def download_dataset(run_name: str) -> dict:
    run = _wandb_run(run_name, "download")
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    total_downloaded = 0
    total_reused = 0
    failures = []
    started = time.perf_counter()
    _log("dataset_download_started", root=str(DATASET_ROOT), workers=16)
    for source, base_url, strip_prefix, _expected_sequences in SOURCES:
        source_started = time.perf_counter()
        jobs = _source_download_jobs(source, base_url, strip_prefix)
        downloaded = 0
        reused = 0
        source_bytes = 0
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = [pool.submit(_download_file, *job) for job in jobs]
            for completed, future in enumerate(as_completed(futures), start=1):
                try:
                    _path, size, was_reused, _elapsed = future.result()
                    source_bytes += size
                    if was_reused:
                        reused += 1
                    else:
                        downloaded += 1
                except Exception as exc:
                    failures.append({"source": source, "error": repr(exc)})
                if completed % 50 == 0 or completed == len(futures):
                    _log(
                        "download_source_progress",
                        source=source,
                        completed=completed,
                        total=len(futures),
                        downloaded=downloaded,
                        reused=reused,
                        failures=sum(1 for failure in failures if failure["source"] == source),
                    )
        eval_volume.commit()
        source_elapsed = time.perf_counter() - source_started
        total_downloaded += downloaded
        total_reused += reused
        _log(
            "download_source_completed",
            source=source,
            downloaded=downloaded,
            reused=reused,
            bytes=source_bytes,
            seconds=round(source_elapsed, 3),
            volume_committed=True,
        )
        run.log(
            {
                f"download/{source}/files_downloaded": downloaded,
                f"download/{source}/files_reused": reused,
                f"download/{source}/bytes": source_bytes,
                f"download/{source}/seconds": source_elapsed,
            }
        )

    if failures:
        _log("dataset_download_failed", failures=failures)
        run.log({"download/failures": len(failures)})
        run.finish(exit_code=1)
        raise RuntimeError(f"{len(failures)} TAPVidMV files failed: {failures[:10]}")

    _run_logged(
        [sys.executable, str(TAPVIDMV_ROOT / "scripts/fix_bucket_exports.py"), "pace"],
        cwd=EVAL_MOUNT,
        stage="pace-float32-fix",
    )
    inventory = _dataset_inventory()
    eval_volume.commit()
    elapsed = time.perf_counter() - started
    _log(
        "dataset_download_completed",
        seconds=round(elapsed, 3),
        downloaded=total_downloaded,
        reused=total_reused,
        inventory=inventory,
        volume_committed=True,
    )
    run.log(
        {
            "download/seconds": elapsed,
            "download/files_downloaded": total_downloaded,
            "download/files_reused": total_reused,
            "download/sequences": sum(item["sequences"] for item in inventory.values()),
        }
    )
    run.summary["inventory"] = inventory
    run.finish()
    return inventory


def _checkpoint_path() -> Path:
    return CHECKPOINT_MOUNT / CHECKPOINT_RELATIVE_PATH


def _prepare_checkpoint() -> Path:
    checkpoint = _checkpoint_path()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    actual_sha = digest.hexdigest()
    if actual_sha != CHECKPOINT_SHA256:
        raise RuntimeError(f"checkpoint SHA-256 {actual_sha} != {CHECKPOINT_SHA256}")
    checkpoint_dir = WORKSPACE / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    expected_path = checkpoint_dir / "mvtracker_200000_june2025.pth"
    expected_path.unlink(missing_ok=True)
    expected_path.symlink_to(checkpoint)
    _log("checkpoint_ready", source=str(checkpoint), linked_as=str(expected_path), sha256=actual_sha)
    return expected_path


def _gpu_monitor(stop: threading.Event) -> None:
    while not stop.wait(30):
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
        )
        _log("gpu_status", output=result.stdout.strip(), returncode=result.returncode)


def _prediction_complete(output_dir: Path) -> bool:
    return (output_dir / "config.yaml").is_file() and (
        output_dir / "tracks_xyz_worldspace.npy"
    ).is_file()


@app.function(
    image=tapvidmv_image,
    gpu="H100",
    cpu=16,
    memory=98304,
    timeout=24 * 60 * 60,
    volumes={
        CHECKPOINT_MOUNT: run_volume.with_mount_options(read_only=True),
        EVAL_MOUNT: eval_volume,
    },
    secrets=[wandb_secret],
)
def predict_reference_depth(run_name: str) -> dict:
    import torch

    run = _wandb_run(run_name, "predict")
    _prepare_checkpoint()
    inventory = _dataset_inventory()
    prediction_parent = RUNS_ROOT / run_name / "tapvidmv_predictions"
    method_root = prediction_parent / "reconstruction__mvtracker"
    method_root.mkdir(parents=True, exist_ok=True)
    _log(
        "prediction_started",
        gpu=torch.cuda.get_device_name(0),
        dataset_root=str(DATASET_ROOT),
        prediction_root=str(method_root),
        inventory=inventory,
    )

    _run_logged(
        [
            sys.executable,
            "-c",
            (
                "from tapvidmv.predictors.reconstruction_conditioned import _build_model; "
                "cfg, model = _build_model('mvtracker', 'cpu'); "
                "print(f'strict checkpoint load passed: {sum(p.numel() for p in model.parameters())} parameters', flush=True)"
            ),
        ],
        cwd=WORKSPACE,
        stage="strict-checkpoint-load",
    )

    stop_monitor = threading.Event()
    monitor = threading.Thread(target=_gpu_monitor, args=(stop_monitor,), daemon=True)
    monitor.start()
    started = time.perf_counter()
    source_timings = {}
    try:
        sys.path.insert(0, str(WORKSPACE))
        from tapvidmv.splits import tapvidmv_splits

        for source in SOURCE_NAMES:
            sequence_names = tapvidmv_splits.get_eval_sequences(source)
            incomplete = [
                name
                for name in sequence_names
                if not _prediction_complete(method_root / source / name)
            ]
            if not incomplete:
                _log("prediction_source_reused", source=source, sequences=len(sequence_names))
                source_timings[source] = 0.0
                continue
            source_started = time.perf_counter()
            lane_count = SOURCE_PREDICTION_LANES.get(source, PREDICTION_LANES)
            lanes = [
                incomplete[lane_idx::lane_count]
                for lane_idx in range(min(lane_count, len(incomplete)))
            ]
            _log(
                "prediction_source_started",
                source=source,
                pending=len(incomplete),
                total=len(sequence_names),
                lanes=len(lanes),
                lane_sizes=[len(lane) for lane in lanes],
            )
            with ThreadPoolExecutor(max_workers=len(lanes)) as executor:
                lane_futures = {}
                for lane_idx, lane_sequences in enumerate(lanes):
                    command = [
                        sys.executable,
                        "-u",
                        "-m",
                        "tapvidmv.run_predictor",
                        "--predictor",
                        "reconstruction__mvtracker",
                        "--tapvidmv_dir",
                        str(DATASET_ROOT),
                        "--tapvidmv_predictions",
                        str(method_root),
                        "--data_sources_to_predict",
                        source,
                        "--resolution",
                        "512",
                        "--seed",
                        "72",
                        "--loader_workers",
                        str(LOADER_WORKERS_PER_LANE),
                        "--sequences",
                        ",".join(lane_sequences),
                        "--overwrite_predictions",
                    ]
                    stage = f"predict-{source}-lane-{lane_idx + 1}"
                    _log(
                        "prediction_lane_started",
                        source=source,
                        lane=lane_idx + 1,
                        sequences=len(lane_sequences),
                    )
                    lane_futures[
                        executor.submit(_run_logged, command, cwd=WORKSPACE, stage=stage)
                    ] = lane_idx + 1
                for future in as_completed(lane_futures):
                    lane_idx = lane_futures[future]
                    future.result()
                    _log("prediction_lane_completed", source=source, lane=lane_idx)
            eval_volume.commit()
            source_elapsed = time.perf_counter() - source_started
            source_timings[source] = source_elapsed
            _log(
                "prediction_source_completed",
                source=source,
                sequences=len(sequence_names),
                seconds=round(source_elapsed, 3),
                volume_committed=True,
            )
            run.log({f"prediction/{source}/seconds": source_elapsed})
    finally:
        stop_monitor.set()
        monitor.join(timeout=5)

    missing = {}
    for source in SOURCE_NAMES:
        sys.path.insert(0, str(WORKSPACE))
        from tapvidmv.splits import tapvidmv_splits

        missing[source] = [
            name
            for name in tapvidmv_splits.get_eval_sequences(source)
            if not _prediction_complete(method_root / source / name)
        ]
    missing = {source: names for source, names in missing.items() if names}
    if missing:
        run.summary["missing_predictions"] = missing
        run.finish(exit_code=1)
        raise RuntimeError(f"missing TAPVidMV predictions: {missing}")
    elapsed = time.perf_counter() - started
    eval_volume.commit()
    _log(
        "prediction_completed",
        seconds=round(elapsed, 3),
        source_timings=source_timings,
        volume_committed=True,
    )
    run.log({"prediction/seconds": elapsed, "prediction/sequences": 253})
    run.summary["source_seconds"] = source_timings
    run.finish()
    return {"seconds": elapsed, "source_seconds": source_timings}


@app.function(
    image=tapvidmv_image,
    cpu=16,
    memory=32768,
    timeout=12 * 60 * 60,
    volumes={EVAL_MOUNT: eval_volume},
    secrets=[wandb_secret],
)
def evaluate_predictions(run_name: str) -> dict:
    run = _wandb_run(run_name, "evaluate")
    run_root = RUNS_ROOT / run_name
    prediction_parent = run_root / "tapvidmv_predictions"
    method_root = prediction_parent / "reconstruction__mvtracker"
    preview_root = prediction_parent / "previews"
    report_path = prediction_parent / "report/index.html"
    _log(
        "evaluation_started",
        method_root=str(method_root),
        preview_root=str(preview_root),
    )
    started = time.perf_counter()
    for source in SOURCE_NAMES:
        _run_logged(
            [
                sys.executable,
                "-u",
                "-m",
                "tapvidmv.evaluation.evaluate_model",
                "--tapvidmv_dir",
                str(DATASET_ROOT),
                "--tapvidmv_predictions",
                str(method_root),
                "--preview_root",
                str(preview_root),
                "--data_sources_to_evaluate",
                source,
                "--depth_scalings",
                "median_on_queries,median",
                "--world_scalings",
                "median,median_on_queries",
                "--resolution",
                "512",
            ],
            cwd=WORKSPACE,
            stage=f"evaluate-{source}",
        )
        eval_volume.commit()
        _log("evaluation_source_completed", source=source, volume_committed=True)

    for source in SOURCE_NAMES:
        _run_logged(
            [
                sys.executable,
                "-u",
                "-m",
                "tapvidmv.scripts.build_prediction_preview",
                "--method",
                "reconstruction__mvtracker",
                "--data_source",
                source,
                "--tapvidmv_dir",
                str(DATASET_ROOT),
                "--tapvidmv_predictions",
                str(prediction_parent),
                "--preview_root",
                str(preview_root),
                "--preview_depth_scaling",
                "median_on_queries",
                "--report_href",
                "../../../report/index.html",
            ],
            cwd=WORKSPACE,
            stage=f"preview-{source}",
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    _run_logged(
        [
            sys.executable,
            "-u",
            "-m",
            "tapvidmv.scripts.build_evaluation_report",
            "--preview_root",
            str(preview_root),
            "--out_path",
            str(report_path),
        ],
        cwd=WORKSPACE,
        stage="build-report",
    )
    if not report_path.is_file():
        raise FileNotFoundError(report_path)
    metrics_files = sorted(preview_root.rglob("metrics.json"))
    if not metrics_files:
        raise RuntimeError(f"no metrics.json files below {preview_root}")
    elapsed = time.perf_counter() - started
    eval_volume.commit()
    result = {
        "seconds": elapsed,
        "metrics_files": len(metrics_files),
        "report_path": str(report_path),
        "preview_root": str(preview_root),
    }
    _log("evaluation_completed", **result, volume_committed=True)
    run.log({"evaluation/seconds": elapsed, "evaluation/metrics_files": len(metrics_files)})
    run.summary.update(result)
    run.finish()
    return result


@app.local_entrypoint()
def main(stage: str = "full", run_name: str = "") -> None:
    if not run_name:
        run_name = f"tapvidmv-reference-step1500-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    app.set_tags({**TAGS, "run_name": run_name, "stage": stage})
    _log("orchestration_started", stage=stage, run_name=run_name)
    if stage in {"full", "download"}:
        inventory = download_dataset.remote(run_name)
        _log("download_result", inventory=inventory)
    if stage in {"full", "predict"}:
        prediction = predict_reference_depth.remote(run_name)
        _log("prediction_result", result=prediction)
    if stage in {"full", "evaluate"}:
        evaluation = evaluate_predictions.remote(run_name)
        _log("evaluation_result", result=evaluation)
    if stage not in {"full", "download", "predict", "evaluate"}:
        raise ValueError(f"unsupported stage: {stage}")
    _log("orchestration_completed", stage=stage, run_name=run_name)
