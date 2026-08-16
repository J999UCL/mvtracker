"""Modal launcher for the bounded VGGT-Omega H100 throughput profile.

This is intentionally small.  It stages inputs on container-local SSD, then
delegates the actual scene-batch benchmark to the VGGT-Omega preprocessing
profile implementation.  It does not write depth outputs to the Modal Volume.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil

import modal


APP_NAME = "jeet-mvtracker-vggt-omega-profile"
DATA_VOLUME_NAME = "jeet-mvtracker-data-v2"
RUN_VOLUME_NAME = "jeet-mvtracker-runs-v2"
HF_SECRET_NAME = "jeet-mvtracker-huggingface"
WANDB_SECRET_NAME = "jeet-mvtracker-wandb"
DATA_ROOT = Path("/mnt/mvtracker-data")
RUN_ROOT = Path("/mnt/mvtracker-runs")
LOCAL_ROOT = Path("/tmp/vggt-omega-profile")
CHECKPOINT_REPO = "facebook/VGGT-Omega"
CHECKPOINT_REVISION = "ba9db085d6b7349b738fa2e37d198bb4dd077954"
CHECKPOINT_FILENAME = "vggt_omega_1b_512.pt"
BASE_TAGS = {"owner": "jeet", "project": "mvtracker", "purpose": "profiling"}
_COMMIT = re.compile(r"[0-9a-f]{40}")


def _source_commit() -> str:
    commit = os.environ.get("MVTRACKER_MODAL_COMMIT", "")
    if _COMMIT.fullmatch(commit) is None:
        raise RuntimeError("MVTRACKER_MODAL_COMMIT must be one full lowercase Git commit")
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
    requirements = Path(__file__).resolve().parents[1] / "requirements.vggt-omega.txt"
    return (
        modal.Image.from_registry(
            "nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.10"
        )
        .apt_install("git", "libgl1", "libglib2.0-0")
        .pip_install(
            "torch==2.7.1",
            "torchvision==0.22.1",
            "torchaudio==2.7.1",
            index_url="https://download.pytorch.org/whl/cu128",
        )
        .pip_install_from_requirements(str(requirements))
        .pip_install("huggingface-hub==0.30.2", "wandb==0.19.9", "Pillow")
        .run_commands(clone)
        .env(
            {
                "MVTRACKER_MODAL_COMMIT": commit,
                "PYTHONPATH": "/opt/mvtracker:/opt/mvtracker/tools",
                "TORCH_CUDA_ARCH_LIST": "9.0",
            }
        )
    )


def _benchmark_dataset(
    name: str,
    sources,
    candidates: tuple[int, ...],
    *,
    loader_workers: tuple[int, ...],
    write_root: Path,
    model,
    device,
    infer_temporal_chunks,
    log,
) -> dict:
    """Choose a small loader setting, then sweep physical scene batch sizes."""
    import time

    import numpy as np
    import torch

    frame_indices = tuple(range(24))
    total_memory = torch.cuda.get_device_properties(device).total_memory
    write_root.mkdir(parents=True, exist_ok=True)
    def measure(batch_size: int, workers: int) -> dict:
        selected_sources = tuple(sources[:batch_size])
        trial = {"dataset": name, "batch_size": batch_size, "loader_workers": workers}
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            infer_temporal_chunks(
                selected_sources,
                frame_indices,
                model,
                device=device,
                image_resolution=512,
                loader_workers=workers,
            )
            torch.cuda.synchronize(device)
            measurements = []
            for iteration in range(3):
                result = infer_temporal_chunks(
                    selected_sources,
                    frame_indices,
                    model,
                    device=device,
                    image_resolution=512,
                    loader_workers=workers,
                )
                torch.cuda.synchronize(device)
                measurements.append(result.timings)
                write_started = time.perf_counter()
                for scene_index, scene in enumerate(result.scenes):
                    prefix = write_root / f"{name}-{scene_index}"
                    np.save(prefix.with_name(prefix.name + "-depth.npy"), scene.depth)
                    np.save(
                        prefix.with_name(prefix.name + "-cleaned-mask.npy"),
                        scene.cleaned_mask,
                    )
                    np.save(
                        prefix.with_name(prefix.name + "-intrinsics.npy"),
                        scene.intrinsics,
                    )
                    np.save(
                        prefix.with_name(prefix.name + "-extrinsics-w2c.npy"),
                        scene.extrinsics_w2c,
                    )
                    np.save(
                        prefix.with_name(prefix.name + "-scales.npy"),
                        np.asarray(scene.scale, dtype=np.float32),
                    )
                result_write_seconds = time.perf_counter() - write_started
                measurements[-1] = (result.timings, result_write_seconds)
                del result
            total = sorted(item.total_seconds + write for item, write in measurements)[1]
            load = sorted(item.load_preprocess_seconds for item, _ in measurements)[1]
            model_seconds = sorted(item.model_seconds for item, _ in measurements)[1]
            post = sorted(item.postprocess_seconds for item, _ in measurements)[1]
            write_seconds = sorted(write for _, write in measurements)[1]
            peak_reserved = torch.cuda.max_memory_reserved(device)
            safe = peak_reserved <= total_memory * 0.90
            trial.update(
                {
                    "status": "ok" if safe else "unsafe_vram",
                    "scene_count": batch_size,
                    "image_count": measurements[0][0].image_count,
                    "total_seconds": total,
                    "load_preprocess_seconds": load,
                    "model_seconds": model_seconds,
                    "postprocess_seconds": post,
                    "write_seconds": write_seconds,
                    "scenes_per_second": batch_size / total,
                    "peak_reserved_bytes": int(peak_reserved),
                    "peak_vram_fraction": peak_reserved / total_memory,
                }
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            trial["status"] = "oom"
        log({f"profile/{name}/{key}": value for key, value in trial.items() if isinstance(value, (int, float))})
        return trial

    worker_trials = []
    for workers in loader_workers:
        trial = measure(1, workers)
        worker_trials.append(trial)
    safe_workers = [trial for trial in worker_trials if trial["status"] == "ok"]
    worker_choice = max(safe_workers, key=lambda trial: trial["scenes_per_second"])
    trials = []
    for batch_size in candidates:
        trial = measure(batch_size, worker_choice["loader_workers"])
        trials.append(trial)
        if trial["status"] in {"oom", "unsafe_vram"}:
            break
    return {
        "loader_trials": worker_trials,
        "selected_loader_workers": worker_choice["loader_workers"],
        "batch_trials": trials,
        "selected": max(
            (trial for trial in trials if trial["status"] == "ok"),
            key=lambda trial: trial["scenes_per_second"],
            default=None,
        ),
    }


app = modal.App(APP_NAME, tags={**BASE_TAGS, "experiment": "vggt-omega-throughput"})
image = _image()
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False, version=2)
run_volume = modal.Volume.from_name(RUN_VOLUME_NAME, create_if_missing=True, version=2)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])
wandb_secret = modal.Secret.from_name(WANDB_SECRET_NAME, required_keys=["WANDB_API_KEY"])


@app.function(
    image=image,
    gpu="H100!",
    secrets=[hf_secret, wandb_secret],
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
        str(RUN_ROOT): run_volume,
    },
    cpu=8,
    memory=65536,
    ephemeral_disk=512 * 1024,
    timeout=45 * 60,
    max_containers=1,
    include_source=False,
)
def profile(run_name: str = "vggt-omega-h100-throughput") -> dict:
    """Stage the profile subset and run the core VGGT-Omega benchmark."""
    import time

    import torch
    import wandb
    from huggingface_hub import hf_hub_download

    from mvtracker.preprocessing.vggt_omega import (
        MVKubricSceneSource,
        TapVid3DSceneSource,
        infer_temporal_chunks,
        load_model,
    )
    from mvtracker.profiling.modal_vggt_omega import (
        stage_diegesis,
        stage_mvkubric,
        write_staging_report,
    )

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        raise ValueError("run name contains unsupported characters")
    if LOCAL_ROOT.exists():
        shutil.rmtree(LOCAL_ROOT)
    LOCAL_ROOT.mkdir(parents=True)
    run = wandb.init(
        project="mvtracker-modal-profiling",
        job_type="vggt-omega-throughput",
        name=run_name,
        tags=["modal", "h100", "vggt-omega", "throughput"],
        config={
            **BASE_TAGS,
            "gpu": "H100",
            "temporal_chunk_size": 24,
            "image_resolution": 512,
            "diegesis_revision": "81389015a6d713a848a120e34850f360621bcdce",
            "mvkubric_scenes": ["900", "901", "902", "903"],
        },
    )

    started = time.perf_counter()
    diegesis = stage_diegesis(LOCAL_ROOT / "diegesis", os.environ["HF_TOKEN"])
    mvkubric = stage_mvkubric(
        DATA_ROOT / "datasets/kubric-multiview/train",
        LOCAL_ROOT / "mv-kubric/train",
    )
    checkpoint_root = LOCAL_ROOT / "checkpoint"
    checkpoint_path = Path(
        hf_hub_download(
            repo_id=CHECKPOINT_REPO,
            filename=CHECKPOINT_FILENAME,
            revision=CHECKPOINT_REVISION,
            token=os.environ["HF_TOKEN"],
            local_dir=checkpoint_root,
        )
    )
    staging = write_staging_report(
        LOCAL_ROOT / "staging.json", diegesis=diegesis, mvkubric=mvkubric
    )
    run.summary.update(
        {
            "staging/diegesis_download_seconds": diegesis["download_seconds"],
            "staging/diegesis_bytes": diegesis["size_bytes"],
            "staging/mvkubric_bytes": mvkubric["size_bytes"],
            "staging/seconds": time.perf_counter() - started,
        }
    )

    diegesis_root = Path(diegesis["scenes_root"])
    diegesis_scene_names = (
        "kitchen03",
        "kitchen04",
        "livingroom01",
        "livingroom03",
        "livingroom04",
        "livingroom05",
        "diningroom01",
        "diningroom03",
    )
    diegesis_sources = [
        TapVid3DSceneSource(
            diegesis_root / scene / "tracking/sequence", view_ids=(0, 1, 2, 3)
        )
        for scene in diegesis_scene_names
    ]
    mvkubric_sources = [
        MVKubricSceneSource(Path(mvkubric["local_root"]) / scene)
        for scene in mvkubric["scene_ids"]
    ]
    device = torch.device("cuda")
    model = load_model(checkpoint_path, device)
    profile_report = {
        "diegesis": _benchmark_dataset(
            "diegesis",
            diegesis_sources,
            (1, 2, 4, 6, 8),
            loader_workers=(1, 4, 8),
            write_root=LOCAL_ROOT / "profile-output",
            model=model,
            device=device,
            infer_temporal_chunks=infer_temporal_chunks,
            log=run.log,
        ),
        "mv-kubric": _benchmark_dataset(
            "mv-kubric",
            mvkubric_sources,
            (1, 2, 3),
            loader_workers=(1, 4, 8),
            write_root=LOCAL_ROOT / "profile-output",
            model=model,
            device=device,
            infer_temporal_chunks=infer_temporal_chunks,
            log=run.log,
        ),
        "loader": {"source": "local container SSD", "candidates": [1, 4, 8]},
    }
    report = {
        "format": "mvtracker_vggt_omega_modal_profile",
        "run_name": run_name,
        "staging": staging,
        "checkpoint": {
            "repo_id": CHECKPOINT_REPO,
            "revision": CHECKPOINT_REVISION,
            "filename": CHECKPOINT_FILENAME,
        },
        "profile": profile_report,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output = RUN_ROOT / run_name / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run.summary.update({"report_path": str(output), "profile": profile_report})
    run.finish()
    run_volume.commit()
    return report


if __name__ == "__main__":
    print("Use: modal run --timestamps tools/modal_vggt_omega_profile.py::profile")
