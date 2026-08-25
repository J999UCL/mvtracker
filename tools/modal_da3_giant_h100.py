"""DA3-Giant-1.1 H100 capacity, throughput, and DIEGESIS quality benchmark."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time

import modal


APP_NAME = "jeet-mvtracker-da3-giant-benchmark"
DATA_ROOT = Path("/mnt/mvtracker-data")
RUN_ROOT = Path("/mnt/mvtracker-runs/da3-giant-benchmark")
MODEL_ID = "depth-anything/DA3-GIANT-1.1"
DA3_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
TAGS = {
    "owner": "jeet",
    "project": "mvtracker",
    "purpose": "profiling",
    "experiment": "da3-giant-1.1-diegesis",
    "gpu": "h100-h200",
}


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


image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-runtime-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
        index_url="https://download.pytorch.org/whl/cu128",
    )
    .pip_install("wandb==0.19.9", "hf-xet==1.1.9")
    .pip_install(f"git+https://github.com/ByteDance-Seed/Depth-Anything-3.git@{DA3_REVISION}")
)

app = modal.App(APP_NAME, tags=TAGS)
data_volume = modal.Volume.from_name("jeet-mvtracker-data-v2", create_if_missing=False, version=2)
run_volume = modal.Volume.from_name("jeet-mvtracker-runs-v2", create_if_missing=True, version=2)
hf_secret = modal.Secret.from_name("jeet-mvtracker-huggingface", required_keys=["HF_TOKEN"])
wandb_secret = modal.Secret.from_name("jeet-mvtracker-wandb", required_keys=["WANDB_API_KEY"])


def _intrinsics_matrix(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float32)
    if values.shape == (3, 3):
        return values
    fx, fy, cx, cy = values
    return np.asarray(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)), dtype=np.float32)


def _metrics(prediction, target, mask):
    import numpy as np

    usable = mask & np.isfinite(prediction) & np.isfinite(target) & (prediction > 0) & (target > 0)
    pred = prediction[usable].astype(np.float64)
    gt = target[usable].astype(np.float64)
    ratio = np.maximum(pred / gt, gt / pred)
    error = pred - gt
    return {
        "pixels": int(gt.size),
        "abs_rel": float(np.mean(np.abs(error) / gt)),
        "rmse_m": float(np.sqrt(np.mean(error * error))),
        "delta1": float(np.mean(ratio < 1.25)),
        "pred_to_gt_median_ratio": float(np.median(pred / gt)),
    }


@app.function(
    image=image,
    gpu="H100!",
    cpu=8,
    memory=65536,
    timeout=30 * 60,
    volumes={DATA_ROOT: data_volume, Path("/mnt/mvtracker-runs"): run_volume},
    secrets=[hf_secret, wandb_secret],
)
def benchmark(
    scene: str = "diningroom02",
    timestamp_count: int = 24,
    gpu_label: str = "h100",
):
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    import wandb
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.pose_align import align_poses_umeyama

    run_name = (
        f"da3-giant-1.1-{gpu_label}-{scene}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_root = RUN_ROOT / run_name
    output_root.mkdir(parents=True, exist_ok=False)
    run = wandb.init(
        project="mvtracker-depth-evaluation",
        name=run_name,
        tags=["modal", "diegesis", "da3-giant-1.1", "pose-conditioned", gpu_label],
        config={
            **TAGS,
            "runtime_gpu": gpu_label,
            "model": MODEL_ID,
            "scene": scene,
            "timestamp_count": timestamp_count,
        },
    )
    _log("job_started", run_name=run_name, gpu=torch.cuda.get_device_name(0))

    scene_root = DATA_ROOT / "source/diegesis/scenes" / scene / "tracking/sequence"
    frame_count = int(np.load(scene_root / "tracks_xyz.npy", mmap_mode="r").shape[0])
    comparison_frames = [0, 34, 68, 102, 136, 170, 204, 239]
    extra_frames = [
        int(value)
        for value in np.linspace(0, frame_count - 1, max(timestamp_count * 2, 32))
    ]
    frame_indices = comparison_frames + [value for value in extra_frames if value not in comparison_frames]
    frame_indices = frame_indices[:timestamp_count]
    views = tuple(
        sorted(int(path.name) for path in scene_root.iterdir() if path.is_dir() and path.name.isdigit())
    )
    _log("scene_opened", scene=scene, frames=frame_indices, views=views)

    jpeg = {
        view: np.load(scene_root / str(view) / "images_jpeg_bytes.npy", allow_pickle=True)
        for view in views
    }
    depth = {
        view: np.load(scene_root / str(view) / "depth.npy", mmap_mode="r", allow_pickle=False)
        for view in views
    }
    foreground = {
        view: np.load(scene_root / str(view) / "foreground_mask.npy", mmap_mode="r", allow_pickle=False)
        for view in views
    }
    intrinsics = {
        view: _intrinsics_matrix(np.load(scene_root / str(view) / "intrinsics.npy"))
        for view in views
    }
    extrinsics = {
        view: np.load(scene_root / str(view) / "extrinsics_w2c.npy", mmap_mode="r")
        for view in views
    }

    _log("model_download_started", model=MODEL_ID)
    model_started = time.perf_counter()
    model = DepthAnything3.from_pretrained(MODEL_ID).to("cuda").eval()
    torch.cuda.synchronize()
    _log(
        "model_ready",
        seconds=round(time.perf_counter() - model_started, 3),
        allocated_gib=round(torch.cuda.memory_allocated() / 2**30, 3),
    )

    processed_images = []
    processed_extrinsics = []
    processed_intrinsics = []
    target_depths = []
    target_foregrounds = []
    _log("preprocessing_started", timestamps=len(frame_indices), images=len(frame_indices) * len(views))
    for index, frame in enumerate(frame_indices):
        images = []
        for view in views:
            encoded = np.asarray(jpeg[view][frame], dtype=np.uint8)
            with Image.open(io.BytesIO(encoded.tobytes())) as loaded:
                images.append(loaded.convert("RGB"))
        frame_extrinsics = np.stack([extrinsics[view][frame] for view in views]).astype(np.float32)
        frame_intrinsics = np.stack([intrinsics[view] for view in views]).astype(np.float32)
        imgs, exts, ints = model._preprocess_inputs(
            images,
            frame_extrinsics,
            frame_intrinsics,
            process_res=504,
            process_res_method="upper_bound_resize",
        )
        processed_images.append(imgs)
        processed_extrinsics.append(exts)
        processed_intrinsics.append(ints)
        target_depths.append(np.stack([np.asarray(depth[view][frame]) for view in views]))
        target_foregrounds.append(np.stack([np.asarray(foreground[view][frame]) for view in views]))
        _log("timestamp_preprocessed", completed=index + 1, total=len(frame_indices), frame=frame)

    cpu_images = torch.stack(processed_images)
    cpu_extrinsics = torch.stack(processed_extrinsics)
    cpu_intrinsics = torch.stack(processed_intrinsics)
    target_depths = np.stack(target_depths)
    target_foregrounds = np.stack(target_foregrounds)
    total_vram = torch.cuda.get_device_properties(0).total_memory
    trials = []

    batch_sizes = (20, 40, 44, 48) if gpu_label == "h200" else (1, 4, 8, 12, 16, 20, 24)
    for batch_size in (size for size in batch_sizes if size <= len(frame_indices)):
        _log("trial_started", batch_size=batch_size, images=batch_size * len(views))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        try:
            images = cpu_images[:batch_size].to("cuda", non_blocking=True).float()
            exts = cpu_extrinsics[:batch_size].to("cuda", non_blocking=True).float()
            ints = cpu_intrinsics[:batch_size].to("cuda", non_blocking=True).float()
            normalized_exts = torch.cat(
                [model._normalize_extrinsics(exts[index : index + 1].clone()) for index in range(batch_size)]
            )

            with torch.inference_mode():
                warm = model.forward(images, normalized_exts, ints, export_feat_layers=[])
            torch.cuda.synchronize()
            del warm

            timings = []
            output = None
            for _ in range(2):
                started = time.perf_counter()
                with torch.inference_mode():
                    output = model.forward(images, normalized_exts, ints, export_feat_layers=[])
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - started)

            median_seconds = float(np.median(timings))
            peak_reserved = torch.cuda.max_memory_reserved()
            trial = {
                "batch_size": batch_size,
                "images": batch_size * len(views),
                "median_seconds": median_seconds,
                "images_per_second": batch_size * len(views) / median_seconds,
                "peak_reserved_gib": peak_reserved / 2**30,
                "peak_vram_fraction": peak_reserved / total_vram,
                "status": "safe" if peak_reserved / total_vram <= 0.92 else "over_92pct",
            }

            predicted_depth = output["depth"].squeeze(-1).float().cpu().numpy()
            predicted_poses = output["extrinsics"].float().cpu().numpy()
            predicted_confidence = output["depth_conf"].float().cpu().numpy()
            for index in range(batch_size):
                _, _, scale, _ = align_poses_umeyama(
                    predicted_poses[index],
                    cpu_extrinsics[index].numpy(),
                    ransac=False,
                    return_aligned=True,
                )
                predicted_depth[index] /= scale

            target_hw = target_depths.shape[-2:]
            predicted_depth = F.interpolate(
                torch.from_numpy(predicted_depth.reshape(-1, *predicted_depth.shape[-2:])).unsqueeze(1),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )[:, 0].reshape(batch_size, len(views), *target_hw).numpy()
            predicted_confidence = F.interpolate(
                torch.from_numpy(predicted_confidence.reshape(-1, *predicted_confidence.shape[-2:])).unsqueeze(1),
                size=target_hw,
                mode="bilinear",
                align_corners=False,
            )[:, 0].reshape(batch_size, len(views), *target_hw).numpy()
            targets = target_depths[:batch_size]
            valid = np.isfinite(targets) & (targets > 0)
            conf_cutoff = float(np.quantile(predicted_confidence[valid], 0.40))
            trial["quality"] = {
                "all": _metrics(predicted_depth, targets, valid),
                "foreground": _metrics(
                    predicted_depth,
                    targets,
                    valid & target_foregrounds[:batch_size],
                ),
                "top_60pct_confidence": _metrics(
                    predicted_depth,
                    targets,
                    valid & (predicted_confidence >= conf_cutoff),
                ),
            }
            trials.append(trial)
            run.log(
                {
                    "capacity/batch_size": batch_size,
                    "capacity/images_per_second": trial["images_per_second"],
                    "capacity/peak_vram_fraction": trial["peak_vram_fraction"],
                    "quality/abs_rel": trial["quality"]["all"]["abs_rel"],
                    "quality/foreground_abs_rel": trial["quality"]["foreground"]["abs_rel"],
                },
                step=batch_size,
            )
            _log("trial_finished", **trial)
            del images, exts, ints, normalized_exts, output
            if trial["status"] != "safe":
                break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            trial = {"batch_size": batch_size, "images": batch_size * len(views), "status": "oom"}
            trials.append(trial)
            _log("trial_finished", **trial)
            break

    safe = [trial for trial in trials if trial["status"] == "safe"]
    selected = max(safe, key=lambda item: item["batch_size"])
    report = {
        "run_name": run_name,
        "model": MODEL_ID,
        "scene": scene,
        "gpu": torch.cuda.get_device_name(0),
        "views_per_timestamp": len(views),
        "trials": trials,
        "largest_safe": selected,
    }
    (output_root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    run.summary.update(
        {
            "largest_safe_batch": selected["batch_size"],
            "largest_safe_images_per_second": selected["images_per_second"],
            "largest_safe_peak_vram_fraction": selected["peak_vram_fraction"],
        }
    )
    run.finish()
    run_volume.commit()
    _log("benchmark_finished", output=str(output_root), largest_safe=selected)
    return report


@app.local_entrypoint()
def main(
    scene: str = "diningroom02",
    gpu: str = "H100!",
    timestamp_count: int = 24,
):
    result = benchmark.with_options(gpu=gpu).remote(
        scene=scene,
        timestamp_count=timestamp_count,
        gpu_label=gpu.lower().removesuffix("!"),
    )
    print(json.dumps(result, indent=2))
