"""Lean DA3-Large-1.1 metric-depth evaluation on one DIEGESIS scene."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time

import modal


APP_NAME = "jeet-mvtracker-da3-diegesis-eval"
DATA_VOLUME_NAME = "jeet-mvtracker-data-v2"
RUN_VOLUME_NAME = "jeet-mvtracker-runs-v2"
DATA_ROOT = Path("/mnt/mvtracker-data")
RUN_ROOT = Path("/mnt/mvtracker-runs/da3-diegesis-eval")
MODEL_ID = "depth-anything/DA3-LARGE-1.1"
DA3_REVISION = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
TAGS = {
    "owner": "jeet",
    "project": "mvtracker",
    "purpose": "evaluation",
    "experiment": "da3-large-1.1-diegesis",
    "gpu": "l4",
}


def _log(event: str, **fields) -> None:
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(json.dumps({"time": timestamp, "event": event, **fields}, sort_keys=True), flush=True)


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
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False, version=2)
run_volume = modal.Volume.from_name(RUN_VOLUME_NAME, create_if_missing=True, version=2)
hf_secret = modal.Secret.from_name("jeet-mvtracker-huggingface", required_keys=["HF_TOKEN"])
wandb_secret = modal.Secret.from_name("jeet-mvtracker-wandb", required_keys=["WANDB_API_KEY"])


def _intrinsics_matrix(values):
    import numpy as np

    values = np.asarray(values, dtype=np.float32)
    if values.shape == (3, 3):
        return values
    fx, fy, cx, cy = values
    return np.asarray(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)), dtype=np.float32)


def _metric_summary(prediction, target, mask):
    import numpy as np

    usable = mask & np.isfinite(prediction) & np.isfinite(target) & (prediction > 0) & (target > 0)
    pred = prediction[usable].astype(np.float64)
    gt = target[usable].astype(np.float64)
    ratio = np.maximum(pred / gt, gt / pred)
    error = pred - gt
    return {
        "pixels": int(gt.size),
        "coverage": float(usable.mean()),
        "abs_rel": float(np.mean(np.abs(error) / gt)),
        "rmse_m": float(np.sqrt(np.mean(error * error))),
        "mae_m": float(np.mean(np.abs(error))),
        "delta1": float(np.mean(ratio < 1.25)),
        "pred_to_gt_median_ratio": float(np.median(pred / gt)),
    }


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=30 * 60,
    volumes={DATA_ROOT: data_volume, Path("/mnt/mvtracker-runs"): run_volume},
    secrets=[hf_secret, wandb_secret],
)
def evaluate(scene: str = "diningroom02", timestamps: int = 8):
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    import wandb
    from depth_anything_3.api import DepthAnything3

    started = time.perf_counter()
    run_name = f"da3-large-1.1-{scene}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_root = RUN_ROOT / run_name
    output_root.mkdir(parents=True, exist_ok=False)
    run = wandb.init(
        project="mvtracker-depth-evaluation",
        name=run_name,
        tags=["modal", "diegesis", "da3-large-1.1", "pose-conditioned", "l4"],
        config={**TAGS, "model": MODEL_ID, "scene": scene, "timestamps": timestamps},
    )
    _log("job_started", run_name=run_name, gpu=torch.cuda.get_device_name(0))

    scene_root = DATA_ROOT / "source/diegesis/scenes" / scene / "tracking/sequence"
    tracks = np.load(scene_root / "tracks_xyz.npy", mmap_mode="r", allow_pickle=False)
    frame_indices = np.linspace(0, tracks.shape[0] - 1, timestamps, dtype=np.int64)
    views = tuple(int(path.name) for path in scene_root.iterdir() if path.is_dir() and path.name.isdigit())
    views = tuple(sorted(views))
    _log("scene_opened", scene=scene, frames=int(tracks.shape[0]), selected_frames=frame_indices.tolist(), views=views)

    jpeg_arrays = {
        view: np.load(scene_root / str(view) / "images_jpeg_bytes.npy", allow_pickle=True)
        for view in views
    }
    depth_arrays = {
        view: np.load(scene_root / str(view) / "depth.npy", mmap_mode="r", allow_pickle=False)
        for view in views
    }
    foreground_arrays = {
        view: np.load(scene_root / str(view) / "foreground_mask.npy", mmap_mode="r", allow_pickle=False)
        for view in views
    }
    intrinsics = {
        view: _intrinsics_matrix(np.load(scene_root / str(view) / "intrinsics.npy", allow_pickle=False))
        for view in views
    }
    extrinsics = {
        view: np.load(scene_root / str(view) / "extrinsics_w2c.npy", mmap_mode="r", allow_pickle=False)
        for view in views
    }
    _log("scene_metadata_loaded", elapsed_seconds=round(time.perf_counter() - started, 3))

    model_started = time.perf_counter()
    _log("model_download_started", model=MODEL_ID)
    model = DepthAnything3.from_pretrained(MODEL_ID).to("cuda").eval()
    torch.cuda.synchronize()
    _log(
        "model_ready",
        seconds=round(time.perf_counter() - model_started, 3),
        allocated_gib=round(torch.cuda.memory_allocated() / 2**30, 3),
    )

    predictions = []
    targets = []
    foregrounds = []
    confidences = []
    per_timestamp = []
    for sample_index, frame in enumerate(frame_indices):
        images = []
        target_depth = []
        target_foreground = []
        for view in views:
            encoded = np.asarray(jpeg_arrays[view][frame], dtype=np.uint8)
            with Image.open(io.BytesIO(encoded.tobytes())) as loaded:
                images.append(loaded.convert("RGB"))
            target_depth.append(np.asarray(depth_arrays[view][frame], dtype=np.float32))
            target_foreground.append(np.asarray(foreground_arrays[view][frame], dtype=bool))
        frame_extrinsics = np.stack([extrinsics[view][frame] for view in views]).astype(np.float32)
        frame_intrinsics = np.stack([intrinsics[view] for view in views]).astype(np.float32)
        _log("inference_started", sample=sample_index, frame=int(frame), images=len(images))
        infer_started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        with torch.inference_mode():
            prediction = model.inference(
                image=images,
                extrinsics=frame_extrinsics,
                intrinsics=frame_intrinsics,
                align_to_input_ext_scale=True,
                process_res=504,
                process_res_method="upper_bound_resize",
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - infer_started
        predicted_depth = torch.from_numpy(np.asarray(prediction.depth)).unsqueeze(1)
        target_hw = target_depth[0].shape
        predicted_depth = F.interpolate(
            predicted_depth,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )[:, 0].numpy()
        confidence = torch.from_numpy(np.asarray(prediction.conf)).unsqueeze(1)
        confidence = F.interpolate(confidence, size=target_hw, mode="bilinear", align_corners=False)[:, 0].numpy()
        predictions.append(predicted_depth)
        targets.append(np.stack(target_depth))
        foregrounds.append(np.stack(target_foreground))
        confidences.append(confidence)
        timing = {
            "sample": sample_index,
            "frame": int(frame),
            "seconds": seconds,
            "images_per_second": len(images) / seconds,
            "peak_vram_gib": torch.cuda.max_memory_reserved() / 2**30,
        }
        per_timestamp.append(timing)
        wandb.log({f"inference/{key}": value for key, value in timing.items() if key not in {"sample", "frame"}}, step=sample_index)
        _log("inference_finished", **{key: round(value, 4) if isinstance(value, float) else value for key, value in timing.items()})

    prediction = np.stack(predictions)
    target = np.stack(targets)
    foreground = np.stack(foregrounds)
    confidence = np.stack(confidences)
    valid = np.isfinite(target) & (target > 0)
    confidence_cutoff = float(np.quantile(confidence[valid], 0.40))
    report = {
        "run_name": run_name,
        "model": MODEL_ID,
        "scene": scene,
        "frames": frame_indices.tolist(),
        "views": list(views),
        "all_valid": _metric_summary(prediction, target, valid),
        "foreground": _metric_summary(prediction, target, valid & foreground),
        "top_60pct_confidence": _metric_summary(prediction, target, valid & (confidence >= confidence_cutoff)),
        "confidence_40th_percentile": confidence_cutoff,
        "timings": per_timestamp,
        "steady_state_median_seconds_per_timestamp": float(np.median([item["seconds"] for item in per_timestamp[1:]])),
        "steady_state_images_per_second": float(len(views) / np.median([item["seconds"] for item in per_timestamp[1:]])),
        "total_seconds": time.perf_counter() - started,
    }
    (output_root / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    run.log(
        {
            "quality/all_abs_rel": report["all_valid"]["abs_rel"],
            "quality/all_delta1": report["all_valid"]["delta1"],
            "quality/foreground_abs_rel": report["foreground"]["abs_rel"],
            "quality/foreground_delta1": report["foreground"]["delta1"],
            "throughput/steady_state_images_per_second": report["steady_state_images_per_second"],
            "throughput/median_seconds_per_timestamp": report["steady_state_median_seconds_per_timestamp"],
        }
    )
    run.finish()
    run_volume.commit()
    _log("evaluation_finished", output=str(output_root), report=report)
    return report


@app.local_entrypoint()
def main(scene: str = "diningroom02", timestamps: int = 8):
    print(json.dumps(evaluate.remote(scene=scene, timestamps=timestamps), indent=2))
