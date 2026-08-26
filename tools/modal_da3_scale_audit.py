"""Replay one recipe sample and audit DA3 metric-scale alignment."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import modal

from modal_continual_training import da3_training_image
from modal_training_profile import data_volume, hf_secret, run_volume, wandb_secret


APP_NAME = "jeet-mvtracker-da3-scale-audit"
DATA_ROOT = Path("/mnt/mvtracker-data")
RUN_ROOT = Path("/mnt/mvtracker-runs/da3-scale-audits")
RECIPE_ROOT = Path("/mnt/mvtracker-runs/training-recipes")
TAGS = {
    "owner": "jeet",
    "project": "mvtracker",
    "purpose": "evaluation",
    "experiment": "da3-scale-audit",
}


app = modal.App(APP_NAME, tags=TAGS)


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


def _camera_centers(extrinsics):
    import numpy as np

    values = np.asarray(extrinsics, dtype=np.float64)[..., :3, :4]
    rotation = values[..., :3]
    translation = values[..., 3]
    return -np.einsum("...ji,...j->...i", rotation, translation)


def _depth_metrics(prediction, target, mask):
    import numpy as np

    usable = (
        mask
        & np.isfinite(prediction)
        & np.isfinite(target)
        & (prediction > 0)
        & (target > 0)
    )
    pred = prediction[usable].astype(np.float64)
    gt = target[usable].astype(np.float64)
    ratio = pred / gt
    error = pred - gt
    return {
        "pixels": int(gt.size),
        "coverage": float(usable.mean()),
        "abs_rel": float(np.mean(np.abs(error) / gt)),
        "rmse_m": float(np.sqrt(np.mean(error * error))),
        "median_ratio": float(np.median(ratio)),
        "ratio_p10": float(np.quantile(ratio, 0.10)),
        "ratio_p90": float(np.quantile(ratio, 0.90)),
        "delta1": float(np.mean(np.maximum(ratio, 1.0 / ratio) < 1.25)),
    }


@app.function(
    image=da3_training_image,
    gpu="H200",
    cpu=8,
    memory=65536,
    timeout=30 * 60,
    volumes={DATA_ROOT: data_volume.with_mount_options(read_only=True), RUN_ROOT.parent: run_volume},
    secrets=[hf_secret, wandb_secret],
)
def audit(recipe_name: str, virtual_index: int = 822) -> dict:
    import numpy as np
    import torch
    import torch.nn.functional as F
    import wandb
    from depth_anything_3.api import DepthAnything3
    from depth_anything_3.utils.pose_align import align_poses_umeyama

    from mvtracker.datasets.training_recipe import RecipeReader, RecipeRecord
    from mvtracker.preprocessing.runtime_da3 import (
        IMAGE_CAPACITY,
        MODEL_ID,
        _PackedScenes,
    )

    run_name = (
        f"syn4d-v{virtual_index}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    output_root = RUN_ROOT / run_name
    output_root.mkdir(parents=True, exist_ok=False)
    run = wandb.init(
        project="mvtracker-depth-evaluation",
        name=run_name,
        tags=["modal", "syn4d", "da3-giant-1.1", "scale-audit"],
        config={**TAGS, "recipe": recipe_name, "virtual_index": virtual_index},
    )
    _log("job_started", run_name=run_name, gpu=torch.cuda.get_device_name(0))

    record = None
    recipe_path = RECIPE_ROOT / recipe_name
    for payload in RecipeReader(recipe_path).steps():
        for sample in payload["logical_samples"]:
            candidate = RecipeRecord.from_dict(sample)
            if candidate.source == "syn4d" and candidate.request["virtual_index"] == virtual_index:
                record = candidate
                break
        if record is not None:
            break
    if record is None:
        raise ValueError(f"Syn4D virtual sample {virtual_index} not found")
    _log(
        "recipe_record_found",
        step=record.step,
        scene=record.scene,
        frames=[record.frames[0], record.frames[-1]],
        views=list(record.views),
        tracks=record.track_count,
        depth_source=record.depth_source,
    )

    _log("model_load_started", model=MODEL_ID)
    started = time.perf_counter()
    model = DepthAnything3.from_pretrained(MODEL_ID).to("cuda").eval()
    torch.cuda.synchronize()
    _log(
        "model_ready",
        seconds=round(time.perf_counter() - started, 3),
        allocated_gib=round(torch.cuda.memory_allocated() / 2**30, 3),
    )

    reader = _PackedScenes(DATA_ROOT)
    load_started = time.perf_counter()
    inference_views, images, extrinsics, intrinsics, source_size = reader.load(record)
    selected_positions = [inference_views.index(view) for view in record.views]
    _log(
        "media_ready",
        seconds=round(time.perf_counter() - load_started, 3),
        inference_views=inference_views,
        timestamps=len(record.frames),
        images=len(record.frames) * len(inference_views),
        source_size=source_size,
    )

    predicted_depth_batches = []
    predicted_conf_batches = []
    frame_diagnostics = []
    timestamp_batch = min(
        len(record.frames), max(1, IMAGE_CAPACITY // len(inference_views))
    )
    for start in range(0, len(record.frames), timestamp_batch):
        stop = min(start + timestamp_batch, len(record.frames))
        _log("inference_batch_started", start=start, stop=stop)
        processed_images = []
        processed_extrinsics = []
        processed_intrinsics = []
        for timestamp in range(start, stop):
            imgs, exts, ints = model._preprocess_inputs(
                images[timestamp],
                extrinsics[timestamp],
                intrinsics[timestamp],
                process_res=504,
                process_res_method="upper_bound_resize",
            )
            processed_images.append(imgs)
            processed_extrinsics.append(exts)
            processed_intrinsics.append(ints)
        batch_images = torch.stack(processed_images).to("cuda").float()
        batch_extrinsics = torch.stack(processed_extrinsics).to("cuda").float()
        batch_intrinsics = torch.stack(processed_intrinsics).to("cuda").float()
        normalized_extrinsics = torch.cat(
            [
                model._normalize_extrinsics(
                    batch_extrinsics[index : index + 1].clone()
                )
                for index in range(stop - start)
            ]
        )
        torch.cuda.synchronize()
        inference_started = time.perf_counter()
        with torch.inference_mode():
            result = model.forward(
                batch_images,
                normalized_extrinsics,
                batch_intrinsics,
                export_feat_layers=[],
            )
        torch.cuda.synchronize()
        batch_seconds = time.perf_counter() - inference_started
        predicted_depth = result["depth"].float().cpu().numpy()
        predicted_conf = result["depth_conf"].float().cpu().numpy()
        predicted_poses = result["extrinsics"].float().cpu().numpy()
        for offset in range(stop - start):
            _, _, scale, aligned_gt = align_poses_umeyama(
                predicted_poses[offset],
                processed_extrinsics[offset].numpy(),
                ransac=False,
                return_aligned=True,
            )
            predicted_centers = _camera_centers(predicted_poses[offset])
            aligned_centers = _camera_centers(aligned_gt)
            residuals = np.linalg.norm(predicted_centers - aligned_centers, axis=-1)
            predicted_depth[offset] /= scale
            diagnostic = {
                "position": start + offset,
                "frame": int(record.frames[start + offset]),
                "scale": float(scale),
                "pose_center_rmse": float(np.sqrt(np.mean(residuals**2))),
                "pose_center_max_error": float(residuals.max()),
            }
            frame_diagnostics.append(diagnostic)
            _log("frame_aligned", **diagnostic)
        predicted_depth_batches.append(predicted_depth)
        predicted_conf_batches.append(predicted_conf)
        _log(
            "inference_batch_finished",
            start=start,
            stop=stop,
            seconds=round(batch_seconds, 3),
        )

    predicted_depth = np.concatenate(predicted_depth_batches, axis=0)
    predicted_conf = np.concatenate(predicted_conf_batches, axis=0)
    predicted_depth = F.interpolate(
        torch.from_numpy(predicted_depth.reshape(-1, *predicted_depth.shape[-2:]))[:, None],
        size=source_size,
        mode="bilinear",
        align_corners=False,
    )[:, 0].reshape(len(record.frames), len(inference_views), *source_size).numpy()
    predicted_conf = F.interpolate(
        torch.from_numpy(predicted_conf.reshape(-1, *predicted_conf.shape[-2:]))[:, None],
        size=source_size,
        mode="bilinear",
        align_corners=False,
    )[:, 0].reshape(len(record.frames), len(inference_views), *source_size).numpy()
    predicted_depth = predicted_depth[:, selected_positions]
    predicted_conf = predicted_conf[:, selected_positions]

    scene_root = DATA_ROOT / "datasets/syn4d-mvtracker/train" / record.scene
    target = np.stack(
        [
            np.asarray(
                np.load(scene_root / str(view) / "depth.npy", mmap_mode="r")[list(record.frames)]
            )
            for view in record.views
        ],
        axis=1,
    )
    valid = np.isfinite(target) & (target > 0)
    cleaned = np.zeros_like(valid)
    for position in range(len(record.frames)):
        cutoff = np.quantile(predicted_conf[position], 0.40)
        cleaned[position] = predicted_conf[position] >= cutoff
        frame_diagnostics[position]["confidence_cutoff"] = float(cutoff)
        frame_diagnostics[position]["cleaned_coverage"] = float(cleaned[position].mean())
        frame_diagnostics[position]["raw_depth"] = _depth_metrics(
            predicted_depth[position], target[position], valid[position]
        )
        frame_diagnostics[position]["cleaned_depth"] = _depth_metrics(
            predicted_depth[position],
            target[position],
            valid[position] & cleaned[position],
        )
        _log(
            "frame_compared",
            frame=frame_diagnostics[position]["frame"],
            scale=frame_diagnostics[position]["scale"],
            raw_abs_rel=frame_diagnostics[position]["raw_depth"]["abs_rel"],
            raw_median_ratio=frame_diagnostics[position]["raw_depth"]["median_ratio"],
            cleaned_abs_rel=frame_diagnostics[position]["cleaned_depth"]["abs_rel"],
            cleaned_coverage=frame_diagnostics[position]["cleaned_coverage"],
        )

    scales = np.asarray([item["scale"] for item in frame_diagnostics])
    pose_rmse = np.asarray([item["pose_center_rmse"] for item in frame_diagnostics])
    summary = {
        "scale_min": float(scales.min()),
        "scale_median": float(np.median(scales)),
        "scale_max": float(scales.max()),
        "scale_coefficient_of_variation": float(scales.std() / scales.mean()),
        "max_adjacent_scale_ratio": float(
            np.maximum(scales[1:] / scales[:-1], scales[:-1] / scales[1:]).max()
        ),
        "pose_center_rmse_mean": float(pose_rmse.mean()),
        "pose_center_rmse_max": float(pose_rmse.max()),
        "raw_depth": _depth_metrics(predicted_depth, target, valid),
        "cleaned_depth": _depth_metrics(predicted_depth, target, valid & cleaned),
        "cleaned_coverage": float(cleaned.mean()),
    }
    report = {
        "run_name": run_name,
        "recipe": recipe_name,
        "record": {
            "step": record.step,
            "virtual_index": virtual_index,
            "scene": record.scene,
            "frames": list(record.frames),
            "views": list(record.views),
            "tracks": record.track_count,
            "depth_source": record.depth_source,
        },
        "summary": summary,
        "frames": frame_diagnostics,
    }
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    table = wandb.Table(
        columns=[
            "frame",
            "scale",
            "pose_center_rmse",
            "raw_abs_rel",
            "raw_median_ratio",
            "cleaned_abs_rel",
            "cleaned_coverage",
        ]
    )
    for item in frame_diagnostics:
        table.add_data(
            item["frame"],
            item["scale"],
            item["pose_center_rmse"],
            item["raw_depth"]["abs_rel"],
            item["raw_depth"]["median_ratio"],
            item["cleaned_depth"]["abs_rel"],
            item["cleaned_coverage"],
        )
    run.log({"per_frame": table, **{f"summary/{key}": value for key, value in summary.items() if isinstance(value, float)}})
    run.summary.update({key: value for key, value in summary.items() if isinstance(value, float)})
    run.finish()
    run_volume.commit()
    _log("job_finished", output=str(output_root), summary=summary)
    return report


@app.local_entrypoint()
def main(
    recipe_name: str = "fresh-mixed-da3-r65-singleton-1000-20260825",
    virtual_index: int = 822,
) -> None:
    app.set_tags({**TAGS, "recipe": recipe_name, "virtual_index": str(virtual_index)})
    result = audit.remote(recipe_name, virtual_index)
    print(json.dumps(result["summary"], indent=2))
