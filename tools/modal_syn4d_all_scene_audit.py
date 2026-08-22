"""CPU audit of all Syn4D training scenes against recorded scene loss."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

import modal

from modal_syn4d_track_overlay import _jpeg_frame, _training_dataset
from modal_training_profile import (
    DATA_ROOT,
    RUN_ROOT,
    _dependency_image,
    _source_commit,
    _source_image,
    data_volume,
    run_volume,
    wandb_secret,
)


APP_NAME = "jeet-mvtracker-syn4d-all-scene-audit"
TRAINING_RUN = (
    RUN_ROOT
    / "continual-training"
    / "gt-replay-syn4d-envsplit-v2-ddp2-h200-20260822T113334Z"
)
OUTPUT_ROOT = RUN_ROOT / "syn4d-all-scene-audits"
TAGS = {"owner": "jeet", "project": "mvtracker", "purpose": "evaluation"}
SAMPLES_PER_SCENE = 50


app = modal.App(APP_NAME, tags={**TAGS, "experiment": "syn4d-all-scene-audit"})
image = _source_image(_dependency_image())


def _quantiles(values):
    import numpy as np

    values = np.asarray(values)
    return {
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
    }


def _camera_centres(extrinsics):
    import numpy as np

    rotations = extrinsics[..., :3, :3]
    translations = extrinsics[..., :3, 3]
    return -np.einsum("vtji,vtj->vti", rotations, translations)


def _loss_records():
    grouped = {}
    with (TRAINING_RUN / "per_scene_losses.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            for record in json.loads(line)["top_scenes"]:
                if record["source"] == "syn4d":
                    grouped.setdefault(record["scene"], []).append(record)
    return grouped


def _scene_data_statistics(root: Path, manifest):
    import cv2
    import numpy as np

    frames = int(manifest["frames"])
    views = int(manifest["views"])
    frame_indices = np.linspace(0, frames - 1, min(8, frames), dtype=np.int64)
    extrinsics = np.stack(
        [np.load(root / str(view) / "extrinsics_w2c.npy", mmap_mode="r") for view in range(views)]
    )
    centres = _camera_centres(extrinsics)
    anchor = centres[:, 0].mean(axis=0)
    rig_radius = np.linalg.norm(centres[:, 0] - anchor, axis=-1)
    camera_paths = np.linalg.norm(np.diff(centres, axis=1), axis=-1).sum(axis=1)

    tracks = np.load(root / "tracks_xyz.npy", mmap_mode="r")
    valid = np.load(root / "track_valid.npy", mmap_mode="r")
    sampled_tracks = np.asarray(tracks[frame_indices, ::8], dtype=np.float32)
    sampled_valid = np.asarray(valid[frame_indices, ::8], dtype=bool)
    centred_norms = np.linalg.norm(sampled_tracks - anchor, axis=-1)[sampled_valid]
    movement = np.load(root / "motion_path_length.npy", mmap_mode="r")

    depths = []
    luminance = []
    contrast = []
    texture = []
    saturated = []
    handles = {}
    offsets = {
        view: np.load(root / f"view_{view}" / "jpeg_offsets.npy")
        for view in range(views)
    }
    try:
        for view in range(views):
            depth = np.load(root / str(view) / "depth.npy", mmap_mode="r")
            for frame in frame_indices[::2]:
                frame_depth = np.asarray(depth[frame], dtype=np.float32)
                usable = frame_depth[np.isfinite(frame_depth) & (frame_depth > 0)]
                if usable.size:
                    depths.append(usable[::16])
                image = _jpeg_frame(root, view, int(frame), offsets, handles)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
                luminance.append(float(gray.mean()))
                contrast.append(float(gray.std()))
                texture.append(
                    float(
                        np.hypot(
                            cv2.Sobel(gray, cv2.CV_32F, 1, 0),
                            cv2.Sobel(gray, cv2.CV_32F, 0, 1),
                        ).mean()
                    )
                )
                saturated.append(float(((gray < 5) | (gray > 250)).mean()))
    finally:
        for handle in handles.values():
            handle.close()
    depth_values = np.concatenate(depths)
    return {
        "frames": frames,
        "duration_seconds": float(frames / manifest["fps"]),
        "anchor_x": float(anchor[0]),
        "anchor_y": float(anchor[1]),
        "anchor_z": float(anchor[2]),
        "anchor_norm_m": float(np.linalg.norm(anchor)),
        "camera_rig_radius_p90_m": float(np.quantile(rig_radius, 0.90)),
        "camera_path_mean_m": float(camera_paths.mean()),
        "centred_track_radius_p90_m": float(np.quantile(centred_norms, 0.90)),
        "valid_track_fraction": float(np.asarray(valid).mean()),
        "full_static_fraction": float((movement < 0.01).mean()),
        "full_dynamic_fraction": float((movement > 0.1).mean()),
        "full_very_dynamic_fraction": float((movement > 2.0).mean()),
        "depth_p50_m": _quantiles(depth_values)["p50"],
        "depth_p90_m": _quantiles(depth_values)["p90"],
        "depth_p99_m": _quantiles(depth_values)["p99"],
        "rgb_luminance_mean": float(np.mean(luminance)),
        "rgb_contrast_mean": float(np.mean(contrast)),
        "rgb_texture_mean": float(np.mean(texture)),
        "rgb_saturated_fraction": float(np.mean(saturated)),
    }


def _sample_statistics(dataset, scene: str, records, scene_index: int):
    import numpy as np

    from mvtracker.datasets.mixed_source_schedule import ScheduledSampleRequest
    from mvtracker.datasets.tapvid3d_multiview_dataset import _visible_path_lengths

    generated_requests = not records
    if records:
        positions = np.linspace(
            0, len(records) - 1, min(SAMPLES_PER_SCENE, len(records)), dtype=np.int64
        )
        requests = [
            ScheduledSampleRequest(
                virtual_index=int(records[position]["virtual_index"]),
                scene_index=scene_index,
                view_count=None,
            )
            for position in positions
        ]
    else:
        requests = []
    movement = np.load(
        Path(dataset.data_root) / scene / "motion_path_length.npy", mmap_mode="r"
    )
    tracks = np.load(
        Path(dataset.data_root) / scene / "tracks_xyz.npy", mmap_mode="r"
    )
    values = []
    candidate = 0
    while len(values) < SAMPLES_PER_SCENE:
        if generated_requests:
            request = ScheduledSampleRequest(
                virtual_index=scene_index + candidate * len(dataset.seq_names),
                scene_index=scene_index,
                view_count=None,
            )
            candidate += 1
        else:
            if len(values) == len(requests):
                break
            request = requests[len(values)]
        plan = dataset.plan_sample(request)
        if plan is None:
            if generated_requests:
                continue
            raise RuntimeError(f"recorded accepted sample was rejected: {scene} {request}")
        selected = np.asarray(plan.selected_global_track_indices, dtype=np.int64)
        window_movement = _visible_path_lengths(
            np.asarray(tracks[plan.frame_indices][:, selected]),
            np.asarray(plan.visibility),
        )
        full_movement = np.asarray(movement[selected])
        values.append(
            {
                "views": len(plan.views),
                "tracks": plan.track_count,
                "unique_track_fraction": np.unique(selected).size / len(selected),
                "visible_fraction": np.asarray(plan.visibility).any(axis=0).mean(),
                "window_path_mean_m": window_movement.mean(),
                "window_static_fraction": (window_movement < 0.01).mean(),
                "full_dynamic_window_static_fraction": (
                    (full_movement > 0.1) & (window_movement < 0.01)
                ).mean(),
                "full_very_dynamic_window_static_fraction": (
                    (full_movement > 2.0) & (window_movement < 0.01)
                ).mean(),
            }
        )
    return {
        name: float(np.mean([value[name] for value in values]))
        for name in values[0]
    }


def _audit_scene(dataset, scene: str, records):
    import numpy as np

    root = Path(dataset.data_root) / scene
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    losses = [float(record["total_loss"]) for record in records]
    trajectory = [float(record["trajectory_loss"]) for record in records]
    visibility = [float(record["visibility_loss"]) for record in records]
    report = {
        "scene": scene,
        "top4_record_count": len(records),
        "recorded_total_loss_median": float(np.median(losses)) if losses else None,
        "recorded_total_loss_p90": float(np.quantile(losses, 0.90)) if losses else None,
        "recorded_trajectory_loss_median": (
            float(np.median(trajectory)) if trajectory else None
        ),
        "recorded_visibility_loss_median": (
            float(np.median(visibility)) if visibility else None
        ),
        **_scene_data_statistics(root, manifest),
        **_sample_statistics(dataset, scene, records, dataset.seq_names.index(scene)),
    }
    print(
        "SCENE scene=%s anchor=%.2fm loss=%s mismatch=%.3f"
        % (
            scene,
            report["anchor_norm_m"],
            (
                f"{report['recorded_total_loss_median']:.3f}"
                if report["recorded_total_loss_median"] is not None
                else "not-top4"
            ),
            report["full_dynamic_window_static_fraction"],
        ),
        flush=True,
    )
    return report


def _correlations(reports):
    import numpy as np
    from scipy.stats import spearmanr

    usable = [report for report in reports if report["recorded_total_loss_median"] is not None]
    losses = np.asarray([report["recorded_total_loss_median"] for report in usable])
    ignored = {
        "scene",
        "recorded_total_loss_median",
        "recorded_total_loss_p90",
        "recorded_trajectory_loss_median",
        "recorded_visibility_loss_median",
    }
    result = {}
    for name in usable[0]:
        if name in ignored:
            continue
        values = np.asarray([report[name] for report in usable], dtype=np.float64)
        rho, pvalue = spearmanr(values, losses)
        result[name] = {"rho": float(rho), "pvalue": float(pvalue)}
    return result


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
        str(RUN_ROOT): run_volume,
    },
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    max_containers=1,
    retries=0,
    include_source=False,
)
def audit_remote(run_name: str):
    import wandb

    run = wandb.init(
        entity="jeetucl-ucl",
        project="mvtracker-continual-training",
        group="syn4d-data-audit",
        job_type="all-scene-audit",
        name=run_name,
        tags=["syn4d", "cpu", "scene-loss", "data-audit"],
        config={"source_commit": _source_commit(), **TAGS},
    )
    dataset = _training_dataset()
    records = _loss_records()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            scene: executor.submit(_audit_scene, dataset, scene, records.get(scene, []))
            for scene in dataset.seq_names
        }
        reports = [futures[scene].result() for scene in dataset.seq_names]
    result = {
        "scenes": sorted(
            reports,
            key=lambda report: report["recorded_total_loss_median"] or -1,
            reverse=True,
        ),
        "scene_loss_correlations": _correlations(reports),
        "loss_censoring_note": "the old run retained only the global top four scenes per optimizer step",
    }
    output_root = OUTPUT_ROOT / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "report.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    run_volume.commit()
    run.summary["output_root"] = str(output_root)
    run.finish()
    return {"output_root": str(output_root), **result}


@app.local_entrypoint()
def main(run_name: str = ""):
    selected = run_name or (
        "syn4d-all-scene-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    app.set_tags({**TAGS, "experiment": selected, "gpu": "cpu"})
    print(json.dumps(audit_remote.remote(selected), indent=2))
