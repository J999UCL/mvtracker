"""CPU audit of DIEGESIS training scenes against recorded scene loss."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path

import modal

from modal_syn4d_all_scene_audit import _camera_centres, _correlations, _quantiles
from modal_syn4d_track_overlay import _jpeg_frame
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


APP_NAME = "jeet-mvtracker-diegesis-scene-loss-audit"
TRAINING_RUN = (
    RUN_ROOT
    / "continual-training"
    / "gt-replay-syn4d-envsplit-v2-ddp2-h200-20260822T113334Z"
)
OUTPUT_ROOT = RUN_ROOT / "diegesis-scene-loss-audits"
TAGS = {"owner": "jeet", "project": "mvtracker", "purpose": "evaluation"}
SAMPLES_PER_SCENE = 50


app = modal.App(APP_NAME, tags={**TAGS, "experiment": "diegesis-scene-loss-audit"})
image = _source_image(_dependency_image())


def _training_dataset():
    from omegaconf import OmegaConf

    from mvtracker.datasets.tapvid3d_multiview_dataset import TapVid3DMultiViewDataset

    source_root = Path("/opt/mvtracker")
    cfg = OmegaConf.merge(
        OmegaConf.load(source_root / "configs/train.yaml"),
        OmegaConf.load(
            source_root / "configs/experiment/diegesis_syn4d_mvkubric_gt_ddp.yaml"
        ),
    )
    cfg.datasets.root = str(DATA_ROOT / "datasets")
    cfg.reproducibility.seed = 3479143162

    class Fabric:
        world_size = 2

    source_cfg = cfg.datasets.train.sources.diegesis
    kwargs = TapVid3DMultiViewDataset.from_name(
        source_cfg.name,
        source_cfg.root,
        cfg,
        Fabric(),
        just_return_kwargs=True,
        include_scene_ids=source_cfg.include_scene_ids,
    )
    kwargs["view_count_probabilities"] = tuple(source_cfg.view_count_probabilities)
    return TapVid3DMultiViewDataset(**kwargs)


def _loss_records():
    grouped = {}
    with (TRAINING_RUN / "per_scene_losses.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            for record in json.loads(line)["top_scenes"]:
                if record["source"] == "diegesis":
                    grouped.setdefault(record["scene"], []).append(record)
    return grouped


def _scene_data_statistics(dataset, scene: str):
    import cv2
    import numpy as np

    manifest = dataset._manifest(scene)
    cache_root = Path(dataset.data_root) / scene
    raw_root = Path(dataset.raw_root) / manifest["source_sequence"]
    frames = int(manifest["frame_count"])
    views = list(manifest["views"])
    frame_indices = np.linspace(0, frames - 1, min(8, frames), dtype=np.int64)
    extrinsics = np.stack(
        [np.load(raw_root / str(view) / "extrinsics_w2c.npy", mmap_mode="r") for view in views]
    )
    centres = _camera_centres(extrinsics)
    anchor = centres[:, 0].mean(axis=0)
    rig_radius = np.linalg.norm(centres[:, 0] - anchor, axis=-1)
    camera_paths = np.linalg.norm(np.diff(centres, axis=1), axis=-1).sum(axis=1)

    tracks = np.load(raw_root / "tracks_xyz.npy", mmap_mode="r")
    finite = np.isfinite(tracks).all(axis=-1)
    sampled_tracks = np.asarray(tracks[frame_indices, ::8], dtype=np.float32)
    sampled_finite = np.asarray(finite[frame_indices, ::8], dtype=bool)
    centred_norms = np.linalg.norm(sampled_tracks - anchor, axis=-1)[sampled_finite]
    all_visibility = np.stack(
        [np.load(raw_root / str(view) / "visibility.npy", mmap_mode="r") for view in views]
    )
    from mvtracker.datasets.tapvid3d_multiview_dataset import _visible_path_lengths

    movement = _visible_path_lengths(np.asarray(tracks), np.asarray(all_visibility))
    depths = []
    luminance = []
    contrast = []
    texture = []
    saturated = []
    foreground = []
    handles = {}
    offsets = {
        view: np.load(cache_root / f"view_{view}" / "jpeg_offsets.npy")
        for view in views
    }
    try:
        for view in views:
            depth = np.load(raw_root / str(view) / "depth.npy", mmap_mode="r")
            mask = np.load(raw_root / str(view) / "foreground_mask.npy", mmap_mode="r")
            for frame in frame_indices[::2]:
                frame_depth = np.asarray(depth[frame], dtype=np.float32)
                usable = frame_depth[np.isfinite(frame_depth) & (frame_depth > 0)]
                if usable.size:
                    depths.append(usable[::16])
                foreground.append(float(np.asarray(mask[frame], dtype=bool).mean()))
                image = _jpeg_frame(cache_root, view, int(frame), offsets, handles)
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
    depth_quantiles = _quantiles(depth_values)
    return {
        "frames": frames,
        "points": int(tracks.shape[1]),
        "anchor_norm_m": float(np.linalg.norm(anchor)),
        "camera_rig_radius_p90_m": float(np.quantile(rig_radius, 0.90)),
        "camera_path_mean_m": float(camera_paths.mean()),
        "centred_track_radius_p90_m": float(np.quantile(centred_norms, 0.90)),
        "valid_track_fraction": float(finite.mean()),
        "full_static_fraction": float((movement < 0.01).mean()),
        "full_dynamic_fraction": float((movement > 0.1).mean()),
        "full_very_dynamic_fraction": float((movement > 2.0).mean()),
        "depth_p50_m": depth_quantiles["p50"],
        "depth_p90_m": depth_quantiles["p90"],
        "depth_p99_m": depth_quantiles["p99"],
        "foreground_pixel_fraction": float(np.mean(foreground)),
        "rgb_luminance_mean": float(np.mean(luminance)),
        "rgb_contrast_mean": float(np.mean(contrast)),
        "rgb_texture_mean": float(np.mean(texture)),
        "rgb_saturated_fraction": float(np.mean(saturated)),
    }


def _sample_statistics(dataset, scene: str, records, scene_index: int):
    import numpy as np

    from mvtracker.datasets.mixed_source_schedule import ScheduledSampleRequest
    from mvtracker.datasets.tapvid3d_multiview_dataset import _visible_path_lengths

    positions = np.linspace(
        0, len(records) - 1, min(SAMPLES_PER_SCENE, len(records)), dtype=np.int64
    )
    raw_root = Path(dataset.raw_root) / dataset._manifest(scene)["source_sequence"]
    tracks = np.load(raw_root / "tracks_xyz.npy", mmap_mode="r")
    values = []
    for position in positions:
        record = records[position]
        plan = dataset.plan_sample(
            ScheduledSampleRequest(
                virtual_index=int(record["virtual_index"]),
                scene_index=scene_index,
                view_count=None,
            )
        )
        observed = (
            int(plan.frame_indices[0]),
            int(plan.frame_indices[-1] + 1),
            tuple(plan.views),
            int(plan.track_count),
        )
        expected = (
            int(record["window_start"]),
            int(record["window_end_exclusive"]),
            tuple(record["selected_views"]),
            int(record["tracks"]),
        )
        if observed != expected:
            raise RuntimeError(f"{scene} sample mismatch: {observed} != {expected}")
        selected = np.asarray(plan.selected_global_track_indices, dtype=np.int64)
        selected_views = np.stack(
            [
                np.load(raw_root / str(view) / "visibility.npy", mmap_mode="r")[:, selected]
                for view in plan.views
            ]
        )
        full_movement = _visible_path_lengths(
            np.asarray(tracks[:, selected]), selected_views
        )
        window_movement = _visible_path_lengths(
            np.asarray(tracks[plan.frame_indices][:, selected]),
            np.asarray(plan.visibility),
        )
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

    losses = [float(record["total_loss"]) for record in records]
    trajectory = [float(record["trajectory_loss"]) for record in records]
    visibility = [float(record["visibility_loss"]) for record in records]
    report = {
        "scene": scene,
        "top4_record_count": len(records),
        "recorded_total_loss_median": float(np.median(losses)),
        "recorded_total_loss_p90": float(np.quantile(losses, 0.90)),
        "recorded_trajectory_loss_median": float(np.median(trajectory)),
        "recorded_visibility_loss_median": float(np.median(visibility)),
        **_scene_data_statistics(dataset, scene),
        **_sample_statistics(dataset, scene, records, dataset.seq_names.index(scene)),
    }
    print(
        "DIEGESIS scene=%s loss=%.3f trajectory=%.3f camera_path=%.2fm foreground=%.3f"
        % (
            scene,
            report["recorded_total_loss_median"],
            report["recorded_trajectory_loss_median"],
            report["camera_path_mean_m"],
            report["foreground_pixel_fraction"],
        ),
        flush=True,
    )
    return report


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
        group="diegesis-data-audit",
        job_type="scene-loss-audit",
        name=run_name,
        tags=["diegesis", "cpu", "scene-loss", "data-audit"],
        config={"source_commit": _source_commit(), **TAGS},
    )
    dataset = _training_dataset()
    records = _loss_records()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            scene: executor.submit(_audit_scene, dataset, scene, records[scene])
            for scene in dataset.seq_names
        }
        reports = [futures[scene].result() for scene in dataset.seq_names]
    result = {
        "scenes": sorted(
            reports,
            key=lambda report: report["recorded_total_loss_median"],
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
        "diegesis-scene-loss-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    app.set_tags({**TAGS, "experiment": selected, "gpu": "cpu"})
    print(json.dumps(audit_remote.remote(selected), indent=2))
