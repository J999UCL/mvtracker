"""CPU-only integrity audit for the two high-loss Syn4D scenes."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import modal

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


APP_NAME = "jeet-mvtracker-syn4d-scene-audit"
DATASET_ROOT = DATA_ROOT / "datasets/syn4d-mvtracker/train"
OUTPUT_ROOT = RUN_ROOT / "syn4d-scene-audits"
TAGS = {"owner": "jeet", "project": "mvtracker", "purpose": "evaluation"}
SCENES = (
    {
        "name": "planet_bald__seq_000018",
        "window": (12, 36),
        "views": (0, 1, 2, 5, 6, 7),
        "training_step": 26,
    },
    {
        "name": "castle__seq_000007",
        "window": (8, 32),
        "views": (1,),
        "training_step": 132,
    },
)
JUMP_THRESHOLDS_METRES = (0.1, 0.25, 0.5, 1.0, 2.0)


app = modal.App(APP_NAME, tags={**TAGS, "experiment": "syn4d-scene-integrity"})
image = _source_image(_dependency_image())


def _distribution(values):
    import numpy as np

    values = np.asarray(values)
    if not values.size:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "p99": float(np.quantile(values, 0.99)),
        "p999": float(np.quantile(values, 0.999)),
        "max": float(values.max()),
    }


def _camera_matrix(intrinsics, frame):
    import numpy as np

    matrix = intrinsics if intrinsics.ndim == 2 else intrinsics[frame]
    if matrix.shape == (3, 3):
        return matrix
    if matrix.shape == (4,):
        fx, fy, cx, cy = matrix
        return np.asarray(((fx, 0, cx), (0, fy, cy), (0, 0, 1)), dtype=np.float32)
    raise ValueError(f"unsupported intrinsics shape: {intrinsics.shape}")


def _audit_depth(scene_root, tracks, valid, start, stop, views):
    import numpy as np

    reports = {}
    for view in views:
        print(f"AUDIT scene={scene_root.name} phase=depth view={view}", flush=True)
        root = scene_root / str(view)
        depth = np.load(root / "depth.npy", mmap_mode="r", allow_pickle=False)
        visibility = np.load(root / "visibility.npy", mmap_mode="r", allow_pickle=False)
        intrinsics = np.load(root / "intrinsics.npy", mmap_mode="r", allow_pickle=False)
        extrinsics = np.load(root / "extrinsics_w2c.npy", mmap_mode="r", allow_pickle=False)
        nearest_residuals = []
        neighbourhood_residuals = []
        visible_points = 0
        projected_points = 0
        depth_points = 0
        visible_but_invalid = 0
        for frame in range(start, stop):
            visible = np.asarray(visibility[frame], dtype=bool)
            frame_valid = np.asarray(valid[frame], dtype=bool)
            visible_points += int(visible.sum())
            visible_but_invalid += int((visible & ~frame_valid).sum())
            mask = visible & frame_valid & np.isfinite(tracks[frame]).all(axis=1)
            if not mask.any():
                continue
            points = np.asarray(tracks[frame, mask], dtype=np.float32)
            transform = np.asarray(extrinsics[frame], dtype=np.float32)
            transform = transform[:3, :4]
            camera = points @ transform[:, :3].T + transform[:, 3]
            z = camera[:, 2]
            k = _camera_matrix(intrinsics, frame)
            x = k[0, 0] * camera[:, 0] / z + k[0, 2]
            y = k[1, 1] * camera[:, 1] / z + k[1, 2]
            px = np.floor(x + 0.5).astype(np.int64)
            py = np.floor(y + 0.5).astype(np.int64)
            height, width = depth.shape[-2:]
            in_frame = (
                np.isfinite(x)
                & np.isfinite(y)
                & np.isfinite(z)
                & (z > 0)
                & (px >= 0)
                & (px < width)
                & (py >= 0)
                & (py < height)
            )
            projected_points += int(in_frame.sum())
            if not in_frame.any():
                continue
            px = px[in_frame]
            py = py[in_frame]
            z = z[in_frame]
            frame_depth = np.asarray(depth[frame])
            nearest = frame_depth[py, px]
            has_nearest = np.isfinite(nearest) & (nearest > 0)
            if has_nearest.any():
                nearest_residuals.append(np.abs(nearest[has_nearest] - z[has_nearest]))
                depth_points += int(has_nearest.sum())
            best = np.full(len(z), np.inf, dtype=np.float32)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nx = px + dx
                    ny = py + dy
                    inside = (nx >= 0) & (nx < width) & (ny >= 0) & (ny < height)
                    if not inside.any():
                        continue
                    sampled = np.zeros(len(z), dtype=np.float32)
                    sampled[inside] = frame_depth[ny[inside], nx[inside]]
                    usable = inside & np.isfinite(sampled) & (sampled > 0)
                    best[usable] = np.minimum(best[usable], np.abs(sampled[usable] - z[usable]))
            usable = np.isfinite(best)
            if usable.any():
                neighbourhood_residuals.append(best[usable])
        nearest = np.concatenate(nearest_residuals) if nearest_residuals else np.empty(0)
        neighbourhood = (
            np.concatenate(neighbourhood_residuals)
            if neighbourhood_residuals
            else np.empty(0)
        )
        reports[str(view)] = {
            "visible_points": visible_points,
            "visible_but_invalid": visible_but_invalid,
            "projected_in_frame": projected_points,
            "positive_nearest_depth": depth_points,
            "nearest_depth_abs_error_metres": _distribution(nearest),
            "best_3x3_depth_abs_error_metres": _distribution(neighbourhood),
            "best_3x3_over_0.05m_fraction": (
                float((neighbourhood > 0.05).mean()) if neighbourhood.size else None
            ),
            "best_3x3_over_0.10m_fraction": (
                float((neighbourhood > 0.10).mean()) if neighbourhood.size else None
            ),
            "best_3x3_over_0.25m_fraction": (
                float((neighbourhood > 0.25).mean()) if neighbourhood.size else None
            ),
        }
    return reports


def _audit_scene(scene_root: Path, specification, output_root: Path):
    import numpy as np

    print(f"AUDIT scene={scene_root.name} phase=load", flush=True)
    manifest = json.loads((scene_root / "manifest.json").read_text(encoding="utf-8"))
    tracks = np.load(scene_root / "tracks_xyz.npy", mmap_mode="r", allow_pickle=False)
    valid = np.load(scene_root / "track_valid.npy", mmap_mode="r", allow_pickle=False)
    views = int(manifest["views"])
    any_visible = np.zeros(valid.shape, dtype=bool)
    for view in range(views):
        any_visible |= np.load(
            scene_root / str(view) / "visibility.npy",
            mmap_mode="r",
            allow_pickle=False,
        )

    print(f"AUDIT scene={scene_root.name} phase=displacement", flush=True)
    displacement = np.linalg.norm(np.diff(tracks, axis=0), axis=-1)
    pair_valid = np.asarray(valid[:-1], dtype=bool) & np.asarray(valid[1:], dtype=bool)
    pair_visible = any_visible[:-1] & any_visible[1:]
    finite = np.isfinite(displacement)
    usable = pair_valid & finite
    visible_usable = usable & pair_visible
    report = {
        "scene": scene_root.name,
        "training_step": int(specification["training_step"]),
        "frames": int(tracks.shape[0]),
        "tracks": int(tracks.shape[1]),
        "fps": float(manifest["fps"]),
        "spike_window": list(specification["window"]),
        "spike_views": list(specification["views"]),
        "valid_displacement_metres": _distribution(displacement[usable]),
        "visible_displacement_metres": _distribution(displacement[visible_usable]),
        "jump_thresholds": {},
    }
    for threshold in JUMP_THRESHOLDS_METRES:
        jumps = usable & (displacement > threshold)
        report["jump_thresholds"][str(threshold)] = {
            "frame_track_pairs": int(jumps.sum()),
            "unique_tracks": int(np.any(jumps, axis=0).sum()),
            "visible_both_ends": int((jumps & pair_visible).sum()),
            "visible_both_fraction": float((jumps & pair_visible).sum() / max(1, jumps.sum())),
        }

    start, stop = (int(value) for value in specification["window"])
    window_slice = slice(start, stop - 1)
    window_usable = usable[window_slice]
    window_visible = pair_visible[window_slice]
    window_displacement = displacement[window_slice]
    report["spike_window_displacement_metres"] = _distribution(
        window_displacement[window_usable]
    )
    report["spike_window_jump_thresholds"] = {}
    for threshold in JUMP_THRESHOLDS_METRES:
        jumps = window_usable & (window_displacement > threshold)
        report["spike_window_jump_thresholds"][str(threshold)] = {
            "frame_track_pairs": int(jumps.sum()),
            "unique_tracks": int(np.any(jumps, axis=0).sum()),
            "visible_both_ends": int((jumps & window_visible).sum()),
        }

    threshold = 0.5
    jump_mask = usable & (displacement > threshold)
    frame_counts = jump_mask.sum(axis=1)
    frame_max = np.where(usable, displacement, -np.inf).max(axis=1)
    top_frames = np.argsort(frame_counts)[-10:][::-1]
    report["top_jump_frames_over_0.5m"] = [
        {
            "frame_from": int(frame),
            "frame_to": int(frame + 1),
            "jumping_tracks": int(frame_counts[frame]),
            "visible_both_ends": int((jump_mask[frame] & pair_visible[frame]).sum()),
            "max_jump_metres": float(frame_max[frame]),
        }
        for frame in top_frames
        if frame_counts[frame] > 0
    ]

    flat = np.where(usable, displacement, -np.inf).reshape(-1)
    top_count = min(100, int(usable.sum()))
    top_indices = np.argpartition(flat, -top_count)[-top_count:]
    top_indices = top_indices[np.argsort(flat[top_indices])[::-1]]
    top_jumps = []
    for index in top_indices:
        frame, track = np.unravel_index(index, displacement.shape)
        top_jumps.append(
            {
                "frame_from": int(frame),
                "frame_to": int(frame + 1),
                "track": int(track),
                "distance_metres": float(displacement[frame, track]),
                "visible_before": bool(any_visible[frame, track]),
                "visible_after": bool(any_visible[frame + 1, track]),
            }
        )

    report["depth_consistency"] = _audit_depth(
        scene_root,
        tracks,
        valid,
        int(start),
        int(stop),
        specification["views"],
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "top_jumps.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=top_jumps[0].keys())
        writer.writeheader()
        writer.writerows(top_jumps)
    print(
        "AUDIT scene=%s phase=complete max_jump_m=%.3f jumps_over_0.5m=%d"
        % (
            scene_root.name,
            report["valid_displacement_metres"]["max"],
            report["jump_thresholds"]["0.5"]["frame_track_pairs"],
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

    output_root = OUTPUT_ROOT / run_name
    run = wandb.init(
        entity="jeetucl-ucl",
        project="mvtracker-continual-training",
        group="syn4d-data-audit",
        job_type="data-integrity-audit",
        name=run_name,
        tags=["syn4d", "cpu", "teleport", "depth-consistency"],
        config={"source_commit": _source_commit(), **TAGS},
    )
    reports = {}
    for specification in SCENES:
        name = specification["name"]
        reports[name] = _audit_scene(
            DATASET_ROOT / name,
            specification,
            output_root / name,
        )
        summary = reports[name]
        run.summary[f"{name}/max_jump_metres"] = summary[
            "valid_displacement_metres"
        ]["max"]
        run.summary[f"{name}/jumps_over_0.5m"] = summary["jump_thresholds"][
            "0.5"
        ]["frame_track_pairs"]
    (output_root / "report.json").write_text(
        json.dumps(reports, indent=2) + "\n", encoding="utf-8"
    )
    run_volume.commit()
    run.summary["output_root"] = str(output_root)
    run.finish()
    return {"run_name": run_name, "output_root": str(output_root), "reports": reports}


@app.local_entrypoint()
def main(run_name: str = ""):
    selected = run_name or (
        "planet-castle-integrity-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    app.set_tags({**TAGS, "experiment": selected, "gpu": "cpu"})
    print(json.dumps(audit_remote.remote(selected), indent=2))
