"""Correlate Syn4D sample motion diagnostics with recorded training loss."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import modal

from modal_syn4d_track_overlay import _training_dataset
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


APP_NAME = "jeet-mvtracker-syn4d-motion-loss-audit"
TRAINING_RUN = (
    RUN_ROOT
    / "continual-training"
    / "gt-replay-syn4d-envsplit-v2-ddp2-h200-20260822T113334Z"
)
OUTPUT_ROOT = RUN_ROOT / "syn4d-motion-loss-audits"
TAGS = {"owner": "jeet", "project": "mvtracker", "purpose": "evaluation"}
SCENES = (
    "planet_bald__seq_000018",
    "castle__seq_000007",
    "cave_group__seq_000008",
    "desert_bald__seq_000012",
)
METRICS = (
    "world_velocity_p90_mps",
    "world_acceleration_p90_mps2",
    "world_jerk_p90_mps3",
    "pixel_displacement_p90_px",
    "visibility_transition_rate",
    "visible_fraction",
    "unique_track_fraction",
)
LOSSES = ("total_loss", "trajectory_loss", "visibility_loss")


app = modal.App(APP_NAME, tags={**TAGS, "experiment": "syn4d-motion-loss"})
image = _source_image(_dependency_image())


def _quantile(values, percentile: float) -> float:
    import numpy as np

    values = np.asarray(values)
    return float(np.quantile(values, percentile)) if values.size else float("nan")


def _records() -> dict[str, list[dict[str, object]]]:
    grouped = {scene: [] for scene in SCENES}
    path = TRAINING_RUN / "per_scene_losses.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            for record in json.loads(line)["top_scenes"]:
                if record["scene"] in grouped:
                    grouped[record["scene"]].append(record)
    return grouped


def _sample_metrics(dataset, scene: str, records: list[dict[str, object]]):
    import numpy as np

    from mvtracker.datasets.mixed_source_schedule import ScheduledSampleRequest

    scene_index = dataset.seq_names.index(scene)
    root = Path(dataset.data_root) / scene
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    fps = float(manifest["fps"])
    tracks = np.load(root / "tracks_xyz.npy", mmap_mode="r")
    validity = np.load(root / "track_valid.npy", mmap_mode="r")
    rows = []
    for position, record in enumerate(records, 1):
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

        start, stop = expected[:2]
        track_ids = np.asarray(plan.selected_global_track_indices, dtype=np.int64)
        xyz = np.asarray(tracks[start:stop, track_ids], dtype=np.float32)
        valid = np.asarray(validity[start:stop, track_ids], dtype=bool)
        visible = np.asarray(plan.visibility, dtype=bool)
        visible_any = visible.any(axis=0)

        velocity = np.diff(xyz, axis=0)
        velocity_mask = valid[:-1] & valid[1:] & visible_any[:-1] & visible_any[1:]
        velocity_norm = np.linalg.norm(velocity, axis=-1)[velocity_mask] * fps

        acceleration = np.diff(velocity, axis=0)
        acceleration_mask = (
            valid[:-2]
            & valid[1:-1]
            & valid[2:]
            & visible_any[:-2]
            & visible_any[1:-1]
            & visible_any[2:]
        )
        acceleration_norm = (
            np.linalg.norm(acceleration, axis=-1)[acceleration_mask] * fps**2
        )

        jerk = np.diff(acceleration, axis=0)
        jerk_mask = (
            valid[:-3]
            & valid[1:-2]
            & valid[2:-1]
            & valid[3:]
            & visible_any[:-3]
            & visible_any[1:-2]
            & visible_any[2:-1]
            & visible_any[3:]
        )
        jerk_norm = np.linalg.norm(jerk, axis=-1)[jerk_mask] * fps**3

        pixels = np.asarray(plan.trajectory[..., :2], dtype=np.float32)
        pixel_motion = np.linalg.norm(np.diff(pixels, axis=1), axis=-1)
        pixel_mask = visible[:, :-1] & visible[:, 1:]
        pixel_motion = pixel_motion[pixel_mask]

        valid_pairs = valid[:-1] & valid[1:]
        transitions = (visible_any[:-1] != visible_any[1:]) & valid_pairs
        row = {
            **record,
            "unique_tracks": int(np.unique(track_ids).size),
            "unique_track_fraction": float(np.unique(track_ids).size / len(track_ids)),
            "visible_fraction": float(visible_any[valid].mean()),
            "visibility_transition_rate": float(
                transitions.sum() / max(1, valid_pairs.sum())
            ),
            "world_velocity_p50_mps": _quantile(velocity_norm, 0.50),
            "world_velocity_p90_mps": _quantile(velocity_norm, 0.90),
            "world_velocity_p99_mps": _quantile(velocity_norm, 0.99),
            "world_acceleration_p50_mps2": _quantile(acceleration_norm, 0.50),
            "world_acceleration_p90_mps2": _quantile(acceleration_norm, 0.90),
            "world_acceleration_p99_mps2": _quantile(acceleration_norm, 0.99),
            "world_jerk_p50_mps3": _quantile(jerk_norm, 0.50),
            "world_jerk_p90_mps3": _quantile(jerk_norm, 0.90),
            "world_jerk_p99_mps3": _quantile(jerk_norm, 0.99),
            "pixel_displacement_p50_px": _quantile(pixel_motion, 0.50),
            "pixel_displacement_p90_px": _quantile(pixel_motion, 0.90),
            "pixel_displacement_p99_px": _quantile(pixel_motion, 0.99),
        }
        rows.append(row)
        if position % 25 == 0 or position == len(records):
            print(f"MOTION scene={scene} samples={position}/{len(records)}", flush=True)
    return rows


def _correlations(rows):
    import numpy as np
    from scipy.stats import spearmanr

    report = {}
    groups = {scene: [row for row in rows if row["scene"] == scene] for scene in SCENES}
    groups["all_scenes"] = rows
    for group, records in groups.items():
        report[group] = {}
        for loss in LOSSES:
            report[group][loss] = {}
            loss_values = np.asarray([record[loss] for record in records], dtype=np.float64)
            for metric in METRICS:
                metric_values = np.asarray(
                    [record[metric] for record in records], dtype=np.float64
                )
                finite = np.isfinite(loss_values) & np.isfinite(metric_values)
                correlation, pvalue = spearmanr(
                    metric_values[finite], loss_values[finite]
                )
                report[group][loss][metric] = {
                    "rho": float(correlation),
                    "pvalue": float(pvalue),
                    "samples": int(finite.sum()),
                }
    return report


def _scene_summary(rows):
    import numpy as np

    report = {}
    for scene in SCENES:
        records = [row for row in rows if row["scene"] == scene]
        report[scene] = {"samples": len(records)}
        for field in (*LOSSES, *METRICS):
            values = np.asarray([row[field] for row in records], dtype=np.float64)
            report[scene][field] = {
                "median": float(np.nanmedian(values)),
                "p90": float(np.nanquantile(values, 0.90)),
            }
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
        group="syn4d-data-audit",
        job_type="motion-loss-audit",
        name=run_name,
        tags=["syn4d", "cpu", "motion", "loss-correlation"],
        config={"source_commit": _source_commit(), **TAGS},
    )
    dataset = _training_dataset()
    records = _records()
    with ThreadPoolExecutor(max_workers=len(SCENES)) as executor:
        futures = {
            scene: executor.submit(_sample_metrics, dataset, scene, records[scene])
            for scene in SCENES
        }
        rows = [row for scene in SCENES for row in futures[scene].result()]
    report = {
        "scene_summary": _scene_summary(rows),
        "correlations": _correlations(rows),
        "selection_note": "per_scene_losses.jsonl contains the top four global scenes per optimizer step",
    }
    output_root = OUTPUT_ROOT / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    run_volume.commit()
    run.summary["output_root"] = str(output_root)
    for scene, summary in report["scene_summary"].items():
        run.summary[f"{scene}/trajectory_loss_median"] = summary["trajectory_loss"]["median"]
        run.summary[f"{scene}/acceleration_p90_median"] = summary[
            "world_acceleration_p90_mps2"
        ]["median"]
    run.finish()
    return {"output_root": str(output_root), **report}


@app.local_entrypoint()
def main(run_name: str = ""):
    selected = run_name or (
        "syn4d-motion-loss-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    app.set_tags({**TAGS, "experiment": selected, "gpu": "cpu"})
    print(json.dumps(audit_remote.remote(selected), indent=2))
