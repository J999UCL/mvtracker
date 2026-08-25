"""Audit camera-centred Syn4D track radii in a prepared training recipe."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import modal


APP_NAME = "jeet-mvtracker-recipe-radius-audit"
DATA_ROOT = Path("/mnt/mvtracker-data")
RUN_ROOT = Path("/mnt/mvtracker-runs")
THRESHOLDS = (24.0, 32.0, 40.0, 50.0, 65.0)
TAGS = {"owner": "jeet", "project": "mvtracker", "purpose": "evaluation"}

app = modal.App(APP_NAME, tags=TAGS)
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "numpy==2.2.6",
    "wandb==0.19.9",
)
data_volume = modal.Volume.from_name("jeet-mvtracker-data-v2", version=2)
run_volume = modal.Volume.from_name("jeet-mvtracker-runs-v2", version=2)
wandb_secret = modal.Secret.from_name(
    "jeet-mvtracker-wandb", required_keys=["WANDB_API_KEY"]
)


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
        str(RUN_ROOT): run_volume.with_mount_options(read_only=True),
    },
    cpu=16,
    memory=32768,
    timeout=2 * 60 * 60,
    max_containers=1,
    retries=0,
)
def audit_recipe(recipe_name: str) -> dict:
    import numpy as np
    import wandb

    recipe = RUN_ROOT / "training-recipes" / recipe_name / "steps.jsonl"
    by_scene: dict[str, list[dict]] = {}
    for line in recipe.open(encoding="utf-8"):
        for sample in json.loads(line)["logical_samples"]:
            if sample["source"] == "syn4d":
                by_scene.setdefault(sample["scene"], []).append(sample)
    print(
        f"RADIUS_AUDIT event=start scenes={len(by_scene)} "
        f"samples={sum(map(len, by_scene.values()))} workers=16",
        flush=True,
    )

    def audit_scene(item):
        scene, samples = item
        root = DATA_ROOT / "datasets/syn4d-mvtracker/train" / scene
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        centres = []
        for view in range(int(manifest["views"])):
            extrinsic = np.load(
                root / str(view) / "extrinsics_w2c.npy",
                mmap_mode="r",
                allow_pickle=False,
            )[0, :3, :4].astype(np.float64)
            centres.append(-extrinsic[:3, :3].T @ extrinsic[:3, 3])
        anchor = np.mean(centres, axis=0)
        tracks = np.load(root / "tracks_xyz.npy", mmap_mode="r", allow_pickle=False)
        sampled_unique = set()
        above_unique = {threshold: set() for threshold in THRESHOLDS}
        above_instances = {threshold: 0 for threshold in THRESHOLDS}
        affected_samples = {threshold: 0 for threshold in THRESHOLDS}
        radii = []
        for sample in samples:
            ids = np.asarray(sample["tracks"], dtype=np.int64)
            frames = np.asarray(sample["frames"], dtype=np.int64)
            xyz = np.asarray(tracks[np.ix_(frames, ids)], dtype=np.float64)
            maximum = np.linalg.norm(xyz - anchor, axis=-1).max(axis=0)
            radii.append(maximum)
            sampled_unique.update(map(int, ids))
            for threshold in THRESHOLDS:
                mask = maximum > threshold
                above_instances[threshold] += int(mask.sum())
                affected_samples[threshold] += int(mask.any())
                above_unique[threshold].update(map(int, ids[mask]))
        return {
            "scene": scene,
            "samples": len(samples),
            "instances": sum(len(sample["tracks"]) for sample in samples),
            "unique": len(sampled_unique),
            "radii": np.concatenate(radii),
            "above_instances": above_instances,
            "above_unique": {key: len(value) for key, value in above_unique.items()},
            "affected_samples": affected_samples,
        }

    results = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(audit_scene, item) for item in by_scene.items()]
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(
                    f"RADIUS_AUDIT event=progress scenes={completed}/{len(futures)}",
                    flush=True,
                )

    radii = np.concatenate([result["radii"] for result in results])
    total_instances = int(sum(result["instances"] for result in results))
    total_samples = int(sum(result["samples"] for result in results))
    report = {
        "recipe": recipe_name,
        "syn4d_samples": total_samples,
        "sampled_track_instances": total_instances,
        "unique_scene_track_pairs": int(sum(result["unique"] for result in results)),
        "radius_quantiles_m": {
            name: float(value)
            for name, value in zip(
                ("p50", "p90", "p95", "p99", "p99_5", "max"),
                np.quantile(radii, (0.5, 0.9, 0.95, 0.99, 0.995, 1.0)),
            )
        },
        "thresholds": {},
    }
    for threshold in THRESHOLDS:
        instances = int(sum(result["above_instances"][threshold] for result in results))
        unique = int(sum(result["above_unique"][threshold] for result in results))
        samples = int(sum(result["affected_samples"][threshold] for result in results))
        report["thresholds"][str(int(threshold))] = {
            "sampled_track_instances": instances,
            "sampled_track_instance_fraction": instances / total_instances,
            "unique_scene_track_pairs": unique,
            "affected_samples": samples,
            "affected_sample_fraction": samples / total_samples,
        }
    print("RADIUS_AUDIT event=complete " + json.dumps(report, sort_keys=True), flush=True)
    run = wandb.init(
        entity="jeetucl-ucl",
        project="mvtracker-continual-training",
        group="recipe-audits",
        job_type="recipe-radius-audit",
        name=f"radius-audit-{recipe_name}",
        tags=["modal", "recipe", "syn4d", "radius-audit"],
        config={"recipe": recipe_name, "source_commit": os.environ["MVTRACKER_MODAL_COMMIT"], **TAGS},
    )
    run.summary.update(
        {
            "syn4d_samples": total_samples,
            "sampled_track_instances": total_instances,
            **{
                f"radius/{threshold}m/instance_fraction": values[
                    "sampled_track_instance_fraction"
                ]
                for threshold, values in report["thresholds"].items()
            },
        }
    )
    run.finish()
    return report


@app.local_entrypoint()
def main(recipe_name: str = "fresh-mixed-da3-r65-singleton-1000-20260825") -> None:
    print(json.dumps(audit_recipe.remote(recipe_name), indent=2, sort_keys=True))
