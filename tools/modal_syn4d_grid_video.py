"""Render eight-view RGB grid videos from the prepared Syn4D cache."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import modal

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


APP_NAME = "jeet-mvtracker-syn4d-grid-video"
DATASET_ROOT = DATA_ROOT / "datasets/syn4d-mvtracker/train"
OUTPUT_ROOT = RUN_ROOT / "syn4d-grid-videos"
TAGS = {"owner": "jeet", "project": "mvtracker", "purpose": "evaluation"}
DEFAULT_SCENES = "cave_group__seq_000008,desert_bald__seq_000012"


app = modal.App(APP_NAME, tags={**TAGS, "experiment": "syn4d-grid-video"})
image = _source_image(_dependency_image())


def _render(scene: str, output_root: Path) -> dict[str, object]:
    import cv2
    import numpy as np
    import subprocess

    root = DATASET_ROOT / scene
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    frame_count = int(manifest["frames"])
    fps = float(manifest["fps"])
    view_count = int(manifest["views"])
    offsets = {
        view: np.load(root / f"view_{view}" / "jpeg_offsets.npy")
        for view in range(view_count)
    }
    tile_width, tile_height = 480, 270
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{scene}_all_8_views.mp4"
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{4 * tile_width}x{2 * tile_height}", "-r", str(fps), "-i", "-",
            "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
    )
    handles = {}
    try:
        for frame in range(frame_count):
            tiles = []
            for view in range(view_count):
                image = _jpeg_frame(root, view, frame, offsets, handles)
                image = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
                cv2.putText(
                    image,
                    f"VIEW {view}",
                    (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                tiles.append(image)
            grid = np.concatenate(
                [np.concatenate(tiles[:4], axis=1), np.concatenate(tiles[4:], axis=1)],
                axis=0,
            )
            encoder.stdin.write(grid.tobytes())
            if frame % 25 == 0:
                print(f"GRID scene={scene} frame={frame}/{frame_count}", flush=True)
    finally:
        encoder.stdin.close()
        return_code = encoder.wait()
        for handle in handles.values():
            handle.close()
    if return_code:
        raise RuntimeError(f"ffmpeg exited with {return_code}")
    return {
        "scene": scene,
        "frames": frame_count,
        "fps": fps,
        "path": str(output_path),
    }


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
def render_remote(run_name: str, scenes: tuple[str, ...]):
    import wandb

    run = wandb.init(
        entity="jeetucl-ucl",
        project="mvtracker-continual-training",
        group="syn4d-data-audit",
        job_type="grid-video",
        name=run_name,
        tags=["syn4d", "cpu", "grid-video"],
        config={"source_commit": _source_commit(), "scenes": list(scenes), **TAGS},
    )
    output_root = OUTPUT_ROOT / run_name
    results = [_render(scene, output_root) for scene in scenes]
    (output_root / "manifest.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    run_volume.commit()
    run.summary["output_root"] = str(output_root)
    run.finish()
    return results


@app.local_entrypoint()
def main(scenes: str = DEFAULT_SCENES, run_name: str = ""):
    selected_scenes = tuple(value.strip() for value in scenes.split(",") if value.strip())
    selected_run = run_name or (
        "syn4d-grid-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    app.set_tags({**TAGS, "experiment": selected_run, "gpu": "cpu"})
    print(json.dumps(render_remote.remote(selected_run, selected_scenes), indent=2))
