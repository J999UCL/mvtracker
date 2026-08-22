"""Render visibility-aware track overlays for the two Syn4D loss spikes."""

from __future__ import annotations

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


APP_NAME = "jeet-mvtracker-syn4d-track-overlay"
DATASET_ROOT = DATA_ROOT / "datasets/syn4d-mvtracker"
OUTPUT_ROOT = RUN_ROOT / "syn4d-track-overlays"
TAGS = {"owner": "jeet", "project": "mvtracker", "purpose": "evaluation"}
SCENES = (
    {
        "name": "planet_bald__seq_000018",
        "virtual_index": 70,
        "view_count": 6,
        "window": (12, 36),
        "views": (0, 1, 2, 5, 6, 7),
        "tracks": 1885,
        "training_step": 26,
    },
    {
        "name": "castle__seq_000007",
        "virtual_index": 321,
        "view_count": 1,
        "window": (8, 32),
        "views": (1,),
        "tracks": 1530,
        "training_step": 132,
    },
)


app = modal.App(APP_NAME, tags={**TAGS, "experiment": "syn4d-track-overlay"})
image = _source_image(_dependency_image())


def _training_dataset():
    from omegaconf import OmegaConf

    from mvtracker.datasets.syn4d_multiview_dataset import Syn4DMultiViewDataset

    source_root = Path("/opt/mvtracker")
    cfg = OmegaConf.merge(
        OmegaConf.load(source_root / "configs/train.yaml"),
        OmegaConf.load(
            source_root / "configs/experiment/diegesis_syn4d_mvkubric_gt_ddp.yaml"
        ),
    )
    cfg.datasets.root = str(DATA_ROOT / "datasets")
    cfg.reproducibility.seed = 2379791757

    class Fabric:
        world_size = 2

    source_cfg = cfg.datasets.train.sources.syn4d
    kwargs = Syn4DMultiViewDataset.from_name(
        source_cfg.name,
        source_cfg.root,
        cfg,
        Fabric(),
        just_return_kwargs=True,
        include_scene_ids=source_cfg.include_scene_ids,
    )
    kwargs["view_count_probabilities"] = tuple(source_cfg.view_count_probabilities)
    return Syn4DMultiViewDataset(**kwargs)


def _sample_plan(dataset, specification):
    from mvtracker.datasets.mixed_source_schedule import ScheduledSampleRequest

    scene_index = dataset.seq_names.index(specification["name"])
    request = ScheduledSampleRequest(
        virtual_index=int(specification["virtual_index"]),
        scene_index=scene_index,
        view_count=None,
    )
    plan = dataset.plan_sample(request)
    observed = {
        "window": (int(plan.frame_indices[0]), int(plan.frame_indices[-1] + 1)),
        "views": tuple(plan.views),
        "tracks": int(plan.track_count),
    }
    expected = {
        "window": tuple(specification["window"]),
        "views": tuple(specification["views"]),
        "tracks": int(specification["tracks"]),
    }
    if observed != expected:
        raise RuntimeError(f"sample reproduction mismatch: {observed} != {expected}")
    return plan


def _jpeg_frame(root: Path, view: int, frame: int, offsets, handles):
    import cv2
    import numpy as np

    handle = handles.setdefault(view, (root / f"view_{view}" / "jpeg_bytes.bin").open("rb"))
    start, stop = (int(value) for value in offsets[view][frame : frame + 2])
    handle.seek(start)
    encoded = np.frombuffer(handle.read(stop - start), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"JPEG decode failed for {root.name} view {view} frame {frame}")
    return image


def _project(points, extrinsic, intrinsic):
    import numpy as np

    transform = np.asarray(extrinsic, dtype=np.float32)[:3, :4]
    camera = points @ transform[:, :3].T + transform[:, 3]
    matrix = np.asarray(intrinsic, dtype=np.float32)
    if matrix.shape == (4,):
        fx, fy, cx, cy = matrix
    else:
        fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    xy = np.stack(
        (fx * camera[:, 0] / camera[:, 2] + cx, fy * camera[:, 1] / camera[:, 2] + cy),
        axis=-1,
    )
    return xy, camera[:, 2]


def _colors(track_ids):
    import cv2
    import numpy as np

    hues = ((np.asarray(track_ids, dtype=np.uint64) * 2654435761) % 180).astype(np.uint8)
    hsv = np.stack((hues, np.full_like(hues, 220), np.full_like(hues, 255)), axis=-1)
    return cv2.cvtColor(hsv[None], cv2.COLOR_HSV2BGR)[0]


def _render_scene(dataset, specification, output_root: Path):
    import cv2
    import numpy as np
    import subprocess

    plan = _sample_plan(dataset, specification)
    root = Path(dataset.data_root) / plan.sequence
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    frame_count = int(manifest["frames"])
    fps = float(manifest["fps"])
    view_count = int(manifest["views"])
    selected_views = set(int(view) for view in plan.views)
    track_ids = np.asarray(plan.selected_global_track_indices, dtype=np.int64)
    tracks = np.load(root / "tracks_xyz.npy", mmap_mode="r")[:, track_ids]
    valid = np.load(root / "track_valid.npy", mmap_mode="r")[:, track_ids]
    visibility = {
        view: np.load(root / str(view) / "visibility.npy", mmap_mode="r")[:, track_ids]
        for view in range(view_count)
    }
    intrinsics = {
        view: np.load(root / str(view) / "intrinsics.npy", mmap_mode="r")
        for view in range(view_count)
    }
    extrinsics = {
        view: np.load(root / str(view) / "extrinsics_w2c.npy", mmap_mode="r")
        for view in range(view_count)
    }
    offsets = {
        view: np.load(root / f"view_{view}" / "jpeg_offsets.npy")
        for view in range(view_count)
    }
    colors = _colors(track_ids)
    window_start, window_stop = (int(value) for value in specification["window"])
    window_tracks = np.asarray(tracks[window_start:window_stop])
    movement = np.linalg.norm(np.diff(window_tracks, axis=0), axis=-1).sum(axis=0)
    moving = np.argsort(movement)[-min(256, len(movement)) :]

    tile_width, tile_height = 480, 270
    columns, rows = 4, 2
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{plan.sequence}_tracks.mp4"
    command = [
        "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{columns * tile_width}x{rows * tile_height}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    handles = {}
    history = {view: [] for view in range(view_count)}
    try:
        for frame in range(frame_count):
            tiles = []
            for view in range(view_count):
                image = _jpeg_frame(root, view, frame, offsets, handles)
                source_height, source_width = image.shape[:2]
                image = cv2.resize(image, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
                selected = view in selected_views
                if not selected:
                    image = (image.astype(np.float32) * 0.28).astype(np.uint8)
                else:
                    points, depth = _project(
                        np.asarray(tracks[frame]),
                        extrinsics[view][frame],
                        intrinsics[view][frame] if intrinsics[view].ndim == 3 else intrinsics[view],
                    )
                    points[:, 0] *= (tile_width - 1) / (source_width - 1)
                    points[:, 1] *= (tile_height - 1) / (source_height - 1)
                    visible = (
                        np.asarray(visibility[view][frame], dtype=bool)
                        & np.asarray(valid[frame], dtype=bool)
                        & np.isfinite(points).all(axis=1)
                        & np.isfinite(depth)
                        & (depth > 0)
                    )
                    history[view].append((points.copy(), visible.copy()))
                    history[view] = history[view][-8:]
                    for index in moving:
                        trail = [
                            tuple(np.rint(old_points[index]).astype(int))
                            for old_points, old_visible in history[view]
                            if old_visible[index]
                        ]
                        if len(trail) >= 2:
                            cv2.polylines(
                                image,
                                [np.asarray(trail, dtype=np.int32)],
                                False,
                                tuple(int(value) for value in colors[index]),
                                1,
                                cv2.LINE_AA,
                            )
                    for index in np.flatnonzero(visible):
                        x, y = np.rint(points[index]).astype(int)
                        if 0 <= x < tile_width and 0 <= y < tile_height:
                            cv2.circle(
                                image,
                                (x, y),
                                1,
                                tuple(int(value) for value in colors[index]),
                                -1,
                                cv2.LINE_AA,
                            )
                border = (70, 220, 70) if selected and window_start <= frame < window_stop else (90, 90, 90)
                cv2.rectangle(image, (0, 0), (tile_width - 1, tile_height - 1), border, 3)
                cv2.putText(
                    image,
                    f"VIEW {view}{'  SELECTED' if selected else ''}",
                    (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                if window_start <= frame < window_stop:
                    cv2.putText(
                        image,
                        f"LOSS-SPIKE SAMPLE  FRAME {frame}",
                        (12, tile_height - 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (80, 230, 255),
                        2,
                        cv2.LINE_AA,
                    )
                tiles.append(image)
            grid = np.concatenate(
                [np.concatenate(tiles[row * columns : (row + 1) * columns], axis=1) for row in range(rows)],
                axis=0,
            )
            encoder.stdin.write(grid.tobytes())
            if frame % 20 == 0:
                print(f"OVERLAY scene={plan.sequence} frame={frame}/{frame_count}", flush=True)
    finally:
        if encoder.stdin is not None:
            encoder.stdin.close()
        return_code = encoder.wait()
        for handle in handles.values():
            handle.close()
    if return_code:
        raise RuntimeError(f"ffmpeg exited with {return_code}")
    return {
        "scene": plan.sequence,
        "path": str(output_path),
        "training_step": int(specification["training_step"]),
        "window": [window_start, window_stop],
        "views": list(plan.views),
        "tracks": int(plan.track_count),
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
def render_remote(run_name: str):
    import wandb

    run = wandb.init(
        entity="jeetucl-ucl",
        project="mvtracker-continual-training",
        group="syn4d-data-audit",
        job_type="track-overlay",
        name=run_name,
        tags=["syn4d", "cpu", "track-overlay"],
        config={"source_commit": _source_commit(), **TAGS},
    )
    dataset = _training_dataset()
    output_root = OUTPUT_ROOT / run_name
    results = [_render_scene(dataset, specification, output_root) for specification in SCENES]
    (output_root / "manifest.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    run_volume.commit()
    run.summary["output_root"] = str(output_root)
    run.finish()
    return results


@app.local_entrypoint()
def main(run_name: str = ""):
    selected = run_name or (
        "planet-castle-track-overlay-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    app.set_tags({**TAGS, "experiment": selected, "gpu": "cpu"})
    print(json.dumps(render_remote.remote(selected), indent=2))
