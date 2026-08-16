#!/usr/bin/env python3
"""Run VGGT-Omega on exactly one temporal chunk and write JSON/PNG QA artifacts.

This is intentionally bounded: it never writes a depth sidecar or any model
array.  Use it to validate a checkpoint, source adapter, metric alignment, and
visual quality before launching a complete preprocessing run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from mvtracker.preprocessing.vggt_omega import (
    MVKubricSceneSource,
    TapVid3DSceneSource,
    checkpoint_sha256,
    infer_temporal_chunk,
    load_model,
)
from mvtracker.preprocessing.vggt_omega_quality import depth_quality_metrics
from scripts.visualize_vggt_omega_quality import _load_source, _save_contact_sheet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("tapvid3d", "mv-kubric"), required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--chunk-length", type=int, default=8)
    parser.add_argument("--views", type=lambda value: tuple(int(item) for item in value.split(",")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-resolution", type=int, default=512)
    return parser.parse_args()


def _source_for(dataset: str, scene_root: Path, views: tuple[int, ...] | None):
    source_type = TapVid3DSceneSource if dataset == "tapvid3d" else MVKubricSceneSource
    return source_type(scene_root, views)


def selected_chunk(frame_count: int, start_frame: int, chunk_length: int) -> tuple[int, ...]:
    """Validate and return one contiguous, bounded temporal chunk."""
    if start_frame < 0 or chunk_length <= 0:
        raise ValueError("start-frame must be non-negative and chunk-length must be positive")
    end_frame = start_frame + chunk_length
    if end_frame > frame_count:
        raise ValueError(f"selected chunk [{start_frame}, {end_frame}) exceeds {frame_count} source frames")
    return tuple(range(start_frame, end_frame))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VGGT-Omega smoke requires CUDA")
    source = _source_for(args.dataset, args.scene_root, args.views)
    description = source.description
    frames = selected_chunk(description.frame_count, args.start_frame, args.chunk_length)
    end_frame = frames[-1] + 1
    view_ids = description.view_ids
    rgbs, ground_truth, gt_intrinsics, gt_w2c = _load_source(
        args.dataset, args.scene_root, view_ids, frames
    )
    model = load_model(args.checkpoint, device)
    started = time.perf_counter()
    result = infer_temporal_chunk(
        source,
        frames,
        model,
        device=device,
        image_resolution=args.image_resolution,
    )
    inference_seconds = time.perf_counter() - started
    known_w2c = source.extrinsics_w2c(frames)
    estimated = np.moveaxis(result.depth, 0, 1)
    cleaned = np.moveaxis(result.cleaned_mask, 0, 1)
    aligned_w2c = np.moveaxis(result.extrinsics_w2c, 0, 1)

    metrics = depth_quality_metrics(
        estimated,
        cleaned,
        ground_truth,
        np.full(len(frames), result.scale, dtype=np.float32),
        aligned_w2c,
        np.moveaxis(known_w2c, 0, 1),
        frames,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / f"{description.name}__frames-{args.start_frame}-{end_frame}__vggt_omega_smoke"
    report = {
        "format": "mvtracker_vggt_omega_smoke",
        "dataset": args.dataset,
        "scene": description.name,
        "view_ids": list(view_ids),
        "frame_indices": list(frames),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "device": str(device),
        "inference_seconds": inference_seconds,
        **metrics,
    }
    (prefix.with_suffix(".json")).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _save_contact_sheet(prefix.with_suffix(".png"), frames, rgbs, ground_truth, estimated, cleaned)
    print(json.dumps({"event": "vggt_omega_smoke_complete", "json": str(prefix.with_suffix('.json')), "png": str(prefix.with_suffix('.png'))}, sort_keys=True))


if __name__ == "__main__":
    main()
