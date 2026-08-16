#!/usr/bin/env python3
"""Generate metric VGGT-Omega depth sidecars for MVTracker training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mvtracker.preprocessing.vggt_omega import (
    MVKubricSceneSource,
    TapVid3DSceneSource,
    checkpoint_sha256,
    load_model,
    preprocess_scene,
    profile_batch_sizes,
)


SOURCES = {
    "tapvid3d": TapVid3DSceneSource,
    "mv-kubric": MVKubricSceneSource,
}


def _csv_ints(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item]
    if not result:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(SOURCES), required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenes", nargs="*", help="scene directory names; defaults to every scene")
    parser.add_argument("--views", type=_csv_ints)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--profile-batch-sizes", type=_csv_ints)
    parser.add_argument("--max-vram-fraction", type=float, default=0.9)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if not 0 < args.max_vram_fraction <= 1:
        raise ValueError("--max-vram-fraction must be in (0, 1]")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("VGGT-Omega preprocessing requires CUDA")
    scene_paths = (
        [args.input_root / name for name in args.scenes]
        if args.scenes
        else sorted(path for path in args.input_root.iterdir() if path.is_dir())
    )
    if not scene_paths:
        raise ValueError(f"no scenes found under {args.input_root}")
    source_type = SOURCES[args.dataset]
    first_source = source_type(scene_paths[0], args.views)
    model = load_model(args.checkpoint, device)
    digest = checkpoint_sha256(args.checkpoint)
    batch_size = args.batch_size
    if args.profile_batch_sizes:
        batch_size = profile_batch_sizes(
            first_source,
            model,
            batch_sizes=args.profile_batch_sizes,
            device=device,
            image_resolution=args.image_resolution,
            max_vram_fraction=args.max_vram_fraction,
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    print(
        json.dumps(
            {
                "event": "run_started",
                "dataset": args.dataset,
                "scene_count": len(scene_paths),
                "batch_size": batch_size,
                "checkpoint_sha256": digest,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    for index, scene_path in enumerate(scene_paths):
        source = first_source if index == 0 else source_type(scene_path, args.views)
        preprocess_scene(
            source,
            args.output_root,
            model,
            args.checkpoint,
            digest,
            batch_size=batch_size,
            device=device,
            image_resolution=args.image_resolution,
        )
if __name__ == "__main__":
    main()
