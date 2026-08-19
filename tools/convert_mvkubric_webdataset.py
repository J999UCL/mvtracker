#!/usr/bin/env python3
"""Convert native MV-Kubric scenes to DALI-indexed WebDataset TAR shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from mvtracker.preprocessing.mvkubric_webdataset import (
    SCENES_PER_SHARD,
    convert_shards,
    discover_scene_ids,
)


def _scene_progress(shard, scene_id, completed, seconds):
    print(
        f"WDS event=scene_completed shard={shard.name} scene={scene_id} "
        f"shard_progress={completed}/{len(shard.scene_ids)} elapsed_seconds={seconds:.1f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--scene", dest="scenes", action="append")
    parser.add_argument("--scenes-per-shard", type=int, default=SCENES_PER_SHARD)
    parser.add_argument("--shard-workers", type=int, default=1)
    parser.add_argument("--read-workers", type=int, default=16)
    parser.add_argument("--index-command", default="widsindex")
    args = parser.parse_args()
    if args.shard_workers < 1 or args.read_workers < 1:
        parser.error("--shard-workers and --read-workers must be positive")

    scene_ids = discover_scene_ids(args.scene_root, args.scenes)
    if not scene_ids:
        parser.error(f"no numeric scenes found under {args.scene_root}")
    started = time.perf_counter()
    print(
        f"WDS event=start scenes={len(scene_ids)} "
        f"shard_workers={args.shard_workers} read_workers={args.read_workers}",
        flush=True,
    )
    def progress(event, *values):
        if event == "shard":
            result, completed, total = values
            print(
                f"WDS event=shard_completed shard={result['name']} "
                f"progress={completed}/{total} bytes={result['bytes']} status={result['status']}",
                flush=True,
            )
            return
        _scene_progress(event, *values)

    manifest = convert_shards(
        args.scene_root,
        args.output_root,
        scene_ids,
        scenes_per_shard=args.scenes_per_shard,
        shard_workers=args.shard_workers,
        read_workers=args.read_workers,
        index_command=args.index_command,
        progress_callback=progress,
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "scenes": len(scene_ids),
                "seconds": time.perf_counter() - started,
                "output_root": str(args.output_root),
                "shards": len(manifest["shards"]),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
