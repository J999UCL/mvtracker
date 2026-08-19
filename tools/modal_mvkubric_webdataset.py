"""Modal CPU conversion driver for MV-Kubric WebDataset pilot shards."""

from __future__ import annotations

import json
import os
from pathlib import Path

import modal

from modal_training_profile import (
    BASE_TAGS,
    DATA_ROOT,
    _runtime_image,
    _source_commit,
    data_volume,
    wandb_secret,
)


APP_NAME = "jeet-mvtracker-mvkubric-webdataset"
WANDB_PROJECT = "mvtracker-modal-profiling"
WANDB_ENTITY = "jeetucl-ucl"
MODAL_TAGS = {**BASE_TAGS, "experiment": "mvkubric-webdataset-conversion", "gpu": "cpu"}
SOURCE_ROOT = DATA_ROOT / "datasets/kubric-multiview/train"
OUTPUT_ROOT = DATA_ROOT / "datasets/kubric-multiview-webdataset/v1/train"


app = modal.App(APP_NAME, tags=MODAL_TAGS)
image = (
    _runtime_image()
    .apt_install("curl")
    .pip_install("nvidia-dali-cuda120==1.53.0")
    .run_commands(
        "curl -fsSL https://raw.githubusercontent.com/NVIDIA/DALI/v1.53.0/tools/wds2idx.py "
        "-o /usr/local/bin/wds2idx && chmod 0755 /usr/local/bin/wds2idx"
    )
)


@app.function(
    image=image,
    secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=16,
    memory=32768,
    ephemeral_disk=256 * 1024,
    timeout=4 * 60 * 60,
    max_containers=8,
    include_source=False,
)
def convert_remote(
    *,
    scene_root: str = str(SOURCE_ROOT),
    output_root: str = str(OUTPUT_ROOT),
    scenes: tuple[str, ...] = (),
    scenes_per_shard: int = 4,
    shard_workers: int = 1,
    read_workers: int = 16,
) -> dict[str, object]:
    import wandb

    from mvtracker.preprocessing.mvkubric_webdataset import (
        convert_shards,
        discover_scene_ids,
    )

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        job_type="mvkubric-webdataset-conversion",
        tags=["modal", "cpu", "mv-kubric", "webdataset", "pilot"],
        config={
            "source_commit": _source_commit(),
            "scene_root": scene_root,
            "output_root": output_root,
            "scenes_per_shard": scenes_per_shard,
            "shard_workers": shard_workers,
            "read_workers": read_workers,
            **MODAL_TAGS,
        },
    )
    scene_ids = discover_scene_ids(Path(scene_root), scenes or None)
    if not scene_ids:
        raise RuntimeError(f"no scenes found under {scene_root}")
    print(
        f"WDS event=start scenes={len(scene_ids)} shards={shard_workers} "
        f"read_workers={read_workers}",
        flush=True,
    )

    def progress(event, *values):
        if event == "shard":
            result, completed, total = values
            print(
                f"WDS event=shard_completed shard={result['name']} "
                f"progress={completed}/{total} bytes={result['bytes']}",
                flush=True,
            )
            run.log(
                {
                    "progress/shards_completed": completed,
                    "progress/shards_total": total,
                    "progress/latest_shard_bytes": result["bytes"],
                }
            )
            return
        shard, scene_id, completed, seconds = values
        print(
            f"WDS event=scene_completed shard={shard.name} scene={scene_id} "
            f"progress={completed}/{len(shard.scene_ids)} elapsed_seconds={seconds:.1f}",
            flush=True,
        )

    manifest = convert_shards(
        Path(scene_root),
        Path(output_root),
        scene_ids,
        scenes_per_shard=scenes_per_shard,
        shard_workers=shard_workers,
        read_workers=read_workers,
        progress_callback=progress,
    )
    data_volume.commit()
    run.summary.update(
        {
            "scene_count": len(manifest["scene_ids"]),
            "shard_count": len(manifest["shards"]),
            "output_root": output_root,
        }
    )
    run.finish()
    return manifest


@app.local_entrypoint(name="convert")
def convert(
    scene_root: str = str(SOURCE_ROOT),
    output_root: str = str(OUTPUT_ROOT),
    scenes: str = "",
    scenes_per_shard: int = 4,
    shard_workers: int = 1,
    read_workers: int = 16,
) -> None:
    selected = tuple(scene.strip() for scene in scenes.split(",") if scene.strip())
    result = convert_remote.remote(
        scene_root=scene_root,
        output_root=output_root,
        scenes=selected,
        scenes_per_shard=scenes_per_shard,
        shard_workers=shard_workers,
        read_workers=read_workers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
