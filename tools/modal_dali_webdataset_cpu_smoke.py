"""CPU-only throughput smoke for DALI's native WebDataset reader."""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import threading
import time

import modal

from modal_training_profile import (
    BASE_TAGS,
    DATA_ROOT,
    _runtime_image,
    _source_commit,
    data_volume,
    wandb_secret,
)


APP_NAME = "jeet-mvtracker-dali-webdataset-cpu-smoke"
TRAIN_ROOT = DATA_ROOT / "datasets/kubric-multiview-webdataset/train"
SCENES_PER_BATCH = 4
RECORDS_PER_SCENE = 11
MODAL_TAGS = {
    **BASE_TAGS,
    "experiment": "dali-webdataset-cpu-smoke",
    "gpu": "cpu",
}

app = modal.App(APP_NAME, tags=MODAL_TAGS)


def _heartbeat(stop: threading.Event, state: dict[str, object]) -> None:
    while not stop.wait(10):
        print(
            "DALI_CPU event=waiting "
            f"batch={state['batch']} elapsed_seconds="
            f"{time.perf_counter() - float(state['started']):.1f}",
            flush=True,
        )


@app.function(
    image=_runtime_image(),
    secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume.with_mount_options(read_only=True)},
    cpu=16,
    memory=32768,
    timeout=60 * 60,
    max_containers=1,
    include_source=False,
)
def benchmark(shards: int = 8) -> dict[str, object]:
    print("DALI_CPU event=function_started", flush=True)
    import numpy as np
    print("DALI_CPU event=numpy_imported", flush=True)
    import nvidia.dali.fn as fn
    from nvidia.dali import pipeline_def
    print("DALI_CPU event=dali_imported", flush=True)
    import wandb
    print("DALI_CPU event=wandb_imported", flush=True)

    manifest = json.loads((TRAIN_ROOT / "manifest.json").read_text())
    print("DALI_CPU event=manifest_loaded", flush=True)
    pairs = [
        (TRAIN_ROOT / shard["tar"], (TRAIN_ROOT / shard["tar"]).with_suffix(".idx"))
        for shard in manifest["shards"]
        if int(shard["nsamples"]) == 44
    ]
    if not 1 <= shards <= len(pairs):
        raise ValueError(f"shards must be between 1 and {len(pairs)}")

    seed = random.SystemRandom().randrange(2**32)
    selected = random.Random(seed).sample(pairs, shards)
    paths = [str(pair[0]) for pair in selected]
    index_paths = [str(pair[1]) for pair in selected]
    batch_records = SCENES_PER_BATCH * RECORDS_PER_SCENE
    batch_count = math.ceil(shards * 44 / batch_records)

    @pipeline_def
    def reader_pipeline():
        outputs = fn.readers.webdataset(
            paths=paths,
            index_paths=index_paths,
            ext=["meta.npz", "rgb.npz", "depth.npz"],
            missing_component_behavior="empty",
            random_shuffle=False,
            prefetch_queue_depth=2,
            name="Reader",
        )
        return tuple(outputs)

    run = wandb.init(
        entity="jeetucl-ucl",
        project="mvtracker-modal-profiling",
        job_type="dali-webdataset-cpu-smoke",
        tags=["modal", "cpu", "dali", "webdataset", "direct-volume"],
        config={
            "source_commit": _source_commit(),
            "seed": seed,
            "shards": shards,
            "paths": paths,
            **MODAL_TAGS,
        },
    )
    print(
        f"DALI_CPU event=start shards={shards} scenes={shards * 4} "
        f"batches={batch_count} seed={seed}",
        flush=True,
    )

    build_started = time.perf_counter()
    print("DALI_CPU event=pipeline_build_started", flush=True)
    pipeline = reader_pipeline(
        batch_size=batch_records,
        num_threads=8,
        device_id=None,
        prefetch_queue_depth=2,
    )
    pipeline.build()
    build_seconds = time.perf_counter() - build_started
    print(
        f"DALI_CPU event=pipeline_build_complete seconds={build_seconds:.3f}",
        flush=True,
    )

    total_bytes = 0
    batch_seconds = []
    started = time.perf_counter()
    for batch_index in range(batch_count):
        state = {"batch": batch_index + 1, "started": time.perf_counter()}
        stop = threading.Event()
        watcher = threading.Thread(target=_heartbeat, args=(stop, state), daemon=True)
        watcher.start()
        outputs = pipeline.run()
        stop.set()
        watcher.join()
        elapsed = time.perf_counter() - float(state["started"])
        batch_seconds.append(elapsed)

        sizes = [[np.asarray(output.at(i)).nbytes for output in outputs] for i in range(len(outputs[0]))]
        for position, (meta_bytes, rgb_bytes, depth_bytes) in enumerate(sizes):
            scene_position = position % RECORDS_PER_SCENE
            if scene_position == 0:
                if not (meta_bytes > 0 and rgb_bytes == 0 and depth_bytes == 0):
                    raise RuntimeError(f"batch {batch_index + 1}: malformed metadata record {position}")
            elif not (meta_bytes == 0 and rgb_bytes > 0 and depth_bytes > 0):
                raise RuntimeError(f"batch {batch_index + 1}: malformed media record {position}")
            total_bytes += meta_bytes + rgb_bytes + depth_bytes

        print(
            f"DALI_CPU event=batch_complete batch={batch_index + 1}/{batch_count} "
            f"seconds={elapsed:.3f} payload_gib={sum(map(sum, sizes)) / 1024**3:.3f}",
            flush=True,
        )

    total_seconds = time.perf_counter() - started
    summary = {
        "format": "mvtracker-dali-webdataset-cpu-smoke",
        "seed": seed,
        "shards": shards,
        "scenes": shards * 4,
        "batches": batch_count,
        "build_seconds": build_seconds,
        "first_batch_seconds": batch_seconds[0],
        "batch_seconds_median": float(np.median(batch_seconds)),
        "total_seconds": total_seconds,
        "payload_bytes": total_bytes,
        "payload_mib_per_second": total_bytes / total_seconds / 1024**2,
        "scenes_per_second": shards * 4 / total_seconds,
    }
    run.summary.update(summary)
    run.finish()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


@app.local_entrypoint()
def main(shards: int = 8) -> None:
    print(json.dumps(benchmark.remote(shards), indent=2, sort_keys=True))
