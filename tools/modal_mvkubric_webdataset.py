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
    run_volume,
    wandb_secret,
)


APP_NAME = "jeet-mvtracker-mvkubric-webdataset"
WANDB_PROJECT = "mvtracker-modal-profiling"
WANDB_ENTITY = "jeetucl-ucl"
MODAL_TAGS = {**BASE_TAGS, "experiment": "mvkubric-webdataset-pilot"}
SOURCE_ROOT = DATA_ROOT / "datasets/kubric-multiview/train"
OUTPUT_ROOT = DATA_ROOT / "datasets/kubric-multiview-webdataset/train"
RUN_ROOT = Path("/mnt/mvtracker-runs")


app = modal.App(APP_NAME, tags=MODAL_TAGS)
conversion_image = _runtime_image()
# The runtime image already installs DALI and its compatible nvImageCodec /
# libnvcomp ABI set. Keep the benchmark image source-free after that layer.
benchmark_image = conversion_image


@app.function(
    image=conversion_image,
    secrets=[wandb_secret],
    volumes={str(DATA_ROOT): data_volume},
    cpu=16,
    memory=32768,
    ephemeral_disk=512 * 1024,
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
        shard = event
        scene_id, completed, seconds = values
        print(
            f"WDS event=scene_completed shard={shard.name} scene={scene_id} "
            f"progress={completed}/{len(shard.scene_ids)} elapsed_seconds={seconds:.1f}",
            flush=True,
        )

    target_root = Path(output_root)
    staging_root = target_root.with_name(target_root.name + ".staging")
    if target_root.exists():
        raise FileExistsError(f"refusing to replace existing canonical output: {target_root}")
    if staging_root.exists():
        raise FileExistsError(f"refusing to reuse existing staging output: {staging_root}")
    manifest = convert_shards(
        Path(scene_root),
        staging_root,
        scene_ids,
        scenes_per_shard=scenes_per_shard,
        shard_workers=shard_workers,
        read_workers=read_workers,
        progress_callback=progress,
    )
    staging_root.replace(target_root)
    data_volume.commit()
    run.summary.update(
        {
            "scene_count": len(manifest["scene_ids"]),
            "shard_count": len(manifest["shards"]),
            "output_root": str(target_root),
        }
    )
    run.finish()
    return manifest


@app.function(
    image=benchmark_image,
    secrets=[wandb_secret],
    volumes={
        str(DATA_ROOT): data_volume.with_mount_options(read_only=True),
        str(RUN_ROOT): run_volume,
    },
    gpu="T4",
    cpu=16,
    memory=32768,
    timeout=2 * 60 * 60,
    max_containers=1,
    include_source=False,
)
def benchmark_remote(
    run_name: str,
    *,
    scenes: tuple[str, ...] = tuple(str(scene) for scene in range(1001, 1033)),
    warmup: int = 4,
    measured: int = 16,
    workers: int = 8,
) -> dict[str, object]:
    import wandb

    from tools.check_dali_decode import check_pair
    from mvtracker.profiling.mvkubric_webdataset_benchmark import benchmark_matrix
    from mvtracker.profiling.t4_loader_benchmark import ContainerHardwareMonitor, GpuHardwareMonitor

    if not run_name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in run_name):
        raise ValueError("run_name contains unsupported characters")
    scene_ids = tuple(str(scene) for scene in scenes)
    if len(scene_ids) != 32:
        raise ValueError("the T4 pilot uses exactly 32 MV-Kubric scenes")
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        job_type="mvkubric-webdataset-t4-benchmark",
        tags=["modal", "t4", "loader", "native-vs-dali", "pilot"],
        config={
            "source_commit": _source_commit(),
            "gpu": "T4",
            "scenes": list(scene_ids),
            "warmup": warmup,
            "measured": measured,
            "workers": workers,
            **MODAL_TAGS,
        },
    )
    output_dir = RUN_ROOT / "t4-mvkubric-webdataset"
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{run_name}.json"
    container_monitor = ContainerHardwareMonitor()
    gpu_monitor = GpuHardwareMonitor()

    def hardware_sample():
        return {"cpu_ram": container_monitor.sample(), "gpu": gpu_monitor.sample()}

    parity_root = DATA_ROOT / "datasets/kubric-multiview/train/1001/view_0"
    print("WDS_BENCH event=parity_start scene=1001 view=0 frame=0", flush=True)
    dali_parity = check_pair(
        parity_root / "rgba_00000.png",
        parity_root / "depth_00000.tiff",
    )
    print(
        "WDS_BENCH event=parity_complete "
        f"rgb_max_abs_error={dali_parity['rgb_max_abs_error']} "
        f"depth_max_abs_error={dali_parity['depth_max_abs_error']}",
        flush=True,
    )
    run.summary.update({"dali_parity": dali_parity})
    run.log(
        {
            "dali_parity/rgb_max_abs_error": dali_parity["rgb_max_abs_error"],
            "dali_parity/depth_max_abs_error": dali_parity["depth_max_abs_error"],
        }
    )

    def progress(case_name, result):
        print(
            f"WDS_BENCH event=case_complete case={case_name} "
            f"native_samples_per_second={result['native']['samples_per_second']:.3f} "
            f"dali_samples_per_second={result['dali']['samples_per_second']:.3f}",
            flush=True,
        )
        scalars = {}
        for path, profile in result.items():
            for key in (
                "startup_seconds", "cold_first_sample_seconds", "read_unpack_seconds_p50",
                "read_unpack_seconds_p95", "gpu_decode_seconds_p50", "gpu_decode_seconds_p95",
                "exposed_wait_seconds_p50", "exposed_wait_seconds_p95", "samples_per_second",
                "encoded_bytes_per_second",
            ):
                if key in profile:
                    scalars[f"cases/{case_name}/{path}/{key}"] = profile[key]
        run.log(scalars)

    try:
        result = benchmark_matrix(
            DATA_ROOT,
            DATA_ROOT / "datasets/kubric-multiview-webdataset",
            scene_ids,
            warmup=warmup,
            measured=measured,
            workers=workers,
            hardware_sampler=hardware_sample,
            progress_callback=progress,
        )
        report = {
            "format": "mvtracker_mvkubric_webdataset_t4_benchmark",
            "source_commit": _source_commit(),
            "modal_tags": MODAL_TAGS,
            "gpu": "T4",
            "data_root": str(DATA_ROOT),
            "webdataset_root": str(DATA_ROOT / "datasets/kubric-multiview-webdataset"),
            "result": result,
            "dali_parity": dali_parity,
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        run.summary.update({"report_path": str(report_path), "scene_count": len(scene_ids)})
        run.summary["view_counts"] = [1, 2, 4, 6]
        run.finish()
        run_volume.commit()
        return {"report_path": str(report_path), **report}
    finally:
        gpu_monitor.close()


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


@app.local_entrypoint(name="benchmark")
def benchmark(
    run_name: str = "",
    scenes: str = ",".join(str(scene) for scene in range(1001, 1033)),
    warmup: int = 4,
    measured: int = 16,
    workers: int = 8,
) -> None:
    import datetime

    selected = tuple(scene.strip() for scene in scenes.split(",") if scene.strip())
    name = run_name or f"mvkubric-webdataset-t4-{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}"
    result = benchmark_remote.remote(name, scenes=selected, warmup=warmup, measured=measured, workers=workers)
    print(json.dumps(result, indent=2, sort_keys=True))
