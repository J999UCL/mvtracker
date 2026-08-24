"""Long-sequence VGGT-Omega inference and storage/readback benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import time

import modal


APP_NAME = "jeet-mvtracker-vggt-omega-long"
DATA_VOLUME_NAME = "jeet-mvtracker-data-v2"
RUN_VOLUME_NAME = "jeet-mvtracker-runs-v2"
DATA_ROOT = Path("/mnt/mvtracker-data")
RUN_ROOT = Path("/mnt/mvtracker-runs")
CHECKPOINT_REPO = "facebook/VGGT-Omega"
CHECKPOINT_REVISION = "ba9db085d6b7349b738fa2e37d198bb4dd077954"
CHECKPOINT_FILENAME = "vggt_omega_1b_512.pt"
BASE_TAGS = {"owner": "jeet", "project": "mvtracker", "purpose": "profiling"}
COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def _commit() -> str:
    commit = os.environ.get("MVTRACKER_MODAL_COMMIT", "")
    if COMMIT_RE.fullmatch(commit) is None:
        raise RuntimeError("MVTRACKER_MODAL_COMMIT must be a full lowercase Git commit")
    return commit


def _image() -> modal.Image:
    commit = _commit()
    clone = (
        "git init /opt/mvtracker && "
        "git -C /opt/mvtracker remote add origin https://github.com/J999UCL/mvtracker.git && "
        f"git -C /opt/mvtracker fetch --depth=1 origin {commit} && "
        "git -C /opt/mvtracker checkout --detach FETCH_HEAD && "
        f'test "$(git -C /opt/mvtracker rev-parse HEAD)" = "{commit}"'
    )
    requirements = Path(__file__).resolve().parents[1] / "requirements.vggt-omega.txt"
    return (
        modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.10")
        .apt_install("git", "libgl1", "libglib2.0-0")
        .pip_install(
            "torch==2.7.1",
            "torchvision==0.22.1",
            "torchaudio==2.7.1",
            index_url="https://download.pytorch.org/whl/cu128",
        )
        .pip_install_from_requirements(str(requirements))
        .pip_install(
            "huggingface-hub==0.30.2",
            "hf_xet==1.1.9",
            "wandb==0.19.9",
            "Pillow",
            "nvidia-dali-cuda120==1.53.0",
        )
        .run_commands(clone)
        .env(
            {
                "MVTRACKER_MODAL_COMMIT": commit,
                "PYTHONPATH": "/opt/mvtracker:/opt/mvtracker/tools",
                "TORCH_CUDA_ARCH_LIST": "9.0",
            }
        )
    )


app = modal.App(APP_NAME, tags={**BASE_TAGS, "experiment": "vggt-omega-long"})
image = _image()
data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=False, version=2)
run_volume = modal.Volume.from_name(RUN_VOLUME_NAME, create_if_missing=True, version=2)
hf_secret = modal.Secret.from_name("jeet-mvtracker-huggingface", required_keys=["HF_TOKEN"])
wandb_secret = modal.Secret.from_name("jeet-mvtracker-wandb", required_keys=["WANDB_API_KEY"])


SCENES = {
    "diegesis": ("diningroom04", "kitchen03"),
    "syn4d": ("flying_group__seq_000015", "countryside__seq_000019"),
    "mv-kubric": ("1001", "1002"),
}


def _sources(data_root: Path):
    from mvtracker.preprocessing.vggt_omega import MVKubricSceneSource, PackedJpegSceneSource

    diegesis = []
    for scene in SCENES["diegesis"]:
        cache = data_root / "datasets/diegesis-mvtracker/TAPVid3D_MVTracker_cache/train" / scene
        # Modal Volume mounts preserve the raw-tree symlinks but do not resolve
        # them.  The immutable source tree is mounted in the same Volume.
        camera_root = data_root / "source/diegesis/scenes" / scene / "tracking/sequence"
        diegesis.append(PackedJpegSceneSource(cache, camera_root=camera_root, view_ids=(0, 1, 2, 3)))
    syn4d = [
        PackedJpegSceneSource(
            data_root / "datasets/syn4d-mvtracker/train" / scene,
            view_ids=tuple(range(8)),
        )
        for scene in SCENES["syn4d"]
    ]
    mvkubric = [
        MVKubricSceneSource(data_root / "datasets/kubric-multiview/train" / scene)
        for scene in SCENES["mv-kubric"]
    ]
    return {"diegesis": diegesis, "syn4d": syn4d, "mv-kubric": mvkubric}


def _windows(frame_count: int, window_size: int) -> list[tuple[int, int, int, int]]:
    if window_size >= frame_count:
        return [(0, frame_count, 0, frame_count)]
    overlap = max(8, window_size // 4)
    stride = window_size - overlap
    starts = list(range(0, max(frame_count - window_size, 0) + 1, stride))
    final_start = frame_count - window_size
    if starts[-1] != final_start:
        starts.append(final_start)
    records = []
    for index, start in enumerate(starts):
        end = min(start + window_size, frame_count)
        left = 0 if index == 0 else (starts[index - 1] + start) // 2
        right = frame_count if index == len(starts) - 1 else (start + starts[index + 1]) // 2
        records.append((start, end, left, right))
    return records


def _probe(source, model, device, candidates, infer_temporal_chunks, *, batch_sources=()):
    import torch

    total_memory = torch.cuda.get_device_properties(device).total_memory
    trials = []
    for temporal_frames in candidates:
        sources = tuple(batch_sources) if batch_sources else (source,)
        if any(temporal_frames > item.description.frame_count for item in sources):
            continue
        result = {
            "temporal_frames": temporal_frames,
            "sequence_images": temporal_frames * len(sources[0].description.view_ids),
            "batch_size": len(sources),
        }
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            output = infer_temporal_chunks(
                sources,
                range(temporal_frames),
                model,
                device=device,
                image_resolution=512,
                loader_workers=8,
            )
            torch.cuda.synchronize(device)
            reserved = torch.cuda.max_memory_reserved(device)
            result.update(
                {
                    "status": "safe" if reserved <= total_memory * 0.90 else "unsafe_vram",
                    "model_seconds": float(output.timings.model_seconds),
                    "peak_reserved_bytes": int(reserved),
                    "peak_vram_fraction": float(reserved / total_memory),
                }
            )
            del output
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            result["status"] = "oom"
        trials.append(result)
        print(json.dumps({"event": "window_trial", **result}, sort_keys=True), flush=True)
    safe = [trial for trial in trials if trial["status"] == "safe"]
    if not safe:
        raise RuntimeError(f"no safe VGGT-Omega temporal window for {source.description.name}")
    selected = max(safe, key=lambda trial: trial["temporal_frames"])
    return {"trials": trials, "selected": selected}


def _write_scene(source, output_root, model, device, temporal_frames, infer_temporal_chunks):
    import numpy as np
    import torch

    description = source.description
    scene_root = output_root / description.name
    staging = output_root / f".{description.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    views = len(description.view_ids)
    frames = description.frame_count
    height, width = description.resolution_hw
    depth = np.lib.format.open_memmap(
        staging / "depth.npy", mode="w+", dtype=np.float32, shape=(views, frames, height, width)
    )
    mask = np.lib.format.open_memmap(
        staging / "cleaned_mask.npy", mode="w+", dtype=np.bool_, shape=(views, frames, height, width)
    )
    scales = np.lib.format.open_memmap(staging / "scales.npy", mode="w+", dtype=np.float32, shape=(frames,))
    intrinsics = np.lib.format.open_memmap(
        staging / "predicted_intrinsics.npy", mode="w+", dtype=np.float32, shape=(views, frames, 3, 3)
    )
    extrinsics = np.lib.format.open_memmap(
        staging / "predicted_extrinsics_w2c.npy", mode="w+", dtype=np.float32, shape=(views, frames, 4, 4)
    )
    records = []
    for start, end, owned_start, owned_end in _windows(frames, temporal_frames):
        result = infer_temporal_chunks(
            (source,),
            range(start, end),
            model,
            device=device,
            image_resolution=512,
            loader_workers=8,
        ).scenes[0]
        for frame in range(owned_start, owned_end):
            local = frame - start
            depth[:, frame] = result.depth[local]
            mask[:, frame] = result.cleaned_mask[local]
            scales[frame] = result.scale
            intrinsics[:, frame] = result.intrinsics[local]
            extrinsics[:, frame] = result.extrinsics_w2c[local]
        records.append(
            {
                "window_start": start,
                "window_end_exclusive": end,
                "owned_start": owned_start,
                "owned_end_exclusive": owned_end,
                "sequence_images": (end - start) * views,
                "scale": result.scale,
                "camera_center_alignment_rmse_m": result.alignment_residual,
            }
        )
        del result
        torch.cuda.empty_cache()
    for array in (depth, mask, scales, intrinsics, extrinsics):
        array.flush()
    manifest = {
        "format": "mvtracker_estimated_depth",
        "schema_version": 3,
        "layout": "per_view",
        "provider": "vggt_omega",
        "complete": True,
        "source_sequence": description.name,
        "source_fingerprint": description.source_fingerprint,
        "view_ids": list(description.view_ids),
        "frame_count": frames,
        "resolution_hw": [height, width],
        "checkpoint_revision": CHECKPOINT_REVISION,
        "preprocessing": {
            "mode": "balanced",
            "image_resolution": 512,
            "patch_size": 16,
            "sequence_order": "timestamp_major_views_minor",
            "temporal_window_size": temporal_frames,
            "overlap_policy": "max(8, floor(window/4)); central ownership",
        },
        "arrays": {
            f"{view}/depth.npy": {"shape": [frames, height, width], "dtype": "float32"}
            for view in description.view_ids
        },
        "window_records": records,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if scene_root.exists():
        shutil.rmtree(scene_root)
    os.replace(staging, scene_root)
    return manifest


def _emit(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def _save_burst(result, source, output_root: Path, start: int, end: int) -> dict:
    import numpy as np

    burst_root = output_root / source.description.name / f"frames-{start:06d}-{end:06d}"
    staging = output_root / source.description.name / f".frames-{start:06d}-{end:06d}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=False)
    np.save(staging / "depth.npy", np.asarray(result.depth, dtype=np.float32))
    np.save(staging / "cleaned_mask.npy", np.asarray(result.cleaned_mask, dtype=np.bool_))
    np.save(staging / "intrinsics.npy", np.asarray(result.intrinsics, dtype=np.float32))
    np.save(staging / "extrinsics_w2c.npy", np.asarray(result.extrinsics_w2c, dtype=np.float32))
    np.save(staging / "scale.npy", np.asarray(result.scale, dtype=np.float32))
    manifest = {
        "format": "mvtracker_vggt_omega_burst",
        "provider": "vggt_omega",
        "complete": True,
        "scene": source.description.name,
        "view_ids": list(source.description.view_ids),
        "frame_start": start,
        "frame_end_exclusive": end,
        "frame_count": end - start,
        "resolution_hw": list(source.description.resolution_hw),
        "depth_shape": list(result.depth.shape),
        "depth_dtype": "float32",
        "cleaned_mask_dtype": "bool",
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if burst_root.exists():
        shutil.rmtree(burst_root)
    os.replace(staging, burst_root)
    return manifest


@app.function(
    image=image,
    gpu="H100!",
    secrets=[hf_secret, wandb_secret],
    volumes={str(DATA_ROOT): data_volume.with_mount_options(read_only=True), str(RUN_ROOT): run_volume},
    cpu=8,
    memory=65536,
    ephemeral_disk=512 * 1024,
    timeout=15 * 60,
    max_containers=1,
    include_source=False,
)
def burst(
    run_name: str = "vggt-omega-bursts",
    dataset: str = "diegesis",
    window_frames: int = 24,
    start_frame: int = 0,
    batch_size: int = 1,
) -> dict:
    """Run and persist one small bounded burst per selected scene group."""
    import torch
    import wandb
    from huggingface_hub import hf_hub_download
    from mvtracker.preprocessing.vggt_omega import infer_temporal_chunks, load_model

    if dataset not in SCENES:
        raise ValueError(f"unsupported dataset {dataset!r}")
    if window_frames not in {24, 48, 96}:
        raise ValueError("window_frames must be one of 24, 48, 96")
    if batch_size not in {1, 2}:
        raise ValueError("batch_size must be 1 or 2")
    run = wandb.init(
        project="mvtracker-modal-profiling",
        job_type="vggt-omega-burst",
        name=run_name,
        tags=["modal", "h100", "vggt-omega", "burst"],
        config={**BASE_TAGS, "dataset": dataset, "window_frames": window_frames, "batch_size": batch_size},
    )
    _emit("burst_start", run_name=run_name, dataset=dataset, window_frames=window_frames, start_frame=start_frame, batch_size=batch_size)
    checkpoint = Path(
        hf_hub_download(
            repo_id=CHECKPOINT_REPO,
            filename=CHECKPOINT_FILENAME,
            revision=CHECKPOINT_REVISION,
            token=os.environ["HF_TOKEN"],
            local_dir="/tmp/vggt-omega-checkpoint",
        )
    )
    device = torch.device("cuda")
    model = load_model(checkpoint, device)
    sources = _sources(DATA_ROOT)[dataset]
    output_root = RUN_ROOT / run_name / "bursts" / dataset
    results = []
    try:
        for group_start in range(0, len(sources), batch_size):
            group = tuple(sources[group_start : group_start + batch_size])
            end = start_frame + window_frames
            if any(end > source.description.frame_count for source in group):
                raise ValueError(f"burst exceeds frame count for {[source.description.name for source in group]}")
            _emit(
                "burst_forward_start",
                dataset=dataset,
                scenes=[source.description.name for source in group],
                frame_start=start_frame,
                frame_end_exclusive=end,
                batch_size=len(group),
            )
            torch.cuda.reset_peak_memory_stats(device)
            result = infer_temporal_chunks(
                group,
                range(start_frame, end),
                model,
                device=device,
                image_resolution=512,
                loader_workers=8,
            )
            torch.cuda.synchronize(device)
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            _emit(
                "burst_forward_complete",
                dataset=dataset,
                scenes=[source.description.name for source in group],
                peak_reserved_bytes=peak_reserved,
                batch_size=len(group),
            )
            for source, scene_result in zip(group, result.scenes):
                _emit("burst_save_start", dataset=dataset, scene=source.description.name)
                manifest = _save_burst(scene_result, source, output_root, start_frame, end)
                _emit("burst_save_complete", dataset=dataset, scene=source.description.name, path=str(output_root / source.description.name / f"frames-{start_frame:06d}-{end:06d}"))
                results.append(manifest)
            run_volume.commit()
            _emit("burst_commit_complete", dataset=dataset, scenes=[source.description.name for source in group])
            run.log({"peak_vram_bytes": peak_reserved, "batch_size": len(group), "window_frames": window_frames})
    except Exception as error:
        _emit("burst_failed", dataset=dataset, error=repr(error))
        run.finish(exit_code=1)
        raise
    run.summary.update({"dataset": dataset, "burst_count": len(results), "artifacts": results})
    run.finish()
    _emit("burst_complete", dataset=dataset, burst_count=len(results))
    return {"format": "mvtracker_vggt_omega_burst_run", "dataset": dataset, "artifacts": results}


@app.function(
    image=image,
    gpu="H100!",
    secrets=[hf_secret, wandb_secret],
    volumes={str(DATA_ROOT): data_volume.with_mount_options(read_only=True), str(RUN_ROOT): run_volume},
    cpu=8,
    memory=65536,
    ephemeral_disk=512 * 1024,
    timeout=45 * 60,
    max_containers=1,
    include_source=False,
)
def infer(run_name: str = "vggt-omega-long") -> dict:
    import torch
    import wandb
    from huggingface_hub import hf_hub_download
    from mvtracker.preprocessing.vggt_omega import infer_temporal_chunks, load_model

    run = wandb.init(
        project="mvtracker-modal-profiling",
        job_type="vggt-omega-long",
        name=run_name,
        tags=["modal", "h100", "vggt-omega", "long-sequence"],
        config={**BASE_TAGS, "gpu": "H100", "temporal_candidates": [24, 48, 96, 192]},
    )
    checkpoint = Path(
        hf_hub_download(
            repo_id=CHECKPOINT_REPO,
            filename=CHECKPOINT_FILENAME,
            revision=CHECKPOINT_REVISION,
            token=os.environ["HF_TOKEN"],
            local_dir="/tmp/vggt-omega-checkpoint",
        )
    )
    device = torch.device("cuda")
    model = load_model(checkpoint, device)
    sources = _sources(DATA_ROOT)
    report = {"format": "mvtracker_vggt_omega_long", "run_name": run_name, "datasets": {}}
    output_root = RUN_ROOT / run_name / "estimated-depth"
    for dataset, dataset_sources in sources.items():
        source_profile = _probe(
            dataset_sources[0],
            model,
            device,
            [24, 48, 96],
            infer_temporal_chunks,
        )
        batch_profile = {
            "batch_1": _probe(
                dataset_sources[0], model, device, [24, 48], infer_temporal_chunks
            ),
            "batch_2": _probe(
                dataset_sources[0], model, device, [24, 48], infer_temporal_chunks,
                batch_sources=dataset_sources,
            )
            if len(dataset_sources) == 2
            else None,
        }
        selected_window = int(source_profile["selected"]["temporal_frames"])
        dataset_report = {
            "scene_names": [source.description.name for source in dataset_sources],
            "window_profile": source_profile,
            "batch_profile": batch_profile,
            "scenes": [],
        }
        for source in dataset_sources:
            manifest = _write_scene(
                source,
                output_root / dataset,
                model,
                device,
                min(selected_window, source.description.frame_count),
                infer_temporal_chunks,
            )
            dataset_report["scenes"].append(
                {
                    "scene": source.description.name,
                    "frame_count": source.description.frame_count,
                    "window_count": len(manifest["window_records"]),
                }
            )
        report["datasets"][dataset] = dataset_report
        run.log(
            {
                f"{dataset}/selected_window_frames": selected_window,
                f"{dataset}/scene_count": len(dataset_sources),
            }
        )
    report_path = RUN_ROOT / run_name / "inference-report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    run.summary.update({"report_path": str(report_path), "note": "only model profiling is timed"})
    run.finish()
    run_volume.commit()
    return report


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=15 * 60,
    volumes={str(RUN_ROOT): run_volume.with_mount_options(read_only=True)},
    include_source=False,
)
def readback(run_name: str = "vggt-omega-long") -> dict:
    import numpy as np
    from mvtracker.datasets.estimated_depth import EstimatedDepthStore

    root = RUN_ROOT / run_name / "estimated-depth"
    report = {"format": "mvtracker_vggt_omega_readback", "datasets": {}}
    for dataset, scenes in SCENES.items():
        dataset_root = root / dataset
        entries = []
        for scene in scenes:
            store = EstimatedDepthStore(dataset_root, "vggt_omega")
            frame_count = int(json.loads((dataset_root / scene / "manifest.json").read_text())["frame_count"])
            frame_indices = list(range(min(24, frame_count)))
            started = time.perf_counter()
            depth, mask = store.load(scene, list(range(4 if dataset != "mv-kubric" else 10)), frame_indices)
            copied = np.asarray(depth, dtype=np.float32).copy()
            copied_mask = np.asarray(mask, dtype=np.bool_).copy()
            entries.append(
                {
                    "scene": scene,
                    "frames": len(frame_indices),
                    "read_seconds": time.perf_counter() - started,
                    "depth_shape": list(copied.shape),
                    "mask_shape": list(copied_mask.shape),
                    "depth_dtype": str(copied.dtype),
                }
            )
        report["datasets"][dataset] = entries
    return report


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=10 * 60,
    volumes={str(RUN_ROOT): run_volume.with_mount_options(read_only=True)},
    include_source=False,
)
def burst_readback(run_name: str = "vggt-omega-bursts", dataset: str = "diegesis") -> dict:
    import numpy as np

    root = RUN_ROOT / run_name / "bursts" / dataset
    if not root.is_dir():
        raise FileNotFoundError(root)
    report = {"format": "mvtracker_vggt_omega_burst_readback", "dataset": dataset, "scenes": []}
    for scene_root in sorted(path for path in root.iterdir() if path.is_dir()):
        for burst_root in sorted(path for path in scene_root.iterdir() if path.is_dir()):
            _emit("burst_read_start", dataset=dataset, scene=scene_root.name, path=str(burst_root))
            started = time.perf_counter()
            depth = np.load(burst_root / "depth.npy", mmap_mode="r", allow_pickle=False)
            mask = np.load(burst_root / "cleaned_mask.npy", mmap_mode="r", allow_pickle=False)
            selected_depth = np.asarray(depth, dtype=np.float32).copy()
            selected_mask = np.asarray(mask, dtype=np.bool_).copy()
            elapsed = time.perf_counter() - started
            entry = {
                "scene": scene_root.name,
                "burst": burst_root.name,
                "read_seconds": elapsed,
                "depth_shape": list(selected_depth.shape),
                "mask_shape": list(selected_mask.shape),
                "depth_dtype": str(selected_depth.dtype),
                "mask_dtype": str(selected_mask.dtype),
            }
            report["scenes"].append(entry)
            _emit("burst_read_complete", **entry)
    return report


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=20 * 60,
    volumes={str(DATA_ROOT): data_volume.with_mount_options(read_only=True), str(RUN_ROOT): run_volume},
    include_source=False,
)
def pack_mvkubric(run_name: str = "vggt-omega-long") -> dict:
    import importlib.util
    import shutil as file_shutil
    import subprocess
    import sys
    from mvtracker.preprocessing.mvkubric_webdataset import (
        SceneShard,
        finalize_shards,
        write_shard,
    )

    output_root = RUN_ROOT / run_name / "mvkubric-vggt-webdataset" / "train"
    output_root.mkdir(parents=True, exist_ok=True)
    source_root = DATA_ROOT / "datasets/kubric-multiview/train"
    estimated_root = RUN_ROOT / run_name / "estimated-depth" / "mv-kubric"
    shard = write_shard(
        source_root,
        SceneShard(name="mvkubric-vggt-00000", scene_ids=SCENES["mv-kubric"], index=0),
        output_root / "mvkubric-vggt-00000.tar",
        read_workers=8,
        estimated_depth_root=estimated_root,
        estimated_depth_cleaned=True,
    )
    index_path = output_root / "mvkubric-vggt-00000.idx"
    dali_tool = file_shutil.which("wds2idx")
    if dali_tool is None:
        spec = importlib.util.find_spec("nvidia.dali")
        if spec is None or spec.origin is None:
            raise RuntimeError("DALI wds2idx is not installed")
        package_root = Path(spec.origin).parent
        for candidate in (package_root / "tools/wds2idx.py", package_root.parent / "tools/wds2idx.py"):
            if candidate.is_file():
                dali_tool = str(candidate)
                break
    if dali_tool is None:
        raise RuntimeError("DALI wds2idx executable is unavailable")
    partial = index_path.with_suffix(".idx.partial")
    command = [dali_tool] if dali_tool.endswith("wds2idx") else [sys.executable, dali_tool]
    subprocess.run([*command, str(output_root / "mvkubric-vggt-00000.tar"), str(partial)], check=True)
    partial.replace(index_path)
    manifest = finalize_shards(output_root, SCENES["mv-kubric"], scenes_per_shard=2)
    run_volume.commit()
    return {"shard": shard, "manifest": manifest}


@app.function(
    image=image,
    gpu="T4",
    secrets=[wandb_secret],
    volumes={str(RUN_ROOT): run_volume.with_mount_options(read_only=True)},
    cpu=8,
    memory=32768,
    timeout=15 * 60,
    include_source=False,
)
def dali_readback(run_name: str = "vggt-omega-long") -> dict:
    import torch
    from mvtracker.datasets.kubric_dali_stream import KubricDaliSceneStream
    from mvtracker.datasets.tapvid3d_multiview_dataset import DaliEncodedImageDecoder

    manifest = RUN_ROOT / run_name / "mvkubric-vggt-webdataset" / "train" / "manifest.json"
    stream = KubricDaliSceneStream(
        manifest,
        rank=0,
        world_size=1,
        seed=0,
        scenes_per_batch=2,
        repeat=False,
        shuffle_shards=False,
    )
    group = stream.next_scene_group()
    decoder = DaliEncodedImageDecoder(torch.device("cuda"), max_encoded_images=512)
    rgb = [payload for scene in group.scenes for payload in scene.rgb_npz]
    depth = [payload for scene in group.scenes for payload in scene.depth_npz]
    started = time.perf_counter()
    decoded_rgb, decoded_depth = decoder.decode(
        [frame for payload in rgb for frame in _packed_frames(payload)],
        [frame for payload in depth for frame in _packed_frames(payload)],
    )
    torch.cuda.synchronize()
    return {
        "format": "mvtracker_vggt_omega_dali_readback",
        "dali_reader_seconds": group.read_seconds,
        "decode_seconds": time.perf_counter() - started,
        "rgb_frames": len(decoded_rgb),
        "depth_frames": len(decoded_depth),
        "depth_dtype": str(decoded_depth[0].dtype),
    }


def _packed_frames(payload: bytes) -> tuple[bytes, ...]:
    import io
    import numpy as np

    with np.load(io.BytesIO(payload), allow_pickle=False) as packed:
        encoded = np.asarray(packed["bytes"], dtype=np.uint8).reshape(-1)
        offsets = np.asarray(packed["offsets"], dtype=np.int64).reshape(-1)
    return tuple(bytes(encoded[start:end]) for start, end in zip(offsets[:-1], offsets[1:]))


if __name__ == "__main__":
    print("Use: modal run --timestamps tools/modal_vggt_omega_long_inference.py::infer")
