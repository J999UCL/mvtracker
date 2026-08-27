"""Recipe-driven DA3-Giant depth production on local GPUs."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import islice
import json
import os
from pathlib import Path
import resource
import shutil
import time

import numpy as np
from PIL import Image

from mvtracker.datasets.kubric_dali_dataset import (
    KubricWebDatasetCatalog,
    _IndexedRecordStore,
    _packed_frames,
    _scene_metadata,
)
from mvtracker.datasets.kubric_dali_stream import KubricDaliSceneBundle
from mvtracker.datasets.io_cache import discard_file_range, flush_and_discard_file
from mvtracker.datasets.training_recipe import RecipeReader, RecipeRecord
from mvtracker.preprocessing.mvkubric_webdataset import META_COMPONENT, RGB_COMPONENT


MODEL_ID = "depth-anything/DA3-GIANT-1.1"
IMAGE_CAPACITY = int(os.environ.get("MVTRACKER_DA3_IMAGE_CAPACITY", "80"))
MIN_ALIGNMENT_VIEWS = 4
MAX_PENDING_SAMPLES = 32
PREFILL_SAMPLES = 64


def _log(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def _intrinsics_matrix(values: np.ndarray, frame: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape == (3, 3):
        return values
    if values.shape != (4,):
        return _intrinsics_matrix(values[frame], 0)
    fx, fy, cx, cy = values
    return np.asarray(((fx, 0.0, cx), (0.0, fy, cy), (0.0, 0.0, 1.0)), dtype=np.float32)


def _decode(encoded: bytes) -> Image.Image:
    with Image.open(BytesIO(encoded)) as image:
        return image.convert("RGB")


def _homogeneous_extrinsics(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.shape[-2:] == (4, 4):
        return values
    bottom = np.zeros((*values.shape[:-2], 1, 4), dtype=np.float32)
    bottom[..., 0, 3] = 1.0
    return np.concatenate((values, bottom), axis=-2)


def _inference_views(selected: tuple[int, ...], available: list[int]) -> list[int]:
    views = list(selected)
    views.extend(view for view in available if view not in views)
    return views[: max(len(selected), MIN_ALIGNMENT_VIEWS)]


class _PackedScenes:
    def __init__(self, data_root: Path):
        self.data_root = data_root

    def _roots(self, source: str, scene: str) -> tuple[Path, Path]:
        if source == "diegesis":
            cache = (
                self.data_root
                / "datasets/diegesis-train/TAPVid3D_MVTracker_cache/train"
                / scene
            )
            cameras = (
                self.data_root
                / "datasets/diegesis-train/TAPVid3D_raw/train"
                / scene
            )
            return cache, cameras
        root = self.data_root / "datasets/syn4d-mvtracker/train" / scene
        return root, root

    def load(self, record: RecipeRecord):
        cache_root, camera_root = self._roots(record.source, record.scene)
        available = sorted(
            int(path.name.removeprefix("view_"))
            for path in cache_root.glob("view_*")
            if path.is_dir()
        )
        inference_views = _inference_views(record.views, available)
        frames = list(record.frames)
        encoded_by_view = {}
        extrinsics = []
        intrinsics = []
        for view in inference_views:
            view_root = cache_root / f"view_{view}"
            offsets = np.load(view_root / "jpeg_offsets.npy", mmap_mode="r", allow_pickle=False)
            descriptor = os.open(view_root / "jpeg_bytes.bin", os.O_RDONLY)
            try:
                encoded_by_view[view] = [
                    os.pread(
                        descriptor,
                        int(offsets[frame + 1] - offsets[frame]),
                        int(offsets[frame]),
                    )
                    for frame in frames
                ]
                start = int(offsets[frames[0]])
                stop = int(offsets[frames[-1] + 1])
                discard_file_range(descriptor, start, stop - start)
            finally:
                os.close(descriptor)
            exts = np.load(camera_root / str(view) / "extrinsics_w2c.npy", mmap_mode="r")
            ints = np.load(camera_root / str(view) / "intrinsics.npy", mmap_mode="r")
            extrinsics.append(np.stack([exts[frame] for frame in frames]))
            intrinsics.append(np.stack([_intrinsics_matrix(ints, frame) for frame in frames]))
        first = _decode(encoded_by_view[inference_views[0]][0])
        return (
            inference_views,
            [[_decode(encoded_by_view[view][position]) for view in inference_views] for position in range(len(frames))],
            _homogeneous_extrinsics(np.stack(extrinsics, axis=1)),
            np.stack(intrinsics, axis=1).astype(np.float32),
            (first.height, first.width),
        )


class _MVKubricScenes:
    def __init__(self, data_root: Path):
        root = data_root / "datasets/kubric-multiview-webdataset/train"
        self.catalog = KubricWebDatasetCatalog(root / "manifest.json")
        self.records = _IndexedRecordStore(self.catalog.record_locator_path)

    def _scene(self, name: str):
        entry = self.catalog.scenes[name]
        record, _ = self.records.read(int(entry["metadata_index"]))
        return _scene_metadata(
            KubricDaliSceneBundle(name, record[f".{META_COMPONENT}"], (), ())
        )

    def load(self, record: RecipeRecord):
        scene = self._scene(record.scene)
        entry = self.catalog.scenes[record.scene]
        available = sorted(int(view) for view in entry["views"])
        inference_views = _inference_views(record.views, available)
        indices = [int(entry["views"][str(view)]["media_index"]) for view in inference_views]
        media, _ = self.records.read_many(indices, components={RGB_COMPONENT})
        frames = list(record.frames)
        encoded_by_view = {
            view: _packed_frames(item[f".{RGB_COMPONENT}"])
            for view, item in zip(inference_views, media, strict=True)
        }
        images = [
            [_decode(encoded_by_view[view][frame]) for view in inference_views]
            for frame in frames
        ]
        extrinsics = _homogeneous_extrinsics(
            np.stack(
                [scene.extrinsics[view, frames] for view in inference_views],
                axis=1,
            )
        )
        intrinsics = np.stack(
            [
                np.repeat(scene.intrinsics[view][None], len(frames), axis=0)
                for view in inference_views
            ],
            axis=1,
        ).astype(np.float32)
        return inference_views, images, extrinsics, intrinsics, scene.resolution_hw


def _sample_root(root: Path, record: RecipeRecord) -> Path:
    return root / f"step-{record.step:06d}" / f"sample-{record.logical_index:02d}"


def _write_sample(root: Path, record: RecipeRecord, depth: np.ndarray, mask: np.ndarray) -> None:
    sample_root = _sample_root(root, record)
    staging = sample_root.with_name(f".{sample_root.name}.partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    depth_path = staging / "depth.npy"
    mask_path = staging / "cleaned_mask.npy"
    np.save(depth_path, depth.astype(np.float32, copy=False), allow_pickle=False)
    np.save(mask_path, mask.astype(np.bool_, copy=False), allow_pickle=False)
    flush_and_discard_file(depth_path)
    flush_and_discard_file(mask_path)
    staging.rename(sample_root)
    (sample_root / "ready").touch()


def _wait_for_consumer(output_root: Path, max_pending_samples: int) -> None:
    started = time.perf_counter()
    last_log = started
    while True:
        pending = sum(1 for _ in output_root.glob("step-*/sample-*/ready"))
        if pending < max_pending_samples:
            return
        now = time.perf_counter()
        if now - last_log >= 10:
            _log(
                "consumer_backpressure",
                pending_samples=pending,
                max_pending_samples=max_pending_samples,
                elapsed_seconds=round(now - started, 1),
            )
            last_log = now
        time.sleep(0.1)


def _depth_records(recipe_path: Path, max_steps: int | None = None):
    """Yield non-GT recipe records in their canonical execution order."""
    ordinal = 0
    for payload in RecipeReader(recipe_path).steps():
        if max_steps is not None and int(payload["step"]) >= int(max_steps):
            break
        for sample in payload["logical_samples"]:
            record = RecipeRecord.from_dict(sample)
            if record.depth_source == "gt":
                continue
            yield ordinal, record
            ordinal += 1


def _prefill_owner(ordinal: int, worker_count: int) -> int:
    if worker_count < 1:
        raise ValueError("worker_count must be positive")
    return int(ordinal) % int(worker_count)


def prefill_ready_paths(
    recipe_path: Path,
    output_root: Path,
    prefill_samples: int,
) -> tuple[Path, ...]:
    records = list(islice(_depth_records(recipe_path), prefill_samples))
    if len(records) != prefill_samples:
        raise ValueError(
            f"recipe contains {len(records)} non-GT samples; expected at least {prefill_samples}"
        )
    return tuple(_sample_root(output_root, record) / "ready" for _, record in records)


def _wait_for_handoff(output_root: Path, worker_id: int) -> None:
    started = time.perf_counter()
    last_log = started
    handoff = output_root / "prefill.ready"
    while not handoff.is_file():
        failed = output_root / "failed"
        if failed.is_file():
            raise RuntimeError(failed.read_text(encoding="utf-8").strip())
        now = time.perf_counter()
        if now - last_log >= 10:
            _log(
                "prefill_handoff_wait",
                worker_id=worker_id,
                elapsed_seconds=round(now - started, 1),
            )
            last_log = now
        time.sleep(0.1)


def _write_metric(metrics_path: Path, latest_path: Path, metric: dict) -> None:
    encoded = json.dumps(metric, separators=(",", ":"))
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(encoded + "\n")
    staging = latest_path.with_suffix(".partial")
    staging.write_text(encoded + "\n", encoding="utf-8")
    staging.replace(latest_path)


def _produce_record(
    model,
    reader,
    record: RecipeRecord,
    output_root: Path,
    *,
    loaded=None,
):
    import torch
    import torch.nn.functional as F
    from depth_anything_3.utils.pose_align import align_poses_umeyama

    if loaded is None:
        loaded = reader.load(record)
    inference_views, images, extrinsics, intrinsics, source_size = loaded
    selected_positions = [inference_views.index(view) for view in record.views]
    timestamp_batch = min(len(record.frames), max(1, IMAGE_CAPACITY // len(inference_views)))
    depth = np.empty((len(record.views), len(record.frames), *source_size), dtype=np.float32)
    cleaned = np.empty_like(depth, dtype=np.bool_)
    model_seconds = 0.0
    for start in range(0, len(record.frames), timestamp_batch):
        stop = min(start + timestamp_batch, len(record.frames))
        processed_images = []
        processed_extrinsics = []
        processed_intrinsics = []
        for timestamp in range(start, stop):
            imgs, exts, ints = model._preprocess_inputs(
                images[timestamp],
                extrinsics[timestamp],
                intrinsics[timestamp],
                process_res=504,
                process_res_method="upper_bound_resize",
            )
            processed_images.append(imgs)
            processed_extrinsics.append(exts)
            processed_intrinsics.append(ints)
        batch_images = torch.stack(processed_images).to("cuda", non_blocking=True).float()
        batch_exts = torch.stack(processed_extrinsics).to("cuda", non_blocking=True).float()
        batch_ints = torch.stack(processed_intrinsics).to("cuda", non_blocking=True).float()
        normalized_exts = torch.cat(
            [model._normalize_extrinsics(batch_exts[index : index + 1].clone()) for index in range(stop - start)]
        )
        torch.cuda.synchronize()
        model_started = time.perf_counter()
        with torch.inference_mode():
            result = model.forward(
                batch_images,
                normalized_exts,
                batch_ints,
                export_feat_layers=[],
            )
        torch.cuda.synchronize()
        model_seconds += time.perf_counter() - model_started
        predicted_depth = result["depth"].float().cpu().numpy()
        predicted_conf = result["depth_conf"].float().cpu().numpy()
        predicted_poses = result["extrinsics"].float().cpu().numpy()
        for offset in range(stop - start):
            _, _, scale, _ = align_poses_umeyama(
                predicted_poses[offset],
                processed_extrinsics[offset].numpy(),
                ransac=False,
                return_aligned=True,
            )
            predicted_depth[offset] /= scale
        predicted_depth = F.interpolate(
            torch.from_numpy(predicted_depth.reshape(-1, *predicted_depth.shape[-2:])).unsqueeze(1),
            size=source_size,
            mode="bilinear",
            align_corners=False,
        )[:, 0].reshape(stop - start, len(inference_views), *source_size).numpy()
        predicted_conf = F.interpolate(
            torch.from_numpy(predicted_conf.reshape(-1, *predicted_conf.shape[-2:])).unsqueeze(1),
            size=source_size,
            mode="bilinear",
            align_corners=False,
        )[:, 0].reshape(stop - start, len(inference_views), *source_size).numpy()
        for offset in range(stop - start):
            selected_depth = predicted_depth[offset, selected_positions]
            selected_conf = predicted_conf[offset, selected_positions]
            depth[:, start + offset] = selected_depth
            cleaned[:, start + offset] = selected_conf >= np.quantile(selected_conf, 0.40)
        del result, batch_images, batch_exts, batch_ints, normalized_exts
    _write_sample(output_root, record, depth, cleaned)
    return model_seconds, len(record.frames) * len(inference_views)


def run(
    recipe_path: Path,
    data_root: Path,
    output_root: Path,
    max_pending_samples: int,
    *,
    worker_id: int = 0,
    worker_count: int = 1,
    prefill_samples: int = PREFILL_SAMPLES,
    continue_after_prefill: bool = True,
    max_steps: int | None = None,
) -> None:
    import torch
    from depth_anything_3.api import DepthAnything3

    if not output_root.is_dir():
        raise FileNotFoundError(
            f"runtime depth root must be initialized by the launcher: {output_root}"
        )
    if worker_id < 0 or worker_id >= worker_count:
        raise ValueError(
            f"worker_id {worker_id} is outside worker_count {worker_count}"
        )
    if prefill_samples < 1:
        raise ValueError("prefill_samples must be positive")
    if max_pending_samples < 1:
        raise ValueError("max_pending_samples must be positive")
    metrics_path = output_root / f"metrics-worker-{worker_id}.jsonl"
    latest_metrics_path = output_root / f"latest-metrics-worker-{worker_id}.json"
    started = time.perf_counter()
    generated_samples = 0
    generated_images = 0
    model_seconds = 0.0
    try:
        _log(
            "model_load_started",
            model=MODEL_ID,
            worker_id=worker_id,
            worker_count=worker_count,
        )
        model = DepthAnything3.from_pretrained(MODEL_ID).to("cuda").eval()
        torch.cuda.synchronize()
        _log(
            "model_ready",
            seconds=round(time.perf_counter() - started, 3),
            gpu=torch.cuda.get_device_name(0),
            worker_id=worker_id,
        )
        packed = _PackedScenes(data_root)
        mvkubric = _MVKubricScenes(data_root)
        selected_records = (
            (ordinal, record)
            for ordinal, record in _depth_records(recipe_path, max_steps=max_steps)
            if (
                _prefill_owner(ordinal, worker_count) == worker_id
                if ordinal < prefill_samples
                else continue_after_prefill
            )
        )
        handoff_started = False

        def load_next():
            nonlocal handoff_started
            try:
                ordinal, record = next(selected_records)
            except StopIteration:
                return None
            if ordinal >= prefill_samples and not handoff_started:
                (output_root / f"worker-{worker_id}.prefill.ready").touch()
                _log(
                    "prefill_shard_ready",
                    worker_id=worker_id,
                    generated_samples=generated_samples,
                    prefill_samples=prefill_samples,
                )
                _wait_for_handoff(output_root, worker_id)
                handoff_started = True
            pending_limit = (
                max(max_pending_samples, prefill_samples)
                if ordinal < prefill_samples
                else max_pending_samples
            )
            _wait_for_consumer(output_root, pending_limit)
            reader = mvkubric if record.source == "mvkubric" else packed
            sample_started = time.perf_counter()
            loaded = reader.load(record)
            return (
                ordinal,
                record,
                reader,
                loaded,
                sample_started,
                time.perf_counter() - sample_started,
            )

        with ThreadPoolExecutor(max_workers=1) as loader:
            loaded_future = loader.submit(load_next)
            while True:
                item = loaded_future.result()
                if item is None:
                    break
                ordinal, record, reader, loaded, sample_started, load_seconds = item
                last_prefill_record = (
                    ordinal < prefill_samples
                    and ordinal + worker_count >= prefill_samples
                )
                if not last_prefill_record:
                    loaded_future = loader.submit(load_next)
                seconds, image_count = _produce_record(
                    model,
                    reader,
                    record,
                    output_root,
                    loaded=loaded,
                )
                generated_samples += 1
                generated_images += image_count
                model_seconds += seconds
                metric = {
                    "ordinal": ordinal,
                    "step": record.step,
                    "logical_index": record.logical_index,
                    "source": record.source,
                    "scene": record.scene,
                    "depth_source": record.depth_source,
                    "worker_id": worker_id,
                    "generated_samples": generated_samples,
                    "generated_images": generated_images,
                    "model_seconds": model_seconds,
                    "model_images_per_second": generated_images
                    / max(model_seconds, 1e-9),
                    "load_seconds": load_seconds,
                    "sample_seconds": time.perf_counter() - sample_started,
                    "pending_ready_samples": sum(
                        1 for _ in output_root.glob("step-*/sample-*/ready")
                    ),
                    "producer_max_rss_gib": (
                        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        / 1024**2
                    ),
                }
                _write_metric(metrics_path, latest_metrics_path, metric)
                _log("sample_ready", **metric)
                if last_prefill_record:
                    loaded_future = loader.submit(load_next)

        shard_ready = output_root / f"worker-{worker_id}.prefill.ready"
        if not shard_ready.is_file():
            if generated_samples < sum(
                1
                for ordinal in range(prefill_samples)
                if _prefill_owner(ordinal, worker_count) == worker_id
            ):
                raise ValueError(
                    f"recipe contains fewer than {prefill_samples} non-GT samples"
                )
            shard_ready.touch()
            _log(
                "prefill_shard_ready",
                worker_id=worker_id,
                generated_samples=generated_samples,
                prefill_samples=prefill_samples,
            )
        (output_root / f"worker-{worker_id}.complete").touch()
        if continue_after_prefill:
            (output_root / "complete").touch()
        _log(
            "producer_complete",
            worker_id=worker_id,
            generated_samples=generated_samples,
            generated_images=generated_images,
            model_images_per_second=generated_images / max(model_seconds, 1e-9),
            wall_seconds=time.perf_counter() - started,
        )
    except BaseException as error:
        (output_root / "failed").write_text(
            f"DA3 worker {worker_id} failed: {error}\n", encoding="utf-8"
        )
        _log("producer_failed", worker_id=worker_id, error=repr(error))
        raise


def download_model() -> None:
    from huggingface_hub import snapshot_download

    _log("model_cache_warm_started", model=MODEL_ID)
    snapshot_download(MODEL_ID)
    _log("model_cache_warm_complete", model=MODEL_ID)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-pending-samples", type=int, default=MAX_PENDING_SAMPLES)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--prefill-samples", type=int, default=PREFILL_SAMPLES)
    parser.add_argument("--prefill-only", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    if args.download_only:
        download_model()
        return
    if args.recipe is None or args.data_root is None or args.output_root is None:
        parser.error("--recipe, --data-root, and --output-root are required")
    run(
        args.recipe,
        args.data_root,
        args.output_root,
        args.max_pending_samples,
        worker_id=args.worker_id,
        worker_count=args.worker_count,
        prefill_samples=args.prefill_samples,
        continue_after_prefill=not args.prefill_only,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
