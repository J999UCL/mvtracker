"""Recipe-driven DA3-Giant depth production on a dedicated local GPU."""

from __future__ import annotations

import argparse
from io import BytesIO
import json
import os
from pathlib import Path
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
from mvtracker.datasets.training_recipe import RecipeReader, RecipeRecord
from mvtracker.preprocessing.mvkubric_webdataset import META_COMPONENT, RGB_COMPONENT


MODEL_ID = "depth-anything/DA3-GIANT-1.1"
IMAGE_CAPACITY = 80
MIN_ALIGNMENT_VIEWS = 4


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
                / "datasets/diegesis-mvtracker/TAPVid3D_MVTracker_cache/train"
                / scene
            )
            cameras = self.data_root / "source/diegesis/scenes" / scene / "tracking/sequence"
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
        self.metadata = {}

    def _scene(self, name: str):
        if name not in self.metadata:
            entry = self.catalog.scenes[name]
            record, _ = self.records.read(int(entry["metadata_index"]))
            self.metadata[name] = _scene_metadata(
                KubricDaliSceneBundle(name, record[f".{META_COMPONENT}"], (), ())
            )
        return self.metadata[name]

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


def _write_sample(root: Path, record: RecipeRecord, depth: np.ndarray, mask: np.ndarray) -> None:
    sample_root = root / f"step-{record.step:06d}" / f"sample-{record.logical_index:02d}"
    staging = sample_root.with_name(f".{sample_root.name}.partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    np.save(staging / "depth.npy", depth.astype(np.float32, copy=False), allow_pickle=False)
    np.save(staging / "cleaned_mask.npy", mask.astype(np.bool_, copy=False), allow_pickle=False)
    staging.rename(sample_root)
    (sample_root / "ready").touch()


def _produce_record(model, reader, record: RecipeRecord, output_root: Path):
    import torch
    import torch.nn.functional as F
    from depth_anything_3.utils.pose_align import align_poses_umeyama

    inference_views, images, extrinsics, intrinsics, source_size = reader.load(record)
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


def run(recipe_path: Path, data_root: Path, output_root: Path, prefill_steps: int) -> None:
    import torch
    from depth_anything_3.api import DepthAnything3

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    metrics_path = output_root / "metrics.jsonl"
    started = time.perf_counter()
    generated_samples = 0
    generated_images = 0
    model_seconds = 0.0
    try:
        _log("model_load_started", model=MODEL_ID)
        model = DepthAnything3.from_pretrained(MODEL_ID).to("cuda").eval()
        torch.cuda.synchronize()
        _log(
            "model_ready",
            seconds=round(time.perf_counter() - started, 3),
            gpu=torch.cuda.get_device_name(0),
        )
        packed = _PackedScenes(data_root)
        mvkubric = _MVKubricScenes(data_root)
        for payload in RecipeReader(recipe_path).steps():
            step = int(payload["step"])
            for sample in payload["logical_samples"]:
                record = RecipeRecord.from_dict(sample)
                if record.depth_source == "gt":
                    continue
                reader = mvkubric if record.source == "mvkubric" else packed
                sample_started = time.perf_counter()
                seconds, image_count = _produce_record(model, reader, record, output_root)
                generated_samples += 1
                generated_images += image_count
                model_seconds += seconds
                metric = {
                    "step": step,
                    "source": record.source,
                    "scene": record.scene,
                    "depth_source": record.depth_source,
                    "generated_samples": generated_samples,
                    "generated_images": generated_images,
                    "model_seconds": model_seconds,
                    "model_images_per_second": generated_images / max(model_seconds, 1e-9),
                    "sample_seconds": time.perf_counter() - sample_started,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metric, separators=(",", ":")) + "\n")
                _log("sample_ready", **metric)
            (output_root / f"step-{step:06d}.ready").touch()
            if step + 1 == prefill_steps:
                (output_root / "prefill.ready").touch()
                _log("prefill_ready", steps=prefill_steps)
        (output_root / "complete").touch()
        _log(
            "producer_complete",
            generated_samples=generated_samples,
            generated_images=generated_images,
            model_images_per_second=generated_images / max(model_seconds, 1e-9),
            wall_seconds=time.perf_counter() - started,
        )
    except BaseException as error:
        (output_root / "failed").write_text(f"DA3 producer failed: {error}\n", encoding="utf-8")
        _log("producer_failed", error=repr(error))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prefill-steps", type=int, default=4)
    args = parser.parse_args()
    run(args.recipe, args.data_root, args.output_root, args.prefill_steps)


if __name__ == "__main__":
    main()
