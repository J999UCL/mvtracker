#!/usr/bin/env python3
"""Audit one VGGT-Omega sidecar against ground-truth depth and save QA artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from mvtracker.preprocessing.vggt_omega import (
    MVKubricSceneSource,
    SCHEMA_VERSION,
    TapVid3DSceneSource,
)
from mvtracker.preprocessing.vggt_omega_quality import (
    depth_quality_metrics,
    representative_frame_indices,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("tapvid3d", "mv-kubric"), required=True)
    parser.add_argument("--scene-root", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--views", type=lambda value: [int(item) for item in value.split(",")])
    parser.add_argument("--rerun-resolution", type=int, default=128)
    parser.add_argument("--no-rerun", action="store_true")
    return parser.parse_args()


def _read_tiff(path: Path) -> np.ndarray:
    import imageio.v2 as imageio

    depth = np.asarray(imageio.imread(path.read_bytes(), format="tiff"), dtype=np.float32)
    return depth[..., 0] if depth.ndim == 3 else depth


def _kubric_depth_to_z(depth: np.ndarray, metadata: dict) -> np.ndarray:
    height, width = depth.shape
    sensor_width = float(metadata["camera"]["sensor_width"])
    focal_length = float(metadata["camera"]["focal_length"])
    sensor_height = sensor_width / width * height
    x = (np.arange(-width / 2, width / 2, dtype=np.float32) + 0.5) / width * sensor_width
    y = (np.arange(-height / 2, height / 2, dtype=np.float32) + 0.5) / height * sensor_height
    xx, yy = np.meshgrid(x, y, indexing="xy")
    result = depth / np.sqrt(1 + (xx * xx + yy * yy) / (focal_length * focal_length))
    result[result > 1000] = 0
    return result.astype(np.float32)


def _load_source(dataset: str, scene_root: Path, view_ids: tuple[int, ...], frames: tuple[int, ...]):
    source_type = TapVid3DSceneSource if dataset == "tapvid3d" else MVKubricSceneSource
    source = source_type(scene_root, view_ids)
    rgbs = np.stack(
        [
            [np.asarray(source.load_rgb(view, frame), dtype=np.uint8) for frame in frames]
            for view in view_ids
        ]
    )
    known_w2c = np.moveaxis(source.extrinsics_w2c(frames), 0, 1)
    ground_truth = []
    intrinsics = []
    for view in view_ids:
        view_depths = []
        if dataset == "tapvid3d":
            view_root = scene_root / str(view)
            all_depth = np.load(view_root / "depth.npy", mmap_mode="r")
            view_depths = [np.asarray(all_depth[frame], dtype=np.float32) for frame in frames]
            fx, fy, cx, cy = np.load(view_root / "intrinsics.npy")
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        else:
            view_root = scene_root / f"view_{view}"
            metadata = json.loads((view_root / "metadata.json").read_text(encoding="utf-8"))
            depth_paths = sorted(view_root.glob("depth_*.tiff"))
            if not depth_paths:
                depth_paths = sorted(view_root.glob("depth_*.tif"))
            view_depths = [_kubric_depth_to_z(_read_tiff(depth_paths[frame]), metadata) for frame in frames]
            width, height = metadata["metadata"]["resolution"]
            K = (
                np.diag([width, height, 1.0])
                @ np.asarray(metadata["camera"]["K"], dtype=np.float64)
                @ np.diag([1.0, -1.0, -1.0])
            ).astype(np.float32)
        ground_truth.append(view_depths)
        intrinsics.append(np.repeat(K[None], len(frames), axis=0))
    return (
        rgbs,
        np.asarray(ground_truth, dtype=np.float32),
        np.asarray(intrinsics, dtype=np.float32),
        known_w2c.astype(np.float32),
    )


def _tile(images: np.ndarray) -> np.ndarray:
    count, height, width = images.shape[:3]
    columns = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / columns))
    tail = images.shape[3:]
    canvas = np.zeros((rows * height, columns * width, *tail), dtype=images.dtype)
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        canvas[row * height : (row + 1) * height, column * width : (column + 1) * width] = image
    return canvas


def _save_contact_sheet(
    path: Path,
    frames: tuple[int, ...],
    rgbs: np.ndarray,
    ground_truth: np.ndarray,
    estimated: np.ndarray,
    cleaned: np.ndarray,
) -> None:
    from matplotlib import pyplot as plt

    figure, axes = plt.subplots(len(frames), 4, figsize=(16, 4 * len(frames)), squeeze=False)
    for column, title in enumerate(("RGB", "GT depth", "VGGT-Omega depth", "absolute error")):
        axes[0, column].set_title(title)
    for timestamp, frame in enumerate(frames):
        gt = ground_truth[:, timestamp]
        estimate = estimated[:, timestamp]
        valid = np.isfinite(gt) & (gt > 0) & np.isfinite(estimate) & (estimate > 0)
        values = gt[np.isfinite(gt) & (gt > 0)]
        minimum, maximum = np.percentile(values, (2, 98))
        error = np.where(valid & cleaned[:, timestamp], np.abs(estimate - gt), np.nan)
        panels = (
            (_tile(rgbs[:, timestamp]), None, None),
            (_tile(gt), "magma", (minimum, maximum)),
            (_tile(np.where(cleaned[:, timestamp], estimate, np.nan)), "magma", (minimum, maximum)),
            (_tile(error), "inferno", (0, np.nanpercentile(error, 98))),
        )
        for column, (panel, cmap, limits) in enumerate(panels):
            kwargs = {} if limits is None else {"vmin": limits[0], "vmax": limits[1]}
            axes[timestamp, column].imshow(panel, cmap=cmap, **kwargs)
            axes[timestamp, column].axis("off")
            if column == 0:
                axes[timestamp, column].set_ylabel(f"frame {frame}")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _resize_for_rerun(array: torch.Tensor, resolution: int, mode: str) -> torch.Tensor:
    views, frames, channels, height, width = array.shape
    resized = F.interpolate(
        array.reshape(views * frames, channels, height, width).float(),
        size=(resolution, resolution),
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    return resized.reshape(views, frames, channels, resolution, resolution)


def _save_rerun(
    path: Path,
    scene_name: str,
    rgbs: np.ndarray,
    ground_truth: np.ndarray,
    estimated: np.ndarray,
    cleaned: np.ndarray,
    gt_intrinsics: np.ndarray,
    gt_w2c: np.ndarray,
    predicted_intrinsics: np.ndarray,
    predicted_w2c: np.ndarray,
    resolution: int,
) -> None:
    import rerun as rr

    from mvtracker.utils.visualizer_rerun import log_pointclouds_to_rerun

    rgb = torch.from_numpy(rgbs).permute(0, 1, 4, 2, 3)
    gt = torch.from_numpy(ground_truth[:, :, None])
    estimate = torch.from_numpy(np.where(cleaned, estimated, 0)[:, :, None])
    source_height, source_width = ground_truth.shape[-2:]
    rgb = _resize_for_rerun(rgb, resolution, "bilinear").clamp(0, 255).byte()
    gt = _resize_for_rerun(gt, resolution, "nearest")
    estimate = _resize_for_rerun(estimate, resolution, "nearest")
    scale_x = (resolution - 1) / max(source_width - 1, 1)
    scale_y = (resolution - 1) / max(source_height - 1, 1)

    def scaled_intrinsics(values: np.ndarray) -> torch.Tensor:
        values = values.copy()
        values[..., 0, 0] *= scale_x
        values[..., 0, 2] *= scale_x
        values[..., 1, 1] *= scale_y
        values[..., 1, 2] *= scale_y
        return torch.from_numpy(values)

    rr.init("vggt-omega-sidecar-quality", recording_id=scene_name)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    common = dict(
        datapoint_idx=scene_name,
        rgbs=rgb[None],
        log_rgb_image=True,
        log_depthmap_as_image_v2=True,
        log_camera_frustrum=True,
        log_rgb_pointcloud=True,
    )
    log_pointclouds_to_rerun(
        dataset_name="ground-truth",
        depths=gt[None],
        intrs=scaled_intrinsics(gt_intrinsics)[None],
        extrs=torch.from_numpy(gt_w2c[..., :3, :])[None],
        **common,
    )
    log_pointclouds_to_rerun(
        dataset_name="vggt-omega-cleaned",
        depths=estimate[None],
        intrs=scaled_intrinsics(predicted_intrinsics)[None],
        extrs=torch.from_numpy(predicted_w2c[..., :3, :])[None],
        **common,
    )
    rr.save(path)


def main() -> None:
    args = parse_args()
    sidecar = args.sidecar_root / args.scene_root.name
    manifest = json.loads((sidecar / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("complete") is not True
        or manifest.get("provider") != "vggt_omega"
        or manifest.get("schema_version") != SCHEMA_VERSION
    ):
        raise ValueError(f"{sidecar}: not a complete VGGT-Omega sidecar")
    available_views = tuple(int(view) for view in manifest["view_ids"])
    view_ids = available_views if args.views is None else tuple(args.views)
    view_positions = [available_views.index(view) for view in view_ids]
    frames = representative_frame_indices(int(manifest["frame_count"]))
    rgbs, ground_truth, gt_intrinsics, gt_w2c = _load_source(
        args.dataset, args.scene_root, view_ids, frames
    )
    frame_positions = list(frames)
    estimated_all = np.load(sidecar / "depth.npy", mmap_mode="r")
    cleaned_all = np.load(sidecar / "cleaned_mask.npy", mmap_mode="r")
    predicted_intrinsics_all = np.load(sidecar / "predicted_intrinsics.npy", mmap_mode="r")
    predicted_w2c_all = np.load(sidecar / "predicted_extrinsics_w2c.npy", mmap_mode="r")
    estimated = np.asarray(estimated_all[np.ix_(view_positions, frame_positions)], dtype=np.float32)
    cleaned = np.asarray(cleaned_all[np.ix_(view_positions, frame_positions)], dtype=bool)
    predicted_intrinsics = np.asarray(
        predicted_intrinsics_all[np.ix_(view_positions, frame_positions)], dtype=np.float32
    )
    predicted_w2c = np.asarray(
        predicted_w2c_all[np.ix_(view_positions, frame_positions)], dtype=np.float32
    )
    metrics = depth_quality_metrics(
        estimated,
        cleaned,
        ground_truth,
        np.load(sidecar / "scales.npy", mmap_mode="r"),
        predicted_w2c,
        gt_w2c,
        frames,
    )
    report = {"scene": args.scene_root.name, "dataset": args.dataset, "view_ids": list(view_ids), **metrics}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / f"{args.scene_root.name}__vggt_omega_quality"
    (prefix.with_suffix(".json")).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _save_contact_sheet(
        prefix.with_suffix(".png"), frames, rgbs, ground_truth, estimated, cleaned
    )
    if not args.no_rerun:
        _save_rerun(
            prefix.with_suffix(".rrd"),
            args.scene_root.name,
            rgbs,
            ground_truth,
            estimated,
            cleaned,
            gt_intrinsics,
            gt_w2c,
            predicted_intrinsics,
            predicted_w2c,
            args.rerun_resolution,
        )
    print(json.dumps({"event": "quality_artifacts_complete", "prefix": str(prefix), **report}))


if __name__ == "__main__":
    main()
