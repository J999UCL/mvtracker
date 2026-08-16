"""Quality metrics for metric VGGT-Omega depth sidecars."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from mvtracker.preprocessing.vggt_omega import camera_centres


def representative_frame_indices(frame_count: int) -> tuple[int, ...]:
    """Return unique start, middle, and end frame indices."""
    if frame_count < 1:
        raise ValueError("frame_count must be positive")
    return tuple(dict.fromkeys((0, frame_count // 2, frame_count - 1)))


def depth_quality_metrics(
    estimated_depth: np.ndarray,
    cleaned_mask: np.ndarray,
    ground_truth_depth: np.ndarray,
    scales: np.ndarray,
    predicted_extrinsics_w2c: np.ndarray,
    known_extrinsics_w2c: np.ndarray,
    frame_indices: Sequence[int],
) -> dict:
    """Compare sampled sidecar depths with metric ground truth.

    Depth tensors are shaped ``[V, T, H, W]`` and camera tensors are shaped
    ``[V, T, 4, 4]``. ``T`` is the number of sampled frames, while
    ``frame_indices`` records their original sequence indices.
    """
    estimated = np.asarray(estimated_depth, dtype=np.float64)
    cleaned = np.asarray(cleaned_mask, dtype=bool)
    ground_truth = np.asarray(ground_truth_depth, dtype=np.float64)
    predicted_w2c = np.asarray(predicted_extrinsics_w2c, dtype=np.float64)
    known_w2c = np.asarray(known_extrinsics_w2c, dtype=np.float64)
    frame_indices = tuple(int(index) for index in frame_indices)
    if estimated.shape != ground_truth.shape or cleaned.shape != estimated.shape:
        raise ValueError("estimated, cleaned, and ground-truth depth shapes must match")
    if estimated.ndim != 4:
        raise ValueError("depth tensors must be shaped [V, T, H, W]")
    views, timestamps = estimated.shape[:2]
    if len(frame_indices) != timestamps:
        raise ValueError("frame_indices must match the sampled timestamp count")
    if predicted_w2c.shape != (views, timestamps, 4, 4):
        raise ValueError("predicted_extrinsics_w2c must be shaped [V, T, 4, 4]")
    if known_w2c.shape != predicted_w2c.shape:
        raise ValueError("known and predicted extrinsics shapes must match")

    gt_valid = np.isfinite(ground_truth) & (ground_truth > 0)
    estimate_valid = np.isfinite(estimated) & (estimated > 0)
    overlap = gt_valid & estimate_valid
    cleaned_overlap = overlap & cleaned
    if not np.any(overlap):
        raise ValueError("sampled frames contain no overlapping valid depth")

    error = estimated[overlap] - ground_truth[overlap]
    absolute_error = np.abs(error)
    relative_error = absolute_error / ground_truth[overlap]
    depth_ratio = estimated[overlap] / ground_truth[overlap]
    cleaned_relative_error = (
        np.abs(estimated[cleaned_overlap] - ground_truth[cleaned_overlap])
        / ground_truth[cleaned_overlap]
    )
    residuals = []
    for timestamp in range(timestamps):
        predicted_centres = camera_centres(predicted_w2c[:, timestamp])
        known_centres = camera_centres(known_w2c[:, timestamp])
        squared_error = np.sum((predicted_centres - known_centres) ** 2, axis=-1)
        residuals.append(float(np.sqrt(np.mean(squared_error))))

    scales = np.asarray(scales, dtype=np.float64)
    if len(scales) == timestamps:
        sampled_scales = scales
    elif frame_indices and min(frame_indices) >= 0 and max(frame_indices) < len(scales):
        sampled_scales = scales[np.asarray(frame_indices)]
    else:
        raise ValueError("scales must be aligned with frame_indices or sampled timestamps")
    return {
        "frame_indices": list(frame_indices),
        "view_count": views,
        "ground_truth_valid_fraction": float(gt_valid.mean()),
        "estimated_valid_fraction": float(estimate_valid.mean()),
        "cleaned_mask_fraction": float(cleaned.mean()),
        "overlap_fraction": float(overlap.mean()),
        "cleaned_overlap_fraction": float(cleaned_overlap.mean()),
        "cleaned_recall_of_gt_valid": float(cleaned_overlap.sum() / gt_valid.sum()),
        "mean_absolute_error_m": float(absolute_error.mean()),
        "median_absolute_error_m": float(np.median(absolute_error)),
        "rmse_m": float(np.sqrt(np.mean(np.square(error)))),
        "mean_absolute_relative_error": float(relative_error.mean()),
        "cleaned_mean_absolute_relative_error": (
            float(cleaned_relative_error.mean()) if cleaned_relative_error.size else None
        ),
        "median_estimated_to_gt_depth_ratio": float(np.median(depth_ratio)),
        "sampled_metric_scales": sampled_scales.tolist(),
        "sampled_camera_center_rmse_m": residuals,
        "mean_camera_center_rmse_m": float(np.mean(residuals)),
        "max_camera_center_rmse_m": float(np.max(residuals)),
    }
