"""Build an interactive Rerun recording from Waymo LiDAR and TAPVid-3D tracks."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image


CAMERA_NAMES = {1: "front", 2: "front_left", 3: "front_right"}


@dataclass(frozen=True)
class TapVid3DAnnotation:
    images_jpeg: tuple[bytes, ...]
    tracks_xyz: np.ndarray
    visibility: np.ndarray
    queries_xyt: np.ndarray
    intrinsics: np.ndarray
    extrinsics_w2c: np.ndarray


def load_tapvid3d_annotation(path: Path) -> TapVid3DAnnotation:
    with np.load(path, allow_pickle=False) as data:
        images = tuple(value.tobytes() for value in data["images_jpeg_bytes"])
        tracks = np.asarray(data["tracks_XYZ"], dtype=np.float32)
        visibility = np.asarray(data["visibility"], dtype=bool)
        queries = np.asarray(data["queries_xyt"], dtype=np.float32)
        intrinsics = np.asarray(data["fx_fy_cx_cy"], dtype=np.float32)
        extrinsics = np.asarray(data["extrinsics_w2c"], dtype=np.float32)
    frame_count, track_count, xyz = tracks.shape
    if xyz != 3 or visibility.shape != (frame_count, track_count):
        raise ValueError("invalid TAPVid-3D trajectory arrays")
    if len(images) != frame_count or extrinsics.shape != (frame_count, 4, 4):
        raise ValueError("TAPVid-3D frame arrays disagree")
    return TapVid3DAnnotation(
        images_jpeg=images,
        tracks_xyz=tracks,
        visibility=visibility,
        queries_xyt=queries,
        intrinsics=intrinsics,
        extrinsics_w2c=extrinsics,
    )


def _jpeg_fingerprint(contents: bytes, size: tuple[int, int] = (48, 32)) -> np.ndarray:
    image = Image.open(BytesIO(contents)).convert("RGB").resize(size, Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)


def match_annotation_frames(
    annotation_jpegs: Sequence[bytes],
    raw_jpegs_by_camera: Mapping[int, Sequence[bytes]],
    *,
    maximum_mse: float = 64.0,
) -> tuple[int, np.ndarray, float]:
    """Find the source camera and monotonically matching raw frame indices."""
    annotation_fingerprints = np.stack([_jpeg_fingerprint(value) for value in annotation_jpegs])
    candidates: list[tuple[float, int, np.ndarray]] = []
    for camera, raw_jpegs in raw_jpegs_by_camera.items():
        exact = {value: index for index, value in enumerate(raw_jpegs)}
        if all(value in exact for value in annotation_jpegs):
            indices = np.asarray([exact[value] for value in annotation_jpegs], dtype=np.int32)
            score = 0.0
        else:
            raw_fingerprints = np.stack([_jpeg_fingerprint(value) for value in raw_jpegs])
            difference = annotation_fingerprints[:, None] - raw_fingerprints[None]
            mse = np.mean(difference * difference, axis=(2, 3, 4))
            indices = np.argmin(mse, axis=1).astype(np.int32)
            score = float(np.mean(mse[np.arange(len(indices)), indices]))
        if np.all(np.diff(indices) > 0):
            candidates.append((score, camera, indices))
    if not candidates:
        raise ValueError("no camera produced strictly increasing TAPVid-3D frame matches")
    score, camera, indices = min(candidates, key=lambda item: item[0])
    if score > maximum_mse:
        raise ValueError(f"best camera-frame match has excessive MSE: {score:.3f}")
    return camera, indices, score


def fit_rigid_transform(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the rigid target_from_source transform and alignment RMSE."""
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("rigid alignment expects matching [N,3] arrays")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_mean - rotation @ source_mean
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    aligned = source @ rotation.T + translation
    rmse = float(np.sqrt(np.mean(np.sum((aligned - target) ** 2, axis=1))))
    return transform, rmse


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


def points_inside_boxes(points: np.ndarray, boxes: Iterable[Sequence[float]]) -> np.ndarray:
    """Mask points inside Waymo boxes `(cx,cy,cz,l,w,h,heading)`."""
    mask = np.zeros(len(points), dtype=bool)
    for cx, cy, cz, length, width, height, heading in boxes:
        offset = points - np.asarray([cx, cy, cz])
        cosine, sine = np.cos(heading), np.sin(heading)
        local_x = cosine * offset[:, 0] + sine * offset[:, 1]
        local_y = -sine * offset[:, 0] + cosine * offset[:, 1]
        mask |= (
            (np.abs(local_x) <= length / 2)
            & (np.abs(local_y) <= width / 2)
            & (np.abs(offset[:, 2]) <= height / 2)
        )
    return mask


def voxel_downsample(points: np.ndarray, colors: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    if len(points) != len(colors):
        raise ValueError("point and color counts disagree")
    cells = np.floor(points / voxel_size).astype(np.int64)
    _, indices = np.unique(cells, axis=0, return_index=True)
    indices.sort()
    return points[indices], colors[indices]


def select_tracks(tracks: np.ndarray, visibility: np.ndarray, count: int = 6) -> np.ndarray:
    finite = np.isfinite(tracks).all(axis=2)
    usable = finite.sum(axis=0) >= max(2, tracks.shape[0] // 2)
    usable &= visibility.sum(axis=0) >= 2
    steps = np.linalg.norm(np.diff(tracks, axis=0), axis=2)
    steps = np.where(np.isfinite(steps), steps, np.nan)
    path_length = np.where(usable, np.nansum(steps, axis=0), -np.inf)
    order = np.argsort(path_length)[::-1]
    selected: list[int] = []
    for index in order:
        if not np.isfinite(path_length[index]):
            break
        center = np.nanmean(tracks[:, index], axis=0)
        if all(np.linalg.norm(center - np.nanmean(tracks[:, other], axis=0)) >= 0.20 for other in selected):
            selected.append(int(index))
        if len(selected) == count:
            break
    if not selected:
        raise ValueError("no usable tracks")
    return np.asarray(selected, dtype=np.int32)


def time_colors(frame_count: int) -> np.ndarray:
    anchors = np.asarray([[0, 220, 255], [110, 255, 30], [255, 40, 20]], dtype=np.float32)
    position = np.linspace(0, 2, frame_count)
    low = np.minimum(position.astype(np.int32), 1)
    fraction = position - low
    return np.rint(anchors[low] * (1 - fraction[:, None]) + anchors[low + 1] * fraction[:, None]).astype(np.uint8)


def trajectory_segments(tracks: np.ndarray, selected: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    colors = time_colors(tracks.shape[0] - 1)
    segments, segment_colors = [], []
    for index in selected:
        for frame in range(tracks.shape[0] - 1):
            segment = tracks[frame : frame + 2, index]
            if np.isfinite(segment).all():
                segments.append(segment)
                segment_colors.append(colors[frame])
    return np.asarray(segments, dtype=np.float32), np.asarray(segment_colors, dtype=np.uint8)


def _camera_images(frame: object) -> dict[int, bytes]:
    return {int(image.name): bytes(image.image) for image in frame.images}


def _camera_poses(frame: object) -> dict[int, np.ndarray]:
    world_from_vehicle = np.asarray(frame.pose.transform, dtype=np.float64).reshape(4, 4)
    result = {}
    for calibration in frame.context.camera_calibrations:
        vehicle_from_camera = np.asarray(calibration.extrinsic.transform, dtype=np.float64).reshape(4, 4)
        result[int(calibration.name)] = world_from_vehicle @ vehicle_from_camera
    return result


def _colorize_points(points: np.ndarray, projections: np.ndarray, images: Mapping[int, bytes]) -> np.ndarray:
    decoded = {camera: np.asarray(Image.open(BytesIO(value)).convert("RGB")) for camera, value in images.items()}
    colors = np.full((len(points), 3), 150, dtype=np.uint8)
    colored = np.zeros(len(points), dtype=bool)
    for slot in (0, 3):
        camera_ids = projections[:, slot]
        for camera in np.unique(camera_ids):
            camera = int(camera)
            if camera == 0 or camera not in decoded:
                continue
            mask = (camera_ids == camera) & ~colored
            x = projections[mask, slot + 1].astype(np.int64)
            y = projections[mask, slot + 2].astype(np.int64)
            image = decoded[camera]
            valid = (x >= 0) & (x < image.shape[1]) & (y >= 0) & (y < image.shape[0])
            indices = np.flatnonzero(mask)[valid]
            colors[indices] = image[y[valid], x[valid]]
            colored[indices] = True
    return colors


def build_waymo_rerun(
    tfrecord_path: Path,
    annotation_path: Path,
    output_path: Path,
    *,
    track_count: int = 6,
    scene_frame_count: int = 13,
    points_per_frame: int = 120_000,
    voxel_size: float = 0.08,
) -> dict[str, object]:
    """Parse one Waymo segment and save a synchronized `.rrd` recording."""
    import tensorflow as tf
    import rerun as rr
    import rerun.blueprint as rrb
    from waymo_open_dataset import dataset_pb2
    from waymo_open_dataset.utils import frame_utils

    annotation = load_tapvid3d_annotation(annotation_path)
    raw_images = {camera: [] for camera in CAMERA_NAMES}
    raw_camera_poses = {camera: [] for camera in CAMERA_NAMES}
    raw_timestamps = []
    for record in tf.data.TFRecordDataset(str(tfrecord_path)):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(record.numpy())
        images = _camera_images(frame)
        poses = _camera_poses(frame)
        for camera in CAMERA_NAMES:
            raw_images[camera].append(images[camera])
            raw_camera_poses[camera].append(poses[camera])
        raw_timestamps.append(int(frame.timestamp_micros))

    camera, raw_indices, image_match_mse = match_annotation_frames(annotation.images_jpeg, raw_images)
    annotation_c2w = np.linalg.inv(annotation.extrinsics_w2c)
    annotation_centers = annotation_c2w[:, :3, 3]
    annotation_world_tracks = np.stack([
        transform_points(annotation_c2w[frame], annotation.tracks_xyz[frame])
        for frame in range(len(annotation.tracks_xyz))
    ]).astype(np.float32)
    source_camera_poses = np.asarray(raw_camera_poses[camera])[raw_indices]
    annotation_from_raw, alignment_rmse = fit_rigid_transform(
        source_camera_poses[:, :3, 3], annotation_centers
    )
    if alignment_rmse > 0.10:
        raise ValueError(f"raw/annotation camera alignment RMSE is {alignment_rmse:.3f}m")

    scene_positions = np.linspace(0, len(raw_indices) - 1, scene_frame_count).round().astype(np.int32)
    scene_raw_indices = set(raw_indices[scene_positions].tolist())
    point_batches, color_batches = [], []
    for raw_index, record in enumerate(tf.data.TFRecordDataset(str(tfrecord_path))):
        if raw_index not in scene_raw_indices:
            continue
        frame = dataset_pb2.Frame()
        frame.ParseFromString(record.numpy())
        range_images, projections, _, top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
        points_list, projection_list = frame_utils.convert_range_image_to_point_cloud(
            frame, range_images, projections, top_pose, ri_index=0
        )
        points_vehicle = np.concatenate(points_list, axis=0)
        camera_projections = np.concatenate(projection_list, axis=0)
        boxes = [
            (label.box.center_x, label.box.center_y, label.box.center_z,
             label.box.length, label.box.width, label.box.height, label.box.heading)
            for label in frame.laser_labels
        ]
        keep = ~points_inside_boxes(points_vehicle, boxes)
        points_vehicle = points_vehicle[keep]
        camera_projections = camera_projections[keep]
        if len(points_vehicle) > points_per_frame:
            sample = np.linspace(0, len(points_vehicle) - 1, points_per_frame).astype(np.int64)
            points_vehicle = points_vehicle[sample]
            camera_projections = camera_projections[sample]
        world_from_vehicle = np.asarray(frame.pose.transform, dtype=np.float64).reshape(4, 4)
        points_raw_world = transform_points(world_from_vehicle, points_vehicle)
        point_batches.append(transform_points(annotation_from_raw, points_raw_world))
        color_batches.append(_colorize_points(points_vehicle, camera_projections, _camera_images(frame)))

    points = np.concatenate(point_batches).astype(np.float32)
    colors = np.concatenate(color_batches)
    points, colors = voxel_downsample(points, colors, voxel_size)
    selected = select_tracks(annotation_world_tracks, annotation.visibility, track_count)
    anchor = annotation_centers[0].astype(np.float32)
    points -= anchor
    tracks = annotation_world_tracks - anchor
    segments, segment_colors = trajectory_segments(tracks, selected)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rr.init("waymo_long_horizon_tracks", spawn=False)
    rr.save(str(output_path))
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Vertical(
                rrb.Spatial3DView(origin="/world", name="Waymo world"),
                rrb.Horizontal(
                    *(rrb.Spatial2DView(origin=f"/cameras/{name}", name=name.replace("_", " ").title())
                      for name in CAMERA_NAMES.values()),
                    column_shares=[1, 1, 1],
                ),
                row_shares=[3, 1],
            ),
            collapse_panels=True,
        )
    )
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/scene", rr.Points3D(points, colors=colors, radii=0.025), static=True)
    rr.log(
        "world/trajectories",
        rr.LineStrips3D(strips=segments, colors=segment_colors, radii=0.045),
        static=True,
    )
    current_colors = time_colors(len(raw_indices))
    for frame_index, raw_index in enumerate(raw_indices):
        rr.set_time_sequence("frame", frame_index)
        for camera_id, name in CAMERA_NAMES.items():
            rr.log(
                f"cameras/{name}",
                rr.EncodedImage(contents=raw_images[camera_id][raw_index], media_type="image/jpeg"),
            )
        visible = annotation.visibility[frame_index, selected]
        rr.log(
            "world/current",
            rr.Points3D(
                tracks[frame_index, selected][visible],
                colors=np.repeat(current_colors[frame_index][None], int(visible.sum()), axis=0),
                radii=0.09,
            ),
        )
    rr.flush(timeout_sec=120.0)
    rr.disconnect()
    return {
        "output_path": str(output_path),
        "frames": len(raw_indices),
        "raw_frame_indices": raw_indices.tolist(),
        "annotation_camera": CAMERA_NAMES[camera],
        "image_match_mse": image_match_mse,
        "alignment_rmse_m": alignment_rmse,
        "scene_points": len(points),
        "track_indices": selected.tolist(),
        "rrd_bytes": output_path.stat().st_size,
        "first_timestamp_micros": raw_timestamps[int(raw_indices[0])],
        "last_timestamp_micros": raw_timestamps[int(raw_indices[-1])],
    }
