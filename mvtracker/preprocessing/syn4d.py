"""Core helpers for converting Syn4D into a TAPVid-MV cache.

The module deliberately contains no download, Modal, video, or OpenEXR
orchestration.  It operates on decoded arrays and official Syn4D metadata so
the expensive conversion stages can be timed and run independently.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

import numpy as np


CACHE_FORMAT = "syn4d-tapvid-mv"
TEMPLE_GROUP_SCENE = "temple_group"
TEMPLE_GROUP_SEQUENCE_BASES = tuple(f"seq_{index:06d}" for index in range(20))
VIEW_COUNT = 8
TRACK_COUNT = 65_536
CACHE_HEIGHT = 384
CACHE_WIDTH = 683


@dataclass(frozen=True)
class SequenceDependencies:
    sequence_base: str
    body_motion: str
    clothing_member: str
    objects: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class TempleGroupDependencies:
    sequences: tuple[SequenceDependencies, ...]
    body_motions: tuple[str, ...]
    clothing_members: tuple[str, ...]
    objects: tuple[tuple[str, str], ...]


def _clothing_member(body_motion: str) -> str:
    subject, animation = body_motion.rsplit("_", 1)
    return f"{subject}/{animation}/{animation}.npz"


def _object_asset(asset: str) -> tuple[str, str]:
    group, object_id = asset.rsplit("_", 1)
    return group, object_id


def sequence_dependencies(
    mapping_csv: Path, *, scene: str, sequence_base: str
) -> SequenceDependencies:
    """Resolve the body, clothing, and objects for one Syn4D sequence."""

    views: set[int] = set()
    bodies: set[str] = set()
    objects: set[tuple[str, str]] = set()
    with Path(mapping_csv).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("scene") != scene:
                continue
            match = re.fullmatch(rf"{re.escape(sequence_base)}_([0-7])", str(row.get("sequence_name", "")))
            if match is None:
                continue
            views.add(int(match.group(1)))
            asset_type = row.get("asset_type")
            asset = str(row.get("asset", ""))
            if asset_type == "bedlam2_body":
                bodies.add(asset)
            elif asset_type == "objaverse_object":
                objects.add(_object_asset(asset))
    if views != set(range(VIEW_COUNT)):
        raise ValueError(f"{scene}/{sequence_base} does not cover all eight cameras")
    if len(bodies) != 1 or len(objects) != 3:
        raise ValueError(
            f"{scene}/{sequence_base} must resolve to one body and three objects"
        )
    body_motion = next(iter(bodies))
    return SequenceDependencies(
        sequence_base=sequence_base,
        body_motion=body_motion,
        clothing_member=_clothing_member(body_motion),
        objects=tuple(sorted(objects)),
    )


def temple_group_dependencies(mapping_csv: Path) -> TempleGroupDependencies:
    """Read the pinned Syn4D mapping and select exactly ``temple_group``.

    Syn4D repeats asset rows for camera-specific sequence names.  Dependencies
    are collapsed to their ``seq_XXXXXX`` base and must agree across all eight
    cameras.
    """

    by_sequence: dict[str, dict[str, set]] = {}
    with Path(mapping_csv).open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("scene") != TEMPLE_GROUP_SCENE:
                continue
            camera_sequence = str(row.get("sequence_name", ""))
            match = re.fullmatch(r"(seq_\d{6})_([0-7])", camera_sequence)
            if match is None:
                raise ValueError(
                    f"unparseable temple_group camera sequence {camera_sequence!r}"
                )
            sequence_base, view_text = match.groups()
            entry = by_sequence.setdefault(
                sequence_base, {"views": set(), "bodies": set(), "objects": set()}
            )
            entry["views"].add(int(view_text))
            asset_type = row.get("asset_type")
            asset = str(row.get("asset", ""))
            if asset_type == "bedlam2_body":
                entry["bodies"].add(asset)
            elif asset_type == "objaverse_object":
                entry["objects"].add(_object_asset(asset))

    if tuple(sorted(by_sequence)) != TEMPLE_GROUP_SEQUENCE_BASES:
        raise ValueError(
            "temple_group mapping must contain seq_000000 through seq_000019"
        )

    sequences: list[SequenceDependencies] = []
    for sequence_base in TEMPLE_GROUP_SEQUENCE_BASES:
        entry = by_sequence[sequence_base]
        if entry["views"] != set(range(VIEW_COUNT)):
            raise ValueError(f"{sequence_base} does not cover all eight cameras")
        bodies = sorted(entry["bodies"])
        objects = tuple(sorted(entry["objects"]))
        if len(bodies) != 1:
            raise ValueError(f"{sequence_base} must resolve to one BEDLAM2 body")
        if len(objects) != 3:
            raise ValueError(f"{sequence_base} must resolve to three animated objects")
        body_motion = bodies[0]
        sequences.append(
            SequenceDependencies(
                sequence_base=sequence_base,
                body_motion=body_motion,
                clothing_member=_clothing_member(body_motion),
                objects=objects,
            )
        )

    body_motions = tuple(sorted({item.body_motion for item in sequences}))
    clothing_members = tuple(sorted({item.clothing_member for item in sequences}))
    objects = tuple(sorted({asset for item in sequences for asset in item.objects}))
    if (len(body_motions), len(clothing_members), len(objects)) != (20, 20, 60):
        raise ValueError(
            "unexpected temple_group dependency counts: "
            f"bodies={len(body_motions)} clothing={len(clothing_members)} "
            f"objects={len(objects)}"
        )
    return TempleGroupDependencies(
        sequences=tuple(sequences),
        body_motions=body_motions,
        clothing_members=clothing_members,
        objects=objects,
    )


def discover_temple_group_sequences(scene_root: Path) -> tuple[str, ...]:
    """Require the complete 20-sequence, eight-camera MP4 inventory."""

    views: dict[str, set[int]] = {}
    for path in sorted((Path(scene_root) / "mp4").glob("*.mp4")):
        match = re.fullmatch(r"(seq_\d{6})_([0-7])\.mp4", path.name)
        if match is not None:
            views.setdefault(match.group(1), set()).add(int(match.group(2)))
    if tuple(sorted(views)) != TEMPLE_GROUP_SEQUENCE_BASES:
        raise ValueError("temple_group MP4 inventory does not contain exactly 20 sequences")
    incomplete = {
        sequence: sorted(set(range(VIEW_COUNT)) - sequence_views)
        for sequence, sequence_views in views.items()
        if sequence_views != set(range(VIEW_COUNT))
    }
    if incomplete:
        raise ValueError(f"temple_group sequences have missing views: {incomplete}")
    return TEMPLE_GROUP_SEQUENCE_BASES


@dataclass(frozen=True)
class SurfaceCandidate:
    """One reusable Syn4D surface point and its reference observation."""

    entity_name: str | None
    face_id: int
    barycentric: tuple[float, float, float]
    normal_offset_m: float
    env_world_xyz: tuple[float, float, float]
    query_xytv: tuple[float, float, float, float]

    @property
    def identity(self) -> tuple:
        # Preserve float32 surface values exactly.  No cache field or identity
        # coordinate is quantized.
        if self.entity_name is None:
            return ("environment", *self.env_world_xyz)
        return (
            self.entity_name,
            self.face_id,
            *self.barycentric,
            self.normal_offset_m,
        )


@dataclass(frozen=True)
class SurfaceBank:
    entity_names: tuple[str, ...]
    entity_id: np.ndarray
    face_id: np.ndarray
    barycentric: np.ndarray
    normal_offset_m: np.ndarray
    env_world_xyz: np.ndarray
    queries_xytv: np.ndarray

    @property
    def count(self) -> int:
        return int(self.entity_id.size)


@dataclass(frozen=True)
class EntityMesh:
    """World-frame animated mesh matching official Syn4D face topology."""

    vertices: np.ndarray
    faces: np.ndarray
    frame_valid: np.ndarray | None = None


def dynamic_surface_coordinates(
    points_xyz: np.ndarray,
    face_ids: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    max_normal_offset_m: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover barycentric coordinates and signed normal offsets.

    This mirrors the official compact face-ID reader: the source point is the
    barycentric triangle point plus ``signed_distance * oriented_normal``.
    """

    points = np.asarray(points_xyz, dtype=np.float32)
    selected_faces = np.asarray(face_ids, dtype=np.int64)
    verts = np.asarray(vertices, dtype=np.float32)
    triangles_index = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points_xyz must be [N,3], got {points.shape}")
    if selected_faces.shape != (points.shape[0],):
        raise ValueError("face_ids must have one entry per point")
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices must be [M,3], got {verts.shape}")
    if triangles_index.ndim != 2 or triangles_index.shape[1] != 3:
        raise ValueError(f"faces must be [K,3], got {triangles_index.shape}")
    if np.any(selected_faces < 0) or np.any(selected_faces >= len(triangles_index)):
        raise ValueError("face_ids index outside faces")
    if np.any(triangles_index < 0) or (
        triangles_index.size and int(triangles_index.max()) >= len(verts)
    ):
        raise ValueError("faces index outside vertices")

    triangles = verts[triangles_index[selected_faces]]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    edge_0, edge_1, point_delta = b - a, c - a, points - a
    d00 = np.sum(edge_0 * edge_0, axis=-1)
    d01 = np.sum(edge_0 * edge_1, axis=-1)
    d11 = np.sum(edge_1 * edge_1, axis=-1)
    d20 = np.sum(point_delta * edge_0, axis=-1)
    d21 = np.sum(point_delta * edge_1, axis=-1)
    denominator = d00 * d11 - d01 * d01
    usable = np.isfinite(denominator) & (np.abs(denominator) > 1e-12)
    safe_denominator = np.where(usable, denominator, 1.0)
    bary_1 = (d11 * d20 - d01 * d21) / safe_denominator
    bary_2 = (d00 * d21 - d01 * d20) / safe_denominator
    barycentric = np.stack((1.0 - bary_1 - bary_2, bary_1, bary_2), axis=-1)

    normal = np.cross(edge_0, edge_1)
    normal_length = np.linalg.norm(normal, axis=-1)
    usable &= np.isfinite(normal_length) & (normal_length > 1e-8)
    oriented_normal = np.zeros_like(normal)
    np.divide(normal, normal_length[:, None], out=oriented_normal, where=usable[:, None])
    projected = np.sum(triangles * barycentric[:, :, None], axis=1)
    signed_distance = np.sum((points - projected) * oriented_normal, axis=-1)
    usable &= (
        np.isfinite(points).all(axis=-1)
        & np.isfinite(barycentric).all(axis=-1)
        & np.isfinite(signed_distance)
        & (np.abs(signed_distance) <= float(max_normal_offset_m))
    )
    return (
        np.ascontiguousarray(barycentric, dtype=np.float32),
        np.ascontiguousarray(signed_distance, dtype=np.float32),
        usable,
    )


def compact_surface_candidates(
    *,
    world_points: np.ndarray,
    valid_flat_idx: np.ndarray,
    face_ids: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    entity_name: str,
    query_time: int,
    query_view: int,
) -> list[SurfaceCandidate]:
    """Turn one official compact face-ID payload into reusable candidates."""

    points_map = np.asarray(world_points, dtype=np.float32)
    if points_map.ndim != 3 or points_map.shape[2] != 3:
        raise ValueError(f"world_points must be [H,W,3], got {points_map.shape}")
    height, width = points_map.shape[:2]
    flat_indices = np.asarray(valid_flat_idx, dtype=np.int64).reshape(-1)
    selected_faces = np.asarray(face_ids, dtype=np.int64).reshape(-1)
    if flat_indices.shape != selected_faces.shape:
        raise ValueError("valid_flat_idx and face_ids must have identical shape")
    in_bounds = (
        (flat_indices >= 0)
        & (flat_indices < height * width)
        & (selected_faces >= 0)
        & (selected_faces < len(faces))
    )
    flat_indices = flat_indices[in_bounds]
    selected_faces = selected_faces[in_bounds]
    selected_points = points_map.reshape(-1, 3)[flat_indices]
    barycentric, signed_distance, usable = dynamic_surface_coordinates(
        selected_points, selected_faces, vertices, faces
    )
    flat_indices = flat_indices[usable]
    selected_faces = selected_faces[usable]
    barycentric = barycentric[usable]
    signed_distance = signed_distance[usable]
    candidates: list[SurfaceCandidate] = []
    for flat_index, face_id, bary, offset in zip(
        flat_indices, selected_faces, barycentric, signed_distance, strict=True
    ):
        y, x = divmod(int(flat_index), width)
        candidates.append(
            SurfaceCandidate(
                entity_name=entity_name,
                face_id=int(face_id),
                barycentric=tuple(float(value) for value in bary),
                normal_offset_m=float(offset),
                env_world_xyz=(0.0, 0.0, 0.0),
                query_xytv=(float(x), float(y), float(query_time), float(query_view)),
            )
        )
    return candidates


def environment_surface_candidates(
    *,
    world_points: np.ndarray,
    valid: np.ndarray,
    query_time: int,
    query_view: int,
) -> list[SurfaceCandidate]:
    points = np.asarray(world_points, dtype=np.float32)
    selected = np.asarray(valid, dtype=bool)
    if points.ndim != 3 or points.shape[2] != 3 or selected.shape != points.shape[:2]:
        raise ValueError("environment points/valid must be [H,W,3] and [H,W]")
    selected &= np.isfinite(points).all(axis=-1)
    candidates: list[SurfaceCandidate] = []
    for y, x in np.argwhere(selected):
        xyz = points[y, x]
        candidates.append(
            SurfaceCandidate(
                entity_name=None,
                face_id=-1,
                barycentric=(0.0, 0.0, 0.0),
                normal_offset_m=0.0,
                env_world_xyz=tuple(float(value) for value in xyz),
                query_xytv=(float(x), float(y), float(query_time), float(query_view)),
            )
        )
    return candidates


def sample_candidate_quarter(
    candidates: Sequence[SurfaceCandidate], rng: np.random.Generator
) -> list[SurfaceCandidate]:
    """Retain a random 25 percent of one anchor's valid candidates."""

    count = len(candidates) // 4
    if count == 0:
        return []
    indices = rng.choice(len(candidates), size=count, replace=False)
    return [candidates[int(index)] for index in indices]


def build_surface_bank(
    candidates: Iterable[SurfaceCandidate], *, count: int = TRACK_COUNT
) -> SurfaceBank:
    """Deduplicate exact surface identities and retain the requested bank size."""

    unique: dict[tuple, SurfaceCandidate] = {}
    for candidate in candidates:
        unique.setdefault(candidate.identity, candidate)
        if len(unique) == count:
            break
    if len(unique) != count:
        raise ValueError(f"found {len(unique)} unique surface points; need {count}")
    selected = tuple(unique.values())
    entity_names = tuple(
        sorted({candidate.entity_name for candidate in selected if candidate.entity_name})
    )
    entity_to_id = {name: index for index, name in enumerate(entity_names)}
    return SurfaceBank(
        entity_names=entity_names,
        entity_id=np.asarray(
            [
                -1 if candidate.entity_name is None else entity_to_id[candidate.entity_name]
                for candidate in selected
            ],
            dtype=np.int16,
        ),
        face_id=np.asarray([candidate.face_id for candidate in selected], dtype=np.int32),
        barycentric=np.asarray(
            [candidate.barycentric for candidate in selected], dtype=np.float32
        ),
        normal_offset_m=np.asarray(
            [candidate.normal_offset_m for candidate in selected], dtype=np.float32
        ),
        env_world_xyz=np.asarray(
            [candidate.env_world_xyz for candidate in selected], dtype=np.float32
        ),
        queries_xytv=np.asarray(
            [candidate.query_xytv for candidate in selected], dtype=np.float32
        ),
    )


def apply_syn4d_body_transform(
    vertices: np.ndarray,
    body_pose_matrix: np.ndarray,
    body_translation: np.ndarray,
    *,
    trans: np.ndarray | None,
) -> np.ndarray:
    """Apply the official body/clothing world transform.

    Naked BEDLAM2 body vertices pass ``trans``.  Clothing passes ``None``;
    the official reader intentionally does not apply ``trans`` to clothing.
    """

    transformed = np.asarray(vertices, dtype=np.float32)
    pose = np.asarray(body_pose_matrix, dtype=np.float32)
    translation = np.asarray(body_translation, dtype=np.float32)
    if trans is not None:
        transformed = np.matmul(
            transformed, np.swapaxes(np.asarray(trans, dtype=np.float32), -1, -2)
        )
    transformed = np.matmul(transformed, np.swapaxes(pose, -1, -2))
    return np.ascontiguousarray(transformed + translation[..., None, :], dtype=np.float32)


def _euler_rotation(yaw_degrees: float, pitch_degrees: float, roll_degrees: float) -> np.ndarray:
    yaw, pitch, roll = np.radians([yaw_degrees, pitch_degrees, roll_degrees])
    r_yaw = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]],
        dtype=np.float64,
    )
    r_pitch = np.array(
        [[np.cos(pitch), 0, -np.sin(pitch)], [0, 1, 0], [np.sin(pitch), 0, np.cos(pitch)]],
        dtype=np.float64,
    )
    r_roll = np.array(
        [[1, 0, 0], [0, np.cos(roll), np.sin(roll)], [0, -np.sin(roll), np.cos(roll)]],
        dtype=np.float64,
    )
    return np.asarray(r_yaw @ r_pitch @ r_roll, dtype=np.float32)


def syn4d_actor_rotation(
    yaw_degrees: float, pitch_degrees: float, roll_degrees: float
) -> np.ndarray:
    """Return the official BEDLAM2 body/clothing pose rotation."""

    object_to_world_axes = np.array(
        [[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32
    )
    return np.asarray(
        _euler_rotation(yaw_degrees, pitch_degrees, roll_degrees)
        @ object_to_world_axes,
        dtype=np.float32,
    )


def syn4d_actor_world_vertices(
    vertices: np.ndarray,
    *,
    yaw_degrees: float,
    pitch_degrees: float,
    roll_degrees: float,
    position_cm: Sequence[float],
    global_translation_m: Sequence[float],
    naked_body: bool,
) -> np.ndarray:
    """Place a BEDLAM2 naked body or clothing mesh in Syn4D world space."""

    bedlam_to_syn4d = np.array(
        [[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32
    )
    translation = np.asarray(position_cm, dtype=np.float32) / np.float32(100.0)
    translation += np.asarray(global_translation_m, dtype=np.float32)
    return apply_syn4d_body_transform(
        vertices,
        syn4d_actor_rotation(yaw_degrees, pitch_degrees, roll_degrees),
        translation,
        trans=bedlam_to_syn4d if naked_body else None,
    )


def tile_syn4d_object_animation(vertices: np.ndarray, *, frame_count: int) -> np.ndarray:
    """Mirror Syn4D's tiled object animation and one-frame source offset."""

    source = np.asarray(vertices, dtype=np.float32)
    if source.ndim != 3 or source.shape[0] == 0 or source.shape[2] != 3:
        raise ValueError("object vertices must be non-empty [F,M,3]")
    repeats = (frame_count + 1 + source.shape[0] - 1) // source.shape[0]
    tiled = np.tile(source, (repeats, 1, 1))[1 : frame_count + 1].copy()
    tiled[..., 1] *= -1.0
    return tiled


def syn4d_static_object_world_vertices(
    vertices: np.ndarray,
    *,
    frame_count: int,
    yaw_degrees: float,
    pitch_degrees: float,
    roll_degrees: float,
    position_cm: Sequence[float],
    global_translation_m: Sequence[float],
    scale: float,
    shift_cm: float = 0.0,
) -> np.ndarray:
    """Apply the pinned no-keyframe Objaverse transform."""

    local = tile_syn4d_object_animation(vertices, frame_count=frame_count)
    local_shift = np.zeros((frame_count, 1, 3), dtype=np.float32)
    local_shift[:, 0, 1] = np.linspace(0.0, float(shift_cm), frame_count) / 100.0
    rotation = _euler_rotation(yaw_degrees, pitch_degrees, roll_degrees)
    translation = np.asarray(position_cm, dtype=np.float32) / np.float32(100.0)
    translation += np.asarray(global_translation_m, dtype=np.float32)
    world = (local * np.float32(scale) + local_shift) @ rotation.T
    return np.ascontiguousarray(world + translation[None, None], dtype=np.float32)


def interpolate_syn4d_object_motion(
    keyframe_frames: np.ndarray,
    keyframe_poses: np.ndarray,
    *,
    frame_count: int,
) -> np.ndarray:
    """Interpolate x/y/z/yaw/pitch/roll/scale and clamp sequence ends."""

    frames = np.asarray(keyframe_frames, dtype=np.float32)
    poses = np.asarray(keyframe_poses, dtype=np.float32)
    if frames.ndim != 1 or poses.shape != (frames.size, 7) or frames.size == 0:
        raise ValueError("keyframes must be [K] with poses [K,7]")
    if np.any(np.diff(frames) <= 0.0):
        raise ValueError("keyframe frame numbers must be strictly increasing")
    targets = np.arange(frame_count, dtype=np.float32)
    return np.stack(
        [np.interp(targets, frames, poses[:, column]) for column in range(7)], axis=-1
    ).astype(np.float32)


def syn4d_moving_object_world_vertices(
    vertices: np.ndarray,
    *,
    keyframe_frames: np.ndarray,
    keyframe_poses: np.ndarray,
    frame_count: int,
    global_translation_m: Sequence[float],
) -> np.ndarray:
    """Apply linearly interpolated per-frame Objaverse transforms.

    Pose columns are ``x_cm, y_cm, z_cm, yaw, pitch, roll, scale``.
    """

    local = tile_syn4d_object_animation(vertices, frame_count=frame_count)
    poses = interpolate_syn4d_object_motion(
        keyframe_frames, keyframe_poses, frame_count=frame_count
    )
    rotations = np.stack(
        [_euler_rotation(*pose[3:6]) for pose in poses], axis=0
    )
    scaled = local * poses[:, None, 6:7]
    world = np.einsum("fvi,fji->fvj", scaled, rotations, optimize=True)
    translations = poses[:, :3] / np.float32(100.0)
    translations += np.asarray(global_translation_m, dtype=np.float32)
    return np.ascontiguousarray(world + translations[:, None], dtype=np.float32)


def reconstruct_surface_tracks(
    bank: SurfaceBank,
    meshes: Mapping[str, EntityMesh],
    *,
    device: str = "cpu",
    point_batch_size: int = 4_096,
    frame_batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct explicit full-sequence tracks with batched PyTorch gathers."""

    import torch

    if set(meshes) != set(bank.entity_names):
        raise ValueError("entity meshes must exactly match the surface bank")
    frame_counts = {np.asarray(mesh.vertices).shape[0] for mesh in meshes.values()}
    if len(frame_counts) != 1:
        raise ValueError("all entity meshes must have the same frame count")
    frames = frame_counts.pop()
    tracks = np.zeros((frames, bank.count, 3), dtype=np.float32)
    valid = np.zeros((frames, bank.count), dtype=bool)

    environment = bank.entity_id == -1
    if np.any(environment):
        xyz = bank.env_world_xyz[environment]
        finite = np.isfinite(xyz).all(axis=-1)
        tracks[:, environment] = xyz[None]
        valid[:, environment] = finite[None]
        tracks[:, environment][:, ~finite] = 0.0

    for entity_id, entity_name in enumerate(bank.entity_names):
        point_indices = np.flatnonzero(bank.entity_id == entity_id)
        if point_indices.size == 0:
            continue
        mesh = meshes[entity_name]
        vertices = np.asarray(mesh.vertices, dtype=np.float32)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if vertices.ndim != 3 or vertices.shape[2] != 3:
            raise ValueError(f"{entity_name} vertices must be [F,M,3]")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"{entity_name} faces must be [K,3]")
        frame_valid = (
            np.ones(frames, dtype=bool)
            if mesh.frame_valid is None
            else np.asarray(mesh.frame_valid, dtype=bool)
        )
        if frame_valid.shape != (frames,):
            raise ValueError(f"{entity_name} frame_valid must be [F]")
        face_ids = bank.face_id[point_indices]
        if np.any(face_ids < 0) or np.any(face_ids >= len(faces)):
            raise ValueError(f"{entity_name} surface face outside mesh topology")
        selected_vertex_ids = torch.as_tensor(
            faces[face_ids], dtype=torch.long, device=device
        )

        for frame_start in range(0, frames, frame_batch_size):
            frame_stop = min(frame_start + frame_batch_size, frames)
            vertex_batch = torch.as_tensor(
                vertices[frame_start:frame_stop], dtype=torch.float32, device=device
            )
            valid_frames = torch.as_tensor(
                frame_valid[frame_start:frame_stop], dtype=torch.bool, device=device
            )
            for point_start in range(0, point_indices.size, point_batch_size):
                point_stop = min(point_start + point_batch_size, point_indices.size)
                local_indices = point_indices[point_start:point_stop]
                corners = vertex_batch[:, selected_vertex_ids[point_start:point_stop], :]
                barycentric = torch.as_tensor(
                    bank.barycentric[local_indices], dtype=torch.float32, device=device
                )
                offset = torch.as_tensor(
                    bank.normal_offset_m[local_indices], dtype=torch.float32, device=device
                )
                base = torch.einsum("bpvc,pv->bpc", corners, barycentric)
                normals = torch.linalg.cross(
                    corners[:, :, 1] - corners[:, :, 0],
                    corners[:, :, 2] - corners[:, :, 0],
                )
                normal_length = torch.linalg.vector_norm(normals, dim=-1)
                finite = (
                    torch.isfinite(corners).all(dim=-1).all(dim=-1)
                    & torch.isfinite(base).all(dim=-1)
                    & torch.isfinite(normal_length)
                    & (normal_length > 1e-8)
                )
                unit_normals = normals / normal_length.clamp_min(1e-8)[..., None]
                reconstructed = base + unit_normals * offset[None, :, None]
                current_valid = (
                    valid_frames[:, None]
                    & finite
                    & torch.isfinite(reconstructed).all(dim=-1)
                )
                reconstructed = torch.where(
                    current_valid[..., None], reconstructed, torch.zeros_like(reconstructed)
                )
                tracks[frame_start:frame_stop, local_indices] = (
                    reconstructed.detach().cpu().numpy()
                )
                valid[frame_start:frame_stop, local_indices] = (
                    current_valid.detach().cpu().numpy()
                )
    return tracks, valid


def motion_path_length(tracks_xyz: np.ndarray, track_valid: np.ndarray) -> np.ndarray:
    tracks = np.asarray(tracks_xyz, dtype=np.float32)
    valid = np.asarray(track_valid, dtype=bool)
    if tracks.ndim != 3 or tracks.shape[2] != 3 or valid.shape != tracks.shape[:2]:
        raise ValueError("tracks_xyz/track_valid must be [F,N,3] and [F,N]")
    adjacent = valid[:-1] & valid[1:]
    distance = np.linalg.norm(tracks[1:] - tracks[:-1], axis=-1)
    distance[~adjacent] = 0.0
    return distance.sum(axis=0, dtype=np.float32)


def depth_centimetres_to_metres(depth_cm: np.ndarray) -> np.ndarray:
    """Convert Syn4D optical-Z depth once and retain unquantized float32."""

    depth_m = np.asarray(depth_cm, dtype=np.float32) / np.float32(100.0)
    depth_m = np.array(depth_m, dtype=np.float32, copy=True)
    depth_m[~np.isfinite(depth_m) | (depth_m <= 0.0)] = 0.0
    return depth_m


def resize_depth_validity_weighted(
    depths_m: np.ndarray,
    *,
    height: int = CACHE_HEIGHT,
    width: int = CACHE_WIDTH,
    device: str = "cpu",
) -> np.ndarray:
    """Bilinearly resize float depth without mixing invalid zero pixels."""

    import torch
    import torch.nn.functional as torch_functional

    depths = np.asarray(depths_m, dtype=np.float32)
    if depths.ndim < 2:
        raise ValueError("depths_m must have image dimensions")
    leading_shape = depths.shape[:-2]
    tensor = torch.as_tensor(depths.reshape(-1, 1, *depths.shape[-2:]), device=device)
    valid = torch.isfinite(tensor) & (tensor > 0.0)
    numerator = torch_functional.interpolate(
        torch.where(valid, tensor, torch.zeros_like(tensor)),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    weight = torch_functional.interpolate(
        valid.to(torch.float32),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    resized = torch.where(weight > 0.0, numerator / weight.clamp_min(1e-12), 0.0)
    return resized.cpu().numpy().reshape(*leading_shape, height, width).astype(np.float32)


def _camera_rotation(yaw_degrees: float, pitch_degrees: float, roll_degrees: float) -> np.ndarray:
    zxy_to_xyz = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]], dtype=np.float64)
    return np.asarray(
        _euler_rotation(yaw_degrees, pitch_degrees, roll_degrees) @ zxy_to_xyz,
        dtype=np.float32,
    )


def camera_from_syn4d_row(
    row: Mapping[str, str], *, source_width: int, source_height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one official camera CSV row into K and 4x4 world-to-camera."""

    focal = float(row["focal_length"])
    sensor_width = float(row["sensor_width"])
    sensor_height = float(row["sensor_height"])
    intrinsics = np.array(
        [
            [focal / sensor_width * source_width, 0.0, source_width / 2.0],
            [0.0, focal / sensor_height * source_height, source_height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    camera_to_world = np.eye(4, dtype=np.float32)
    camera_to_world[:3, :3] = _camera_rotation(
        float(row["yaw"]), float(row["pitch"]), float(row["roll"])
    )
    camera_to_world[:3, 3] = np.asarray(
        [float(row["x"]), float(row["y"]), float(row["z"])], dtype=np.float32
    ) / np.float32(100.0)
    return intrinsics, np.asarray(np.linalg.inv(camera_to_world), dtype=np.float32)


def resize_intrinsics(
    intrinsics: np.ndarray,
    *,
    source_width: int,
    source_height: int,
    target_width: int = CACHE_WIDTH,
    target_height: int = CACHE_HEIGHT,
) -> np.ndarray:
    """Apply the same direct resize used for RGB and depth to camera K."""

    cameras = np.asarray(intrinsics, dtype=np.float32)
    if cameras.shape[-2:] != (3, 3):
        raise ValueError("intrinsics must end in [3,3]")
    transform = np.array(
        [
            [target_width / source_width, 0.0, 0.0],
            [0.0, target_height / source_height, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return np.ascontiguousarray(np.einsum("ij,...jk->...ik", transform, cameras))


def compute_depth_visibility(
    world_tracks: np.ndarray,
    track_valid: np.ndarray,
    depths_m: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics_w2c: np.ndarray,
    *,
    device: str = "cpu",
    absolute_tolerance_m: float = 0.01,
    relative_tolerance: float = 0.01,
    point_batch_size: int = 4_096,
    frame_batch_size: int = 32,
) -> np.ndarray:
    """Project tracks and test optical-Z against a 3x3 depth neighbourhood."""

    import torch

    tracks = np.asarray(world_tracks, dtype=np.float32)
    valid = np.asarray(track_valid, dtype=bool)
    depths = np.asarray(depths_m, dtype=np.float32)
    cameras = np.asarray(intrinsics, dtype=np.float32)
    extrinsics = np.asarray(extrinsics_w2c, dtype=np.float32)
    if tracks.ndim != 3 or tracks.shape[2] != 3 or valid.shape != tracks.shape[:2]:
        raise ValueError("tracks/valid must be [F,N,3] and [F,N]")
    if depths.ndim != 4:
        raise ValueError("depths_m must be [V,F,H,W]")
    views, frames, height, width = depths.shape
    if tracks.shape[0] != frames or cameras.shape != (views, frames, 3, 3):
        raise ValueError("depth, tracks, and intrinsics dimensions disagree")
    if extrinsics.shape != (views, frames, 4, 4):
        raise ValueError("extrinsics_w2c must be [V,F,4,4]")

    visibility = np.zeros((views, frames, tracks.shape[1]), dtype=bool)
    for view in range(views):
        for frame_start in range(0, frames, frame_batch_size):
            frame_stop = min(frame_start + frame_batch_size, frames)
            depth_batch = torch.as_tensor(
                depths[view, frame_start:frame_stop], dtype=torch.float32, device=device
            )
            camera_batch = torch.as_tensor(
                cameras[view, frame_start:frame_stop], dtype=torch.float32, device=device
            )
            extrinsic_batch = torch.as_tensor(
                extrinsics[view, frame_start:frame_stop, :3],
                dtype=torch.float32,
                device=device,
            )
            batch_frames = frame_stop - frame_start
            frame_index = torch.arange(batch_frames, device=device)[:, None]
            for point_start in range(0, tracks.shape[1], point_batch_size):
                point_stop = min(point_start + point_batch_size, tracks.shape[1])
                track_batch = torch.as_tensor(
                    tracks[frame_start:frame_stop, point_start:point_stop],
                    dtype=torch.float32,
                    device=device,
                )
                valid_batch = torch.as_tensor(
                    valid[frame_start:frame_stop, point_start:point_stop],
                    dtype=torch.bool,
                    device=device,
                )
                homogeneous = torch.cat(
                    [track_batch, torch.ones_like(track_batch[..., :1])], dim=-1
                )
                camera_xyz = torch.einsum("bij,bpj->bpi", extrinsic_batch, homogeneous)
                projected = torch.einsum("bij,bpj->bpi", camera_batch, camera_xyz)
                uv = projected[..., :2] / projected[..., 2:].clamp_min(1e-12)
                z = camera_xyz[..., 2]
                finite = (
                    valid_batch
                    & torch.isfinite(uv).all(dim=-1)
                    & torch.isfinite(z)
                    & (z > 0.0)
                )
                safe_uv = torch.where(torch.isfinite(uv), uv, torch.zeros_like(uv))
                x = torch.round(safe_uv[..., 0]).to(torch.long)
                y = torch.round(safe_uv[..., 1]).to(torch.long)
                inside = finite & (x >= 0) & (x < width) & (y >= 0) & (y < height)
                min_residual = torch.full_like(z, torch.inf)
                for delta_y in (-1, 0, 1):
                    for delta_x in (-1, 0, 1):
                        sample_x = x + delta_x
                        sample_y = y + delta_y
                        sample_inside = (
                            inside
                            & (sample_x >= 0)
                            & (sample_x < width)
                            & (sample_y >= 0)
                            & (sample_y < height)
                        )
                        sampled = depth_batch[
                            frame_index,
                            sample_y.clamp(0, height - 1),
                            sample_x.clamp(0, width - 1),
                        ]
                        residual = torch.where(
                            sample_inside & torch.isfinite(sampled) & (sampled > 0.0),
                            torch.abs(sampled - z),
                            torch.inf,
                        )
                        min_residual = torch.minimum(min_residual, residual)
                tolerance = torch.maximum(
                    torch.full_like(z, absolute_tolerance_m),
                    relative_tolerance * torch.abs(z),
                )
                visible = inside & (min_residual <= tolerance)
                visibility[
                    view, frame_start:frame_stop, point_start:point_stop
                ] = visible.cpu().numpy()
    return visibility


@dataclass(frozen=True)
class SequenceCacheWriter:
    destination: Path
    scene: str
    sequence_base: str
    frame_count: int
    track_count: int
    height: int
    width: int
    view_count: int

    def array(self, name: str) -> np.memmap:
        return np.load(self.destination / f"{name}.npy", mmap_mode="r+")

    def view_array(self, view: int, name: str) -> np.memmap:
        return np.load(self.destination / str(view) / f"{name}.npy", mmap_mode="r+")


def _preallocate(path: Path, *, dtype: np.dtype | str, shape: tuple[int, ...]) -> None:
    array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    array.flush()
    del array


def create_sequence_cache(
    destination: Path,
    *,
    scene: str,
    sequence_base: str,
    frame_count: int,
    track_count: int = TRACK_COUNT,
    height: int = CACHE_HEIGHT,
    width: int = CACHE_WIDTH,
    view_count: int = VIEW_COUNT,
) -> SequenceCacheWriter:
    """Preallocate one DIEGESIS-style Syn4D sequence cache."""

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    _preallocate(destination / "tracks_xyz.npy", dtype=np.float32, shape=(frame_count, track_count, 3))
    _preallocate(destination / "track_valid.npy", dtype=bool, shape=(frame_count, track_count))
    _preallocate(destination / "motion_path_length.npy", dtype=np.float32, shape=(track_count,))
    _preallocate(destination / "queries_xytv.npy", dtype=np.float32, shape=(track_count, 4))
    for view in range(view_count):
        view_root = destination / str(view)
        view_root.mkdir()
        _preallocate(view_root / "depth.npy", dtype=np.float32, shape=(frame_count, height, width))
        _preallocate(view_root / "intrinsics.npy", dtype=np.float32, shape=(frame_count, 3, 3))
        _preallocate(view_root / "extrinsics_w2c.npy", dtype=np.float32, shape=(frame_count, 4, 4))
        _preallocate(view_root / "visibility.npy", dtype=bool, shape=(frame_count, track_count))
        (destination / f"view_{view}").mkdir()
    return SequenceCacheWriter(
        destination=destination,
        scene=scene,
        sequence_base=sequence_base,
        frame_count=frame_count,
        track_count=track_count,
        height=height,
        width=width,
        view_count=view_count,
    )


def finalize_sequence_cache(
    writer: SequenceCacheWriter,
    *,
    fps: float,
    source_width: int,
    source_height: int,
) -> Path:
    """Validate array contracts and write ``manifest.json`` as completion marker."""

    global_specs = {
        "tracks_xyz": (np.dtype("float32"), (writer.frame_count, writer.track_count, 3)),
        "track_valid": (np.dtype("bool"), (writer.frame_count, writer.track_count)),
        "motion_path_length": (np.dtype("float32"), (writer.track_count,)),
        "queries_xytv": (np.dtype("float32"), (writer.track_count, 4)),
    }
    view_specs = {
        "depth": (np.dtype("float32"), (writer.frame_count, writer.height, writer.width)),
        "intrinsics": (np.dtype("float32"), (writer.frame_count, 3, 3)),
        "extrinsics_w2c": (np.dtype("float32"), (writer.frame_count, 4, 4)),
        "visibility": (np.dtype("bool"), (writer.frame_count, writer.track_count)),
    }
    for name, (dtype, shape) in global_specs.items():
        array = np.load(writer.destination / f"{name}.npy", mmap_mode="r")
        if array.dtype != dtype or array.shape != shape:
            raise ValueError(f"invalid cache array contract for {name}")
    for view in range(writer.view_count):
        for name, (dtype, shape) in view_specs.items():
            array = np.load(writer.destination / str(view) / f"{name}.npy", mmap_mode="r")
            if array.dtype != dtype or array.shape != shape:
                raise ValueError(f"invalid cache array contract for view {view}/{name}")
        jpeg_root = writer.destination / f"view_{view}"
        offsets = np.load(jpeg_root / "jpeg_offsets.npy", allow_pickle=False)
        byte_count = (jpeg_root / "jpeg_bytes.bin").stat().st_size
        if (
            offsets.dtype != np.int64
            or offsets.shape != (writer.frame_count + 1,)
            or offsets[0] != 0
            or offsets[-1] != byte_count
            or np.any(offsets[1:] <= offsets[:-1])
        ):
            raise ValueError(f"invalid JPEG store contract for view {view}")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be positive and finite")
    manifest = {
        "format": CACHE_FORMAT,
        "scene": writer.scene,
        "sequence_base": writer.sequence_base,
        "frames": writer.frame_count,
        "tracks": writer.track_count,
        "views": writer.view_count,
        "cache_resolution": [writer.height, writer.width],
        "source_resolution": [source_height, source_width],
        "fps": float(fps),
        "coordinate_frame": "syn4d_world_metres",
        "depth": "float32_optical_z_metres_zero_invalid",
        "rgb": "jpeg_quality_95_rgb",
        "queries": "cache_pixel_xytv",
    }
    temporary = writer.destination / ".manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = writer.destination / "manifest.json"
    temporary.replace(manifest_path)
    return manifest_path


def convert_syn4d_sequence(
    scene_root: Path,
    metadata_root: Path,
    output_root: Path,
    *,
    official_visualizer_root: Path,
    sequence: str,
    device: str = "cuda",
    progress=None,
):
    """Lazy public entry point for the single-sequence converter."""

    from mvtracker.preprocessing.syn4d_conversion import (
        convert_syn4d_sequence as run_conversion,
    )

    return run_conversion(
        scene_root,
        metadata_root,
        output_root,
        official_visualizer_root=official_visualizer_root,
        sequence=sequence,
        device=device,
        progress=progress,
    )


__all__ = [
    "CACHE_FORMAT",
    "CACHE_HEIGHT",
    "CACHE_WIDTH",
    "EntityMesh",
    "SequenceCacheWriter",
    "SequenceDependencies",
    "SurfaceBank",
    "SurfaceCandidate",
    "TEMPLE_GROUP_SCENE",
    "TEMPLE_GROUP_SEQUENCE_BASES",
    "TRACK_COUNT",
    "TempleGroupDependencies",
    "apply_syn4d_body_transform",
    "interpolate_syn4d_object_motion",
    "build_surface_bank",
    "camera_from_syn4d_row",
    "compact_surface_candidates",
    "compute_depth_visibility",
    "convert_syn4d_sequence",
    "create_sequence_cache",
    "depth_centimetres_to_metres",
    "discover_temple_group_sequences",
    "dynamic_surface_coordinates",
    "environment_surface_candidates",
    "finalize_sequence_cache",
    "motion_path_length",
    "reconstruct_surface_tracks",
    "resize_depth_validity_weighted",
    "resize_intrinsics",
    "sample_candidate_quarter",
    "sequence_dependencies",
    "syn4d_actor_rotation",
    "syn4d_actor_world_vertices",
    "syn4d_moving_object_world_vertices",
    "syn4d_static_object_world_vertices",
    "temple_group_dependencies",
    "tile_syn4d_object_animation",
]
