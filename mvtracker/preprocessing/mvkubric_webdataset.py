"""Convert native MV-Kubric scenes to indexed, uncompressed WebDataset shards.

Each WebDataset sample is one complete scene.  Components intentionally hold
the original encoded frame bytes; DALI can seek to a sample using the index,
while MV-Tracker remains responsible for selecting a view subset and decoding
the selected frames.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time
from typing import Iterable, Sequence

import numpy as np


FORMAT_VERSION = 1
SCENES_PER_SHARD = 4
META_COMPONENT = "meta"
RGB_COMPONENTS = tuple(f"rgb{view}" for view in range(6))
DEPTH_COMPONENTS = tuple(f"depth{view}" for view in range(6))
COMPONENTS = (META_COMPONENT, *RGB_COMPONENTS, *DEPTH_COMPONENTS)


@dataclass(frozen=True)
class SceneShard:
    """A shard name and the scene IDs assigned to it."""

    name: str
    scene_ids: tuple[str, ...]


def discover_scene_ids(scene_root: Path, include: Iterable[str] | None = None) -> tuple[str, ...]:
    """Return numeric scene IDs in numeric order."""
    scene_root = Path(scene_root)
    allowed = None if include is None else {str(scene) for scene in include}
    scenes = sorted(
        path.name
        for path in scene_root.iterdir()
        if path.is_dir() and path.name.isdigit() and (allowed is None or path.name in allowed)
    )
    return tuple(sorted(scenes, key=int))


def split_scene_ids(
    scene_ids: Sequence[str], scenes_per_shard: int = SCENES_PER_SHARD
) -> tuple[SceneShard, ...]:
    if scenes_per_shard <= 0:
        raise ValueError("scenes_per_shard must be positive")
    ordered = tuple(sorted((str(scene) for scene in scene_ids), key=int))
    return tuple(
        SceneShard(
            name=f"mvkubric-{index:05d}",
            scene_ids=ordered[start : start + scenes_per_shard],
        )
        for index, start in enumerate(range(0, len(ordered), scenes_per_shard))
    )


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.savez(stream, **arrays)
    return stream.getvalue()


def _packed_encoded_frames(paths: Sequence[Path]) -> bytes:
    """Pack encoded files as one byte vector plus offsets without decoding."""
    payloads: list[bytes] = []
    offsets = [0]
    for path in paths:
        payload = path.read_bytes()
        payloads.append(payload)
        offsets.append(offsets[-1] + len(payload))
    return _npz_bytes(
        payload=np.frombuffer(b"".join(payloads), dtype=np.uint8),
        offsets=np.asarray(offsets, dtype=np.int64),
    )


def _load_tracks(scene_root: Path, view_names: Sequence[str]) -> dict[str, np.ndarray]:
    with np.load(scene_root / "tracks_3d.npz") as payload:
        tracks_3d = np.asarray(payload["tracks_3d"], dtype=np.float32)
    tracks_2d: list[np.ndarray] = []
    occlusion: list[np.ndarray] = []
    for view_name in view_names:
        with np.load(scene_root / view_name / "tracks_2d.npz") as payload:
            tracks_2d.append(np.asarray(payload["tracks_2d"], dtype=np.float32))
            occlusion.append(np.asarray(payload["occlusion"], dtype=np.bool_))
    return {
        "tracks_3d": tracks_3d,
        "tracks_2d": np.stack(tracks_2d),
        "occlusion": np.stack(occlusion),
    }


def _load_camera(scene_root: Path, view_names: Sequence[str]) -> dict[str, np.ndarray | str]:
    intrinsics = []
    extrinsics = []
    sensor_widths = []
    focal_lengths = []
    resolution: tuple[int, int] | None = None
    for view_name in view_names:
        metadata = json.loads((scene_root / view_name / "metadata.json").read_text())
        camera = metadata["camera"]
        intr = np.asarray(camera["K"], dtype=np.float64)
        positions = np.asarray(camera["positions"], dtype=np.float64)
        quaternions = np.asarray(camera["quaternions"], dtype=np.float64)
        quaternions /= np.linalg.norm(quaternions, axis=-1, keepdims=True)
        w, x, y, z = quaternions.T
        rotations = np.stack(
            (
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ),
            axis=-1,
        ).reshape(-1, 3, 3)
        camera_to_world = np.zeros((len(positions), 4, 4), dtype=np.float64)
        camera_to_world[:, :3, :3] = rotations
        camera_to_world[:, :3, 3] = positions
        camera_to_world[:, 3, 3] = 1.0
        extr = np.linalg.inv(camera_to_world)[:, :3]
        width, height = metadata["metadata"]["resolution"]
        intr = np.diag([width, height, 1.0]) @ intr @ np.diag([1.0, -1.0, -1.0])
        extr = np.diag([1.0, -1.0, -1.0]) @ extr
        intrinsics.append(intr.astype(np.float32))
        extrinsics.append(extr.astype(np.float32))
        sensor_widths.append(float(metadata["camera"]["sensor_width"]))
        focal_lengths.append(float(metadata["camera"]["focal_length"]))
        current_resolution = tuple(int(value) for value in metadata["metadata"]["resolution"])
        if resolution is None:
            resolution = current_resolution
        elif resolution != current_resolution:
            raise ValueError(f"{scene_root}: views have different resolutions")
    assert resolution is not None
    invalid = []
    scene_json = scene_root / "scene.json"
    if scene_json.is_file():
        scene = json.loads(scene_json.read_text())
        invalid = scene.get("output", {}).get("rgb", {}).get("invalid_frame_indices", [])
    return {
        "intrinsics": np.stack(intrinsics),
        "extrinsics": np.stack(extrinsics),
        "sensor_widths": np.asarray(sensor_widths, dtype=np.float32),
        "focal_lengths": np.asarray(focal_lengths, dtype=np.float32),
        "resolution": np.asarray(resolution, dtype=np.int32),
        "invalid_frame_indices": np.asarray(invalid, dtype=np.int64),
    }


def _scene_components(scene_root: Path, scene_id: str, read_workers: int) -> dict[str, bytes]:
    view_paths = sorted(
        (path for path in scene_root.iterdir() if path.is_dir() and path.name.startswith("view_")),
        key=lambda path: int(path.name.rsplit("_", 1)[1]),
    )
    if len(view_paths) != 6:
        raise ValueError(f"{scene_root}: expected six views, found {len(view_paths)}")
    view_names = tuple(path.name for path in view_paths)
    tracks = _load_tracks(scene_root, view_names)
    camera = _load_camera(scene_root, view_names)
    n_frames = int(tracks["tracks_3d"].shape[0])
    if tracks["tracks_2d"].shape[:2] != (6, n_frames):
        raise ValueError(f"{scene_root}: tracks_2d shape does not match tracks_3d")

    meta = {
        **tracks,
        **camera,
        "view_names": np.asarray(view_names),
        "scene_id": np.asarray(str(scene_id)),
    }
    jobs: list[tuple[str, tuple[Path, ...]]] = []
    for view_index, view_path in enumerate(view_paths):
        rgb = tuple(sorted(view_path.glob("rgba_*")))
        depth = tuple(sorted(view_path.glob("depth_*")))
        if len(rgb) != n_frames or len(depth) != n_frames:
            raise ValueError(
                f"{scene_root}: expected {n_frames} RGB/depth frames, got {len(rgb)}/{len(depth)}"
            )
        jobs.extend(((f"rgb{view_index}", rgb), (f"depth{view_index}", depth)))

    with ThreadPoolExecutor(max_workers=read_workers) as executor:
        futures = {name: executor.submit(_packed_encoded_frames, paths) for name, paths in jobs}
        encoded = {name: future.result() for name, future in futures.items()}
    return {META_COMPONENT: _npz_bytes(**meta), **encoded}


def _tar_add_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(payload)
    info.mtime = 0
    info.mode = 0o644
    archive.addfile(info, io.BytesIO(payload))


def write_shard(
    scene_root: Path,
    shard: SceneShard,
    output_tar: Path,
    *,
    read_workers: int = 16,
    progress_callback=None,
) -> dict[str, object]:
    """Write one uncompressed TAR and return its inventory."""
    output_tar = Path(output_tar)
    output_tar.parent.mkdir(parents=True, exist_ok=True)
    partial = output_tar.with_suffix(output_tar.suffix + ".partial")
    partial.unlink(missing_ok=True)
    started = time.perf_counter()
    with tarfile.open(partial, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for completed, scene_id in enumerate(shard.scene_ids, start=1):
            components = _scene_components(Path(scene_root) / scene_id, scene_id, read_workers)
            for component in COMPONENTS:
                _tar_add_bytes(archive, f"{scene_id}.{component}", components[component])
            if progress_callback is not None:
                progress_callback(shard, scene_id, completed, time.perf_counter() - started)
    partial.replace(output_tar)
    return {
        "name": shard.name,
        "scene_ids": list(shard.scene_ids),
        "tar": str(output_tar),
        "bytes": output_tar.stat().st_size,
        "seconds": time.perf_counter() - started,
        "components_per_scene": len(COMPONENTS),
    }


def build_wds_index(archive: Path, index: Path | None = None, command: str = "wds2idx") -> Path:
    """Create the DALI index with NVIDIA's standard ``wds2idx`` utility."""
    archive = Path(archive).resolve()
    index = Path(index or archive.with_suffix(".idx")).resolve()
    index.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([command, str(archive), str(index)], check=True)
    return index


def convert_split(
    scene_root: Path,
    output_root: Path,
    scene_ids: Sequence[str],
    *,
    scenes_per_shard: int = SCENES_PER_SHARD,
    read_workers: int = 16,
    index_command: str = "wds2idx",
    progress_callback=None,
) -> dict[str, object]:
    """Convert a split, publishing only completed TAR/index pairs."""
    split_root = Path(output_root)
    split_root.mkdir(parents=True, exist_ok=True)
    shards = split_scene_ids(scene_ids, scenes_per_shard)
    results: list[dict[str, object]] = []
    for shard in shards:
        archive = split_root / f"{shard.name}.tar"
        index = split_root / f"{shard.name}.idx"
        expected = set(shard.scene_ids)
        if archive.is_file() and index.is_file():
            result = {
                "name": shard.name,
                "scene_ids": list(shard.scene_ids),
                "tar": str(archive),
                "idx": str(index),
                "bytes": archive.stat().st_size,
                "status": "skipped-existing",
            }
            results.append(result)
            continue
        result = write_shard(scene_root, shard, archive, read_workers=read_workers, progress_callback=progress_callback)
        build_wds_index(archive, index, command=index_command)
        result["idx"] = str(index)
        result["status"] = "created"
        if set(result["scene_ids"]) != expected:
            raise RuntimeError(f"{shard.name}: scene inventory changed while converting")
        results.append(result)
    manifest = {
        "format": "mvtracker_mvkubric_webdataset",
        "version": FORMAT_VERSION,
        "split": split_root.name,
        "scenes_per_shard": scenes_per_shard,
        "components": list(COMPONENTS),
        "scene_ids": [scene for result in results for scene in result["scene_ids"]],
        "shards": results,
    }
    temporary = split_root / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(split_root / "manifest.json")
    return manifest


def convert_shards(
    scene_root: Path,
    output_root: Path,
    scene_ids: Sequence[str],
    *,
    scenes_per_shard: int = SCENES_PER_SHARD,
    shard_workers: int = 1,
    read_workers: int = 16,
    index_command: str = "wds2idx",
    progress_callback=None,
) -> dict[str, object]:
    """Convert all shards in a split concurrently and publish one manifest."""
    if shard_workers < 1:
        raise ValueError("shard_workers must be positive")
    split_root = Path(output_root)
    split_root.mkdir(parents=True, exist_ok=True)
    shards = split_scene_ids(scene_ids, scenes_per_shard)

    def convert_one(shard: SceneShard) -> dict[str, object]:
        archive = split_root / f"{shard.name}.tar"
        index = split_root / f"{shard.name}.idx"
        if archive.is_file() and index.is_file():
            return {
                "name": shard.name,
                "scene_ids": list(shard.scene_ids),
                "tar": str(archive),
                "idx": str(index),
                "bytes": archive.stat().st_size,
                "status": "skipped-existing",
            }
        result = write_shard(
            scene_root,
            shard,
            archive,
            read_workers=read_workers,
            progress_callback=progress_callback,
        )
        build_wds_index(archive, index, command=index_command)
        result["idx"] = str(index)
        result["status"] = "created"
        return result

    with ThreadPoolExecutor(max_workers=shard_workers) as executor:
        futures = {executor.submit(convert_one, shard): shard for shard in shards}
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if progress_callback is not None:
                progress_callback("shard", result, len(results), len(shards))
    results.sort(key=lambda result: result["name"])
    manifest = {
        "format": "mvtracker_mvkubric_webdataset",
        "version": FORMAT_VERSION,
        "split": split_root.name,
        "scenes_per_shard": scenes_per_shard,
        "components": list(COMPONENTS),
        "scene_ids": [scene for result in results for scene in result["scene_ids"]],
        "shards": results,
    }
    temporary = split_root / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(split_root / "manifest.json")
    return manifest


def read_component(payload: bytes) -> dict[str, np.ndarray]:
    """Decode one metadata or packed-byte component returned by DALI."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}
