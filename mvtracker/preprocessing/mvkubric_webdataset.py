"""Convert native MV-Kubric scenes to indexed, uncompressed WebDataset shards.

Each scene contributes one metadata sample and one media sample per source
view.  The media samples retain the native encoded PNG/TIFF bytes and store a
small offsets array so a random-access loader can fetch only selected views.
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


WEB_DATASET_FORMAT = "mvtracker-kubric-webdataset"
SCENES_PER_SHARD = 4
META_COMPONENT = "meta.npz"
RGB_COMPONENT = "rgb.npz"
DEPTH_COMPONENT = "depth.npz"
WIDS_INDEX = "shards.json"
CATALOG = "catalog.json"


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
    """Pack encoded files without decoding or changing their bytes."""
    payloads: list[bytes] = []
    offsets = [0]
    for path in paths:
        payload = path.read_bytes()
        payloads.append(payload)
        offsets.append(offsets[-1] + len(payload))
    return _npz_bytes(
        bytes=np.frombuffer(b"".join(payloads), dtype=np.uint8),
        offsets=np.asarray(offsets, dtype=np.int64),
    )


def _load_camera(scene_root: Path, view_names: Sequence[str]) -> dict[str, np.ndarray]:
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
        sensor_widths.append(float(camera["sensor_width"]))
        focal_lengths.append(float(camera["focal_length"]))
        current_resolution = tuple(int(value) for value in metadata["metadata"]["resolution"])
        if resolution is None:
            resolution = current_resolution
        elif resolution != current_resolution:
            raise ValueError(f"{scene_root}: views have different resolutions")
    if resolution is None:
        raise ValueError(f"{scene_root}: no views found")
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
        "resolution_hw": np.asarray((resolution[1], resolution[0]), dtype=np.int32),
        "invalid_frame_indices": np.asarray(invalid, dtype=np.int64),
    }


def _load_tracks(
    scene_root: Path,
    view_names: Sequence[str],
) -> dict[str, np.ndarray]:
    with np.load(scene_root / "tracks_3d.npz") as payload:
        tracks_3d = np.asarray(payload["tracks_3d"], dtype=np.float32)
    if tracks_3d.ndim != 3 or tracks_3d.shape[-1] != 3:
        raise ValueError(f"{scene_root}: tracks_3d must have shape (frames, tracks, 3)")
    occlusion: list[np.ndarray] = []
    for view_name in view_names:
        with np.load(scene_root / view_name / "tracks_2d.npz") as payload:
            hidden = np.asarray(payload["occlusion"], dtype=np.bool_)
        expected = tracks_3d.shape[:2]
        if hidden.shape != expected:
            raise ValueError(f"{scene_root}/{view_name}: track metadata shape does not match tracks_3d")
        occlusion.append(hidden)
    return {"tracks_3d": tracks_3d, "visibility": ~np.stack(occlusion)}


def _view_paths(scene_root: Path) -> tuple[Path, ...]:
    paths = tuple(
        sorted(
            (path for path in Path(scene_root).iterdir() if path.is_dir() and path.name.startswith("view_")),
            key=lambda path: int(path.name.rsplit("_", 1)[1]),
        )
    )
    if not paths:
        raise ValueError(f"{scene_root}: no view directories found")
    return paths


def _scene_records(
    scene_root: Path, scene_id: str, read_workers: int
) -> tuple[bytes, dict[int, tuple[bytes, bytes]]]:
    view_paths = _view_paths(scene_root)
    view_names = tuple(path.name for path in view_paths)
    camera = _load_camera(scene_root, view_names)
    tracks = _load_tracks(scene_root, view_names)
    n_frames = int(tracks["tracks_3d"].shape[0])
    if tracks["visibility"].shape != (len(view_paths), *tracks["tracks_3d"].shape[:2]):
        raise ValueError(f"{scene_root}: visibility shape does not match tracks_3d")
    meta = {**tracks, **camera, "scene_name": np.asarray(str(scene_id))}
    jobs: list[tuple[int, str, tuple[Path, ...]]] = []
    for view_index, view_path in enumerate(view_paths):
        rgb = tuple(sorted(view_path.glob("rgba_*")))
        depth = tuple(sorted(view_path.glob("depth_*")))
        if len(rgb) != n_frames or len(depth) != n_frames:
            raise ValueError(
                f"{scene_root}/{view_path.name}: expected {n_frames} RGB/depth frames, got {len(rgb)}/{len(depth)}"
            )
        jobs.extend(((view_index, "rgb", rgb), (view_index, "depth", depth)))
    with ThreadPoolExecutor(max_workers=read_workers) as executor:
        futures = {(view, kind): executor.submit(_packed_encoded_frames, paths) for view, kind, paths in jobs}
        media = {
            view: (futures[(view, "rgb")].result(), futures[(view, "depth")].result())
            for view in range(len(view_paths))
        }
    return _npz_bytes(**meta), media


def _scene_components(scene_root: Path, scene_id: str, read_workers: int) -> dict[str, bytes]:
    """Return the mixed sample components for one scene."""
    metadata, media = _scene_records(Path(scene_root), scene_id, read_workers)
    components = {f"scene-{scene_id}.{META_COMPONENT}": metadata}
    for view, (rgb, depth) in media.items():
        key = f"scene-{scene_id}-view-{view:02d}"
        components[f"{key}.{RGB_COMPONENT}"] = rgb
        components[f"{key}.{DEPTH_COMPONENT}"] = depth
    return components


def _scene_view_count(scene_root: Path, scene_id: str) -> int:
    return len(_view_paths(Path(scene_root) / str(scene_id)))


def _require_consistent_view_count(scene_root: Path, scene_ids: Sequence[str]) -> int:
    counts = {_scene_view_count(scene_root, scene_id) for scene_id in scene_ids}
    if len(counts) != 1:
        raise ValueError(f"scene view counts must be consistent within a split: {sorted(counts)}")
    return counts.pop()


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
    """Write one uncompressed TAR and return its sample inventory."""
    output_tar = Path(output_tar)
    output_tar.parent.mkdir(parents=True, exist_ok=True)
    partial = output_tar.with_suffix(output_tar.suffix + ".partial")
    partial.unlink(missing_ok=True)
    started = time.perf_counter()
    expected_view_count = _scene_view_count(scene_root, shard.scene_ids[0])
    sample_records: list[dict[str, object]] = []
    with tarfile.open(partial, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for completed, scene_id in enumerate(shard.scene_ids, start=1):
            view_count = _scene_view_count(scene_root, scene_id)
            if view_count != expected_view_count:
                raise ValueError(f"{shard.name}: scene {scene_id} has an inconsistent view count")
            metadata, media = _scene_records(Path(scene_root) / scene_id, scene_id, read_workers)
            meta_key = f"scene-{scene_id}"
            _tar_add_bytes(archive, f"{meta_key}.{META_COMPONENT}", metadata)
            sample_records.append({"key": meta_key, "scene": str(scene_id), "kind": "metadata"})
            for view in range(view_count):
                key = f"scene-{scene_id}-view-{view:02d}"
                rgb, depth = media[view]
                _tar_add_bytes(archive, f"{key}.{RGB_COMPONENT}", rgb)
                _tar_add_bytes(archive, f"{key}.{DEPTH_COMPONENT}", depth)
                sample_records.append({"key": key, "scene": str(scene_id), "view": view, "kind": "media"})
            if progress_callback is not None:
                progress_callback(shard, scene_id, completed, time.perf_counter() - started)
    partial.replace(output_tar)
    return {
        "name": shard.name,
        "scene_ids": list(shard.scene_ids),
        "tar": output_tar.name,
        "bytes": output_tar.stat().st_size,
        "seconds": time.perf_counter() - started,
        "view_count": expected_view_count,
        "sample_records": sample_records,
        "nsamples": len(sample_records),
    }


def build_wids_index(
    archives: Sequence[Path], index: Path, command: str = "widsindex"
) -> Path:
    """Create the standard WIDS shard-list descriptor for uncompressed TARs."""
    archives = tuple(Path(archive).resolve() for archive in archives)
    if not archives:
        raise ValueError("at least one archive is required")
    index = Path(index).resolve()
    index.parent.mkdir(parents=True, exist_ok=True)
    names = [archive.name for archive in archives]
    subprocess.run(
        [command, "create", "--output", str(index), *names],
        cwd=archives[0].parent,
        check=True,
    )
    return index


def _publish_catalog(split_root: Path, results: Sequence[dict[str, object]]) -> dict[str, object]:
    scenes: dict[str, dict[str, object]] = {}
    sample_index = 0
    for result in sorted(results, key=lambda item: str(item["name"])):
        for record in result["sample_records"]:
            scene = str(record["scene"])
            entry = scenes.setdefault(scene, {"metadata_index": None, "views": {}})
            if record["kind"] == "metadata":
                entry["metadata_index"] = sample_index
            else:
                entry["views"][str(int(record["view"]))] = {"media_index": sample_index}
            sample_index += 1
    if any(entry["metadata_index"] is None for entry in scenes.values()):
        raise RuntimeError("catalog contains a scene without metadata")
    catalog = {"scenes": scenes, "sample_count": sample_index}
    temporary = split_root / f"{CATALOG}.partial"
    temporary.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
    temporary.replace(split_root / CATALOG)
    return catalog


def convert_split(
    scene_root: Path,
    output_root: Path,
    scene_ids: Sequence[str],
    *,
    scenes_per_shard: int = SCENES_PER_SHARD,
    read_workers: int = 16,
    index_command: str = "widsindex",
    progress_callback=None,
) -> dict[str, object]:
    """Convert a split serially, publishing one WIDS descriptor and catalog."""
    return convert_shards(
        scene_root,
        output_root,
        scene_ids,
        scenes_per_shard=scenes_per_shard,
        shard_workers=1,
        read_workers=read_workers,
        index_command=index_command,
        progress_callback=progress_callback,
    )


def convert_shards(
    scene_root: Path,
    output_root: Path,
    scene_ids: Sequence[str],
    *,
    scenes_per_shard: int = SCENES_PER_SHARD,
    shard_workers: int = 1,
    read_workers: int = 16,
    index_command: str = "widsindex",
    progress_callback=None,
) -> dict[str, object]:
    """Convert all shards concurrently and publish a WIDS descriptor/catalog."""
    if shard_workers < 1 or read_workers < 1:
        raise ValueError("shard_workers and read_workers must be positive")
    if not scene_ids:
        raise ValueError("scene_ids must not be empty")
    split_root = Path(output_root)
    split_root.mkdir(parents=True, exist_ok=True)
    view_count = _require_consistent_view_count(scene_root, scene_ids)
    shards = split_scene_ids(scene_ids, scenes_per_shard)

    def convert_one(shard: SceneShard) -> dict[str, object]:
        archive = split_root / f"{shard.name}.tar"
        if archive.is_file():
            raise FileExistsError(f"refusing to overwrite existing shard: {archive}")
        return write_shard(
            scene_root, shard, archive, read_workers=read_workers, progress_callback=progress_callback
        )

    with ThreadPoolExecutor(max_workers=shard_workers) as executor:
        futures = {executor.submit(convert_one, shard): shard for shard in shards}
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if progress_callback is not None:
                progress_callback("shard", result, len(results), len(shards))
    results.sort(key=lambda result: str(result["name"]))
    index = build_wids_index(
        [split_root / str(result["tar"]) for result in results], split_root / WIDS_INDEX, command=index_command
    )
    catalog = _publish_catalog(split_root, results)
    manifest = {
        "format": WEB_DATASET_FORMAT,
        "split": split_root.name,
        "scenes_per_shard": scenes_per_shard,
        "view_count": view_count,
        "scene_ids": [scene for result in results for scene in result["scene_ids"]],
        "shards": [
            {key: value for key, value in result.items() if key != "sample_records"}
            for result in results
        ],
        "wids_descriptor": index.name,
        "catalog": CATALOG,
        "scenes": catalog["scenes"],
    }
    temporary = split_root / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(split_root / "manifest.json")
    return {**manifest, "catalog_data": catalog}


def read_component(payload: bytes) -> dict[str, np.ndarray]:
    """Decode one metadata or packed-byte component."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}
