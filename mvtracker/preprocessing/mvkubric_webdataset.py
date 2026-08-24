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
import tarfile
import time
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


WEB_DATASET_FORMAT = "mvtracker-kubric-webdataset"
SCENES_PER_SHARD = 4
META_COMPONENT = "meta.npz"
RGB_COMPONENT = "rgb.npz"
DEPTH_COMPONENT = "depth.npz"
RECORD_COMPONENTS = (META_COMPONENT, RGB_COMPONENT, DEPTH_COMPONENT)
RECORD_LOCATOR = "record-locator.npz"
RECORD_LOCATOR_FORMAT = "mvtracker-record-locator-v1"
WIDS_INDEX = "shards.json"
CATALOG = "catalog.json"
INVENTORY_SUFFIX = ".inventory.json"


@dataclass(frozen=True)
class SceneShard:
    """A shard name and the scene IDs assigned to it."""

    name: str
    scene_ids: tuple[str, ...]
    index: int = 0


@dataclass(frozen=True)
class DaliIndexComponent:
    extension: str
    offset: int
    size: int
    name: str


@dataclass(frozen=True)
class DaliIndexRecord:
    key: str
    components: tuple[DaliIndexComponent, ...]


def _split_webdataset_name(name: str) -> tuple[str, str]:
    dot = name.find(".", name.rfind("/") + 1)
    if dot < 0:
        raise ValueError(f"WebDataset component has no extension: {name}")
    return name[:dot], name[dot + 1 :]


def parse_dali_index(path: Path) -> tuple[DaliIndexRecord, ...]:
    """Parse one standard NVIDIA DALI WebDataset v1.2 index."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path}: empty DALI index")
    header = lines[0].split()
    if len(header) != 2 or header[0] != "v1.2":
        raise ValueError(f"{path}: expected a DALI v1.2 index")
    expected = int(header[1])
    records: list[DaliIndexRecord] = []
    for line_number, line in enumerate(lines[1:], start=2):
        fields = line.split()
        if len(fields) % 4:
            raise ValueError(f"{path}:{line_number}: malformed component quartet")
        components: list[DaliIndexComponent] = []
        record_key: str | None = None
        for start in range(0, len(fields), 4):
            extension, offset_text, size_text, name = fields[start : start + 4]
            key, name_extension = _split_webdataset_name(name)
            if extension != name_extension:
                raise ValueError(f"{path}:{line_number}: extension does not match {name}")
            if record_key is None:
                record_key = key
            elif record_key != key:
                raise ValueError(f"{path}:{line_number}: components have different sample keys")
            components.append(
                DaliIndexComponent(
                    extension=extension,
                    offset=int(offset_text),
                    size=int(size_text),
                    name=name,
                )
            )
        if record_key is None:
            raise ValueError(f"{path}:{line_number}: empty sample record")
        records.append(DaliIndexRecord(key=record_key, components=tuple(components)))
    if len(records) != expected:
        raise ValueError(f"{path}: header declares {expected} records, found {len(records)}")
    return tuple(records)


def build_record_locator(shards: Sequence[dict[str, object]], locator: Path) -> Path:
    """Build a compact global-record locator from adjacent DALI indexes."""
    if not shards:
        raise ValueError("at least one indexed shard is required")
    locator = Path(locator).resolve()
    locator.parent.mkdir(parents=True, exist_ok=True)
    shard_paths: list[str] = []
    record_shards: list[int] = []
    record_keys: list[str] = []
    offsets: list[list[int]] = []
    sizes: list[list[int]] = []
    component_slots = {name: index for index, name in enumerate(RECORD_COMPONENTS)}
    for shard_index, shard in enumerate(shards):
        relative_tar = Path(str(shard["tar"]))
        archive = relative_tar if relative_tar.is_absolute() else locator.parent / relative_tar
        index_path = archive.with_suffix(".idx")
        records = parse_dali_index(index_path)
        if len(records) != int(shard["nsamples"]):
            raise ValueError(
                f"{index_path}: expected {shard['nsamples']} records, found {len(records)}"
            )
        shard_paths.append(str(relative_tar))
        for record in records:
            record_offsets = [-1] * len(RECORD_COMPONENTS)
            record_sizes = [0] * len(RECORD_COMPONENTS)
            for component in record.components:
                try:
                    slot = component_slots[component.extension]
                except KeyError as error:
                    raise ValueError(
                        f"{index_path}: unsupported component {component.extension!r}"
                    ) from error
                record_offsets[slot] = component.offset
                record_sizes[slot] = component.size
            record_shards.append(shard_index)
            record_keys.append(record.key)
            offsets.append(record_offsets)
            sizes.append(record_sizes)
    temporary = locator.with_suffix(locator.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            format=np.asarray(RECORD_LOCATOR_FORMAT),
            shards=np.asarray(shard_paths),
            keys=np.asarray(record_keys),
            record_shards=np.asarray(record_shards, dtype=np.int32),
            component_names=np.asarray(RECORD_COMPONENTS),
            offsets=np.asarray(offsets, dtype=np.int64),
            sizes=np.asarray(sizes, dtype=np.int64),
        )
    temporary.replace(locator)
    return locator


def publish_record_locator(split_root: Path) -> Path:
    """Build and publish direct-record metadata for an existing split."""
    split_root = Path(split_root).resolve()
    manifest_path = split_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    locator = build_record_locator(manifest["shards"], split_root / RECORD_LOCATOR)
    manifest["record_locator"] = locator.name
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".partial")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return locator


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
    scene_ids: Sequence[str],
    scenes_per_shard: int = SCENES_PER_SHARD,
    shard_offset: int = 0,
) -> tuple[SceneShard, ...]:
    if scenes_per_shard <= 0:
        raise ValueError("scenes_per_shard must be positive")
    if shard_offset < 0:
        raise ValueError("shard_offset must be non-negative")
    ordered = tuple(sorted((str(scene) for scene in scene_ids), key=int))
    return tuple(
        SceneShard(
            name=f"mvkubric-{shard_offset + index:05d}",
            scene_ids=ordered[start : start + scenes_per_shard],
            index=shard_offset + index,
        )
        for index, start in enumerate(range(0, len(ordered), scenes_per_shard))
    )


def inventory_path(output_tar: Path) -> Path:
    """Return the adjacent JSON inventory path for a shard TAR."""
    output_tar = Path(output_tar)
    return output_tar.with_suffix(INVENTORY_SUFFIX)


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


def _packed_float_depth_frames(
    estimated_depth_root: Path,
    scene_id: str,
    view: int,
    *,
    cleaned: bool,
) -> bytes:
    """Encode float32 estimated depth as TIFF bytes for the DALI reader."""
    scene_root = Path(estimated_depth_root) / str(scene_id)
    if (scene_root / "manifest.json").is_file() and (scene_root / str(view) / "depth.npy").is_file():
        manifest = json.loads((scene_root / "manifest.json").read_text(encoding="utf-8"))
        depth = np.load(scene_root / str(view) / "depth.npy", mmap_mode="r", allow_pickle=False)
        cleaned_mask = np.load(
            scene_root / str(view) / "cleaned_mask.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        frame_getter = lambda frame_index: depth[frame_index]
        mask_getter = lambda frame_index: cleaned_mask[frame_index]
        depth_label = scene_root / str(view) / "depth.npy"
        mask_array = cleaned_mask
        frame_count = int(depth.shape[0])
    else:
        burst_roots = sorted(path for path in scene_root.glob("frames-*") if path.is_dir())
        if len(burst_roots) != 1:
            raise ValueError(f"{scene_root}: expected exactly one completed burst")
        burst_root = burst_roots[0]
        manifest = json.loads((burst_root / "manifest.json").read_text(encoding="utf-8"))
        depth = np.load(burst_root / "depth.npy", mmap_mode="r", allow_pickle=False)
        mask = np.load(burst_root / "cleaned_mask.npy", mmap_mode="r", allow_pickle=False)
        frame_getter = lambda frame_index: depth[view, frame_index]
        mask_getter = lambda frame_index: mask[view, frame_index]
        depth_label = burst_root / "depth.npy"
        mask_array = mask
        frame_count = int(depth.shape[1])
    if depth.dtype != np.float32:
        raise ValueError(f"{depth_label} must be float32")
    if cleaned:
        if mask_array.dtype != np.bool_ or mask_array.shape != depth.shape:
            raise ValueError(f"{depth_label.parent}: invalid cleaned mask")
    encoded: list[bytes] = []
    offsets = [0]
    for frame_index in range(frame_count):
        frame = np.asarray(frame_getter(frame_index), dtype=np.float32)
        if cleaned:
            frame = np.where(mask_getter(frame_index), frame, 0.0).astype(np.float32, copy=False)
        stream = io.BytesIO()
        Image.fromarray(frame, mode="F").save(stream, format="TIFF")
        payload = stream.getvalue()
        encoded.append(payload)
        offsets.append(offsets[-1] + len(payload))
    expected_frames = int(manifest["frame_count"])
    if frame_count != expected_frames:
        raise ValueError(f"{scene_root}/{view_name}: frame count does not match manifest")
    return _npz_bytes(
        bytes=np.frombuffer(b"".join(encoded), dtype=np.uint8),
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
    scene_root: Path,
    scene_id: str,
    read_workers: int,
    *,
    estimated_depth_root: Path | None = None,
    estimated_depth_cleaned: bool = False,
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
        jobs.append((view_index, "rgb", rgb))
        if estimated_depth_root is None:
            jobs.append((view_index, "depth", depth))
    with ThreadPoolExecutor(max_workers=read_workers) as executor:
        futures = {(view, kind): executor.submit(_packed_encoded_frames, paths) for view, kind, paths in jobs}
        media = {
            view: (
                futures[(view, "rgb")].result(),
                _packed_float_depth_frames(
                    estimated_depth_root,
                    scene_id,
                    view,
                    cleaned=estimated_depth_cleaned,
                )
                if estimated_depth_root is not None
                else futures[(view, "depth")].result(),
            )
            for view in range(len(view_paths))
        }
    return _npz_bytes(**meta), media


def _scene_components(
    scene_root: Path,
    scene_id: str,
    read_workers: int,
    *,
    estimated_depth_root: Path | None = None,
    estimated_depth_cleaned: bool = False,
) -> dict[str, bytes]:
    """Return the mixed sample components for one scene."""
    metadata, media = _scene_records(
        Path(scene_root),
        scene_id,
        read_workers,
        estimated_depth_root=estimated_depth_root,
        estimated_depth_cleaned=estimated_depth_cleaned,
    )
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
    estimated_depth_root: Path | None = None,
    estimated_depth_cleaned: bool = False,
    progress_callback=None,
) -> dict[str, object]:
    """Write one uncompressed TAR and return its sample inventory."""
    output_tar = Path(output_tar)
    output_tar.parent.mkdir(parents=True, exist_ok=True)
    partial = output_tar.with_suffix(output_tar.suffix + ".partial")
    partial.unlink(missing_ok=True)
    inventory = inventory_path(output_tar)
    inventory.unlink(missing_ok=True)
    started = time.perf_counter()
    expected_view_count = _scene_view_count(scene_root, shard.scene_ids[0])
    sample_records: list[dict[str, object]] = []
    with tarfile.open(partial, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for completed, scene_id in enumerate(shard.scene_ids, start=1):
            view_count = _scene_view_count(scene_root, scene_id)
            if view_count != expected_view_count:
                raise ValueError(f"{shard.name}: scene {scene_id} has an inconsistent view count")
            metadata, media = _scene_records(
                Path(scene_root) / scene_id,
                scene_id,
                read_workers,
                estimated_depth_root=estimated_depth_root,
                estimated_depth_cleaned=estimated_depth_cleaned,
            )
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
            print(
                f"CONVERT event=scene shard={shard.name} scene={scene_id} "
                f"progress={completed}/{len(shard.scene_ids)} "
                f"elapsed_seconds={time.perf_counter() - started:.1f}",
                flush=True,
            )
    partial.replace(output_tar)
    result = {
        "name": shard.name,
        "index": shard.index,
        "scene_ids": list(shard.scene_ids),
        "tar": output_tar.name,
        "inventory": inventory.name,
        "bytes": output_tar.stat().st_size,
        "seconds": time.perf_counter() - started,
        "view_count": expected_view_count,
        "sample_records": sample_records,
        "nsamples": len(sample_records),
    }
    inventory_partial = inventory.with_suffix(inventory.suffix + ".partial")
    inventory_partial.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    inventory_partial.replace(inventory)
    return result


def build_wids_index(
    shards: Sequence[dict[str, object]], index: Path
) -> Path:
    """Create the standard WIDS-v1 descriptor from completed shard inventories."""
    if not shards:
        raise ValueError("at least one archive is required")
    index = Path(index).resolve()
    index.parent.mkdir(parents=True, exist_ok=True)
    descriptor = {
        "__kind__": "wids-shard-index-v1",
        "wids_version": 1,
        "shardlist": [
            {
                "url": str(shard["tar"]),
                "nsamples": int(shard["nsamples"]),
                "filesize": int(shard["bytes"]),
            }
            for shard in shards
        ],
    }
    partial = index.with_suffix(index.suffix + ".partial")
    partial.write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")
    partial.replace(index)
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


def _load_inventory(path: Path) -> dict[str, object]:
    """Load one completed shard inventory."""
    with Path(path).open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"{path}: shard inventory must be a JSON object")
    required = {"name", "index", "scene_ids", "tar", "sample_records", "nsamples", "view_count"}
    missing = required.difference(result)
    if missing:
        raise ValueError(f"{path}: shard inventory is missing {sorted(missing)}")
    tar = Path(path).parent / str(result["tar"])
    if not tar.is_file():
        raise FileNotFoundError(f"{path}: inventory TAR does not exist: {tar}")
    if not isinstance(result["scene_ids"], list) or not isinstance(result["sample_records"], list):
        raise ValueError(f"{path}: invalid scene_ids or sample_records")
    return result


def finalize_shards(
    output_root: Path,
    expected_scene_ids: Sequence[str],
    *,
    scenes_per_shard: int = SCENES_PER_SHARD,
) -> dict[str, object]:
    """Publish WIDS metadata from all completed adjacent shard inventories.

    Inventories are the completion markers: a TAR without its inventory is not
    considered complete.  The finalizer requires exactly the expected scene
    set and contiguous global shard numbering.
    """
    if scenes_per_shard <= 0:
        raise ValueError("scenes_per_shard must be positive")
    split_root = Path(output_root)
    expected = tuple(sorted((str(scene) for scene in expected_scene_ids), key=int))
    if len(set(expected)) != len(expected):
        raise ValueError("expected_scene_ids must be unique")
    if not expected:
        raise ValueError("expected_scene_ids must not be empty")

    inventory_files = sorted(split_root.glob(f"*{INVENTORY_SUFFIX}"), key=lambda path: int(path.name.split("-")[-1].split(".")[0]))
    if not inventory_files:
        raise RuntimeError("no completed shard inventories found")
    expected_shard_count = len(inventory_files)
    results = [_load_inventory(path) for path in inventory_files]
    results.sort(key=lambda item: int(item["index"]))
    if [int(result["index"]) for result in results] != list(range(expected_shard_count)):
        raise RuntimeError("shard inventories do not have contiguous global numbering")
    observed = [str(scene) for result in results for scene in result["scene_ids"]]
    if observed != list(expected):
        raise RuntimeError("shard inventories do not contain exactly the expected scene IDs")

    index = build_wids_index(results, split_root / WIDS_INDEX)
    catalog = _publish_catalog(split_root, results)
    manifest = {
        "format": WEB_DATASET_FORMAT,
        "split": split_root.name,
        "scenes_per_shard": scenes_per_shard,
        "view_count": int(results[0]["view_count"]),
        "scene_ids": list(expected),
        "shards": [
            {key: value for key, value in result.items() if key != "sample_records"}
            for result in results
        ],
        "wids_descriptor": index.name,
        "catalog": CATALOG,
        "scenes": catalog["scenes"],
    }
    temporary = split_root / "manifest.json.partial"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(split_root / "manifest.json")
    return {**manifest, "catalog_data": catalog}


def convert_split(
    scene_root: Path,
    output_root: Path,
    scene_ids: Sequence[str],
    *,
    scenes_per_shard: int = SCENES_PER_SHARD,
    read_workers: int = 16,
    progress_callback=None,
    shard_offset: int = 0,
    finalize: bool = True,
    expected_scene_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Convert a split serially, publishing one WIDS descriptor and catalog."""
    return convert_shards(
        scene_root,
        output_root,
        scene_ids,
        scenes_per_shard=scenes_per_shard,
        shard_workers=1,
        read_workers=read_workers,
        progress_callback=progress_callback,
        shard_offset=shard_offset,
        finalize=finalize,
        expected_scene_ids=expected_scene_ids,
    )


def convert_shards(
    scene_root: Path,
    output_root: Path,
    scene_ids: Sequence[str],
    *,
    scenes_per_shard: int = SCENES_PER_SHARD,
    shard_workers: int = 1,
    read_workers: int = 16,
    progress_callback=None,
    shard_offset: int = 0,
    finalize: bool = True,
    expected_scene_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Convert all shards concurrently, reusing completed inventory pairs.

    Set ``finalize=False`` while processing individual source archives.  Once
    all archive ranges have completed, call :func:`finalize_shards` with the
    complete expected scene allowlist.
    """
    if shard_workers < 1 or read_workers < 1:
        raise ValueError("shard_workers and read_workers must be positive")
    if not scene_ids:
        raise ValueError("scene_ids must not be empty")
    if shard_offset < 0:
        raise ValueError("shard_offset must be non-negative")
    split_root = Path(output_root)
    split_root.mkdir(parents=True, exist_ok=True)
    view_count = _require_consistent_view_count(scene_root, scene_ids)
    shards = split_scene_ids(scene_ids, scenes_per_shard, shard_offset=shard_offset)

    def convert_one(shard: SceneShard) -> dict[str, object]:
        archive = split_root / f"{shard.name}.tar"
        inventory = inventory_path(archive)
        partial = archive.with_suffix(archive.suffix + ".partial")
        inventory_partial = inventory.with_suffix(inventory.suffix + ".partial")
        partial.unlink(missing_ok=True)
        inventory_partial.unlink(missing_ok=True)
        if archive.is_file() and inventory.is_file():
            result = _load_inventory(inventory)
            if result["name"] != shard.name or [str(scene) for scene in result["scene_ids"]] != list(shard.scene_ids):
                raise RuntimeError(f"{inventory}: inventory does not match expected shard {shard.name}")
            if progress_callback is None:
                print(f"CONVERT event=shard_skipped shard={shard.name} reason=inventory", flush=True)
            return result
        if archive.is_file():
            archive.unlink()
        if inventory.is_file():
            inventory.unlink()
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
            print(
                f"CONVERT event=shard completed={len(results)}/{len(shards)} "
                f"shard={result['name']} bytes={result['bytes']}",
                flush=True,
            )
    results.sort(key=lambda result: str(result["name"]))
    if not finalize:
        return {
            "format": WEB_DATASET_FORMAT,
            "split": split_root.name,
            "scenes_per_shard": scenes_per_shard,
            "view_count": view_count,
            "scene_ids": [scene for result in results for scene in result["scene_ids"]],
            "shards": [
                {key: value for key, value in result.items() if key != "sample_records"}
                for result in results
            ],
        }
    return finalize_shards(
        split_root,
        expected_scene_ids if expected_scene_ids is not None else scene_ids,
        scenes_per_shard=scenes_per_shard,
    )


def read_component(payload: bytes) -> dict[str, np.ndarray]:
    """Decode one metadata or packed-byte component."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}
