"""One-time metadata index for native MV-Kubric scenes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch


INDEX_VERSION = 1
INDEX_DIRECTORY_NAME = "MVTracker_index"
MANIFEST_NAME = "manifest.json"


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _camera_matrices(metadata):
    camera = metadata["camera"]
    intrinsics = torch.tensor(camera["K"], dtype=torch.float64)
    positions = torch.tensor(camera["positions"], dtype=torch.float64)
    quaternions = torch.tensor(camera["quaternions"], dtype=torch.float64)
    quaternions = quaternions / torch.linalg.vector_norm(quaternions, dim=-1, keepdim=True)
    w, x, y, z = quaternions.unbind(-1)
    rotations = torch.stack(
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
        dim=-1,
    ).reshape(-1, 3, 3)

    camera_to_world = torch.zeros((len(positions), 4, 4), dtype=torch.float64)
    camera_to_world[:, :3, :3] = rotations
    camera_to_world[:, :3, 3] = positions
    camera_to_world[:, 3, 3] = 1
    extrinsics = torch.linalg.inv(camera_to_world)[:, :3]

    width, height = metadata["metadata"]["resolution"]
    intrinsics = (
        np.diag([width, height, 1.0])
        @ intrinsics.numpy()
        @ np.diag([1.0, -1.0, -1.0])
    )
    extrinsics = np.diag([1.0, -1.0, -1.0]) @ extrinsics.numpy()
    return intrinsics, extrinsics


def _invalid_frames(scene_path: Path):
    scene_json = scene_path / "scene.json"
    if not scene_json.is_file():
        return []
    return _read_json(scene_json).get("output", {}).get("rgb", {}).get(
        "invalid_frame_indices", []
    )


def compute_source_fingerprint(data_root, scene_names):
    """Hash the native file inventory without reading multi-gigabyte frame payloads."""
    data_root = Path(data_root)
    digest = hashlib.sha256()
    for scene_name in sorted(scene_names):
        scene_path = data_root / scene_name
        paths = [
            scene_path / "tracks_3d.npz",
            scene_path / "tracks_segmentation_ids.npz",
            scene_path / "tracked_objects.json",
            scene_path / "scene.json",
            scene_path / "views.npz",
            scene_path / "cameras.npz",
        ]
        for view_path in sorted(scene_path.glob("view_*")):
            paths.extend((view_path / "metadata.json", view_path / "tracks_2d.npz"))
            paths.extend(sorted(view_path.glob("rgba_*")))
            paths.extend(sorted(view_path.glob("depth_*")))
        for path in paths:
            if not path.is_file():
                continue
            stat = path.stat()
            relative = path.relative_to(data_root).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(b"\n")
    return digest.hexdigest()


def _index_scene(scene_path: Path, data_root: Path, scenes_root: Path):
    tracks_path = scene_path / "tracks_3d.npz"
    if not tracks_path.is_file():
        return None
    with np.load(tracks_path) as tracks_file:
        n_frames, n_tracks, _ = tracks_file["tracks_3d"].shape

    view_dirs = sorted(
        (path for path in scene_path.glob("view_*") if path.is_dir()),
        key=lambda path: int(path.name.rsplit("_", 1)[1]),
    )
    intrinsics = []
    extrinsics = []
    sensor_widths = []
    focal_lengths = []
    rgba_files = []
    depth_files = []
    view_names = []
    inventory = [
        scene_path / "tracks_3d.npz",
        scene_path / "tracks_segmentation_ids.npz",
        scene_path / "tracked_objects.json",
        scene_path / "scene.json",
        scene_path / "views.npz",
        scene_path / "cameras.npz",
    ]
    for view_dir in view_dirs:
        metadata_path = view_dir / "metadata.json"
        metadata = _read_json(metadata_path)
        intr, extr = _camera_matrices(metadata)
        sensor_widths.append(metadata["camera"]["sensor_width"])
        focal_lengths.append(metadata["camera"]["focal_length"])
        rgba_paths = sorted(view_dir.glob("rgba_*"))
        depth_paths = sorted(view_dir.glob("depth_*"))
        if len(rgba_paths) != n_frames or len(depth_paths) != n_frames:
            raise ValueError(
                f"{view_dir}: expected {n_frames} RGB/depth frames, got "
                f"{len(rgba_paths)}/{len(depth_paths)}"
            )
        if extr.shape != (n_frames, 3, 4):
            raise ValueError(f"{view_dir}: invalid extrinsics shape {extr.shape}")
        view_names.append(view_dir.name)
        rgba_files.append([path.name for path in rgba_paths])
        depth_files.append([path.name for path in depth_paths])
        intrinsics.append(intr)
        extrinsics.append(extr)
        inventory.extend((metadata_path, view_dir / "tracks_2d.npz"))
        inventory.extend(rgba_paths)
        inventory.extend(depth_paths)

    arrays_name = f"{scene_path.name}.npz"
    np.savez(
        scenes_root / arrays_name,
        intrinsics=np.stack(intrinsics),
        extrinsics=np.stack(extrinsics),
        sensor_widths=np.asarray(sensor_widths),
        focal_lengths=np.asarray(focal_lengths),
    )
    entry = {
        "n_frames": n_frames,
        "n_tracks": n_tracks,
        "invalid_frame_indices": _invalid_frames(scene_path),
        "view_names": view_names,
        "rgba_files": rgba_files,
        "depth_files": depth_files,
        "arrays": f"scenes/{arrays_name}",
    }
    inventory_records = []
    for path in inventory:
        if path.is_file():
            inventory_records.append(
                (path.relative_to(data_root).as_posix(), path.stat().st_size)
            )
    return scene_path.name, entry, inventory_records


def build_kubric_metadata_index(
    data_root,
    index_root=None,
    overwrite=False,
    workers=16,
    progress_every=25,
):
    """Build the compact, relocatable index used by the optimized loader."""
    data_root = Path(data_root)
    index_root = Path(index_root or data_root / INDEX_DIRECTORY_NAME)
    manifest_path = index_root / MANIFEST_NAME
    if manifest_path.exists() and not overwrite:
        return manifest_path

    scenes_root = index_root / "scenes"
    scenes_root.mkdir(parents=True, exist_ok=True)
    scene_paths = sorted(
        (path for path in data_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    scene_entries = {}
    inventories = {}
    started = time.perf_counter()
    total = len(scene_paths)
    print(f"INDEX event=start scenes={total} workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_index_scene, path, data_root, scenes_root): path.name
            for path in scene_paths
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result is not None:
                scene_name, entry, inventory = result
                scene_entries[scene_name] = entry
                inventories[scene_name] = inventory
            if completed % progress_every == 0 or completed == total:
                elapsed = time.perf_counter() - started
                rate = completed / elapsed
                eta = (total - completed) / rate if rate else 0.0
                print(
                    f"INDEX event=progress completed={completed}/{total} "
                    f"rate={rate:.2f}_scenes_per_second eta_seconds={eta:.0f}",
                    flush=True,
                )

    digest = hashlib.sha256()
    for scene_name in sorted(scene_entries):
        for relative, size in inventories[scene_name]:
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(size).encode("ascii"))
            digest.update(b"\n")

    manifest = {
        "version": INDEX_VERSION,
        "source_fingerprint": digest.hexdigest(),
        "scenes": scene_entries,
    }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    temporary.replace(manifest_path)
    print(
        f"INDEX event=complete scenes={len(scene_entries)} "
        f"seconds={time.perf_counter() - started:.1f}",
        flush=True,
    )
    return manifest_path


class KubricMetadataIndex:
    def __init__(self, index_root):
        self.root = Path(index_root)
        manifest_path = self.root / MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"MV-Kubric metadata index is missing: {manifest_path}")
        manifest = _read_json(manifest_path)
        if manifest.get("version") != INDEX_VERSION:
            raise ValueError(
                f"Unsupported MV-Kubric metadata index version "
                f"{manifest.get('version')!r}; expected {INDEX_VERSION}"
            )
        self.scenes = manifest.get("scenes", {})
        self.source_fingerprint = manifest.get("source_fingerprint")
        if not self.source_fingerprint:
            raise ValueError(f"MV-Kubric metadata index has no source fingerprint: {manifest_path}")
        self._arrays = {}
        started = time.perf_counter()

        def load_arrays(item):
            name, entry = item
            arrays_path = self.root / entry["arrays"]
            if not arrays_path.is_file():
                raise FileNotFoundError(f"MV-Kubric indexed arrays are missing: {arrays_path}")
            with np.load(arrays_path) as arrays:
                return name, {key: arrays[key].copy() for key in arrays.files}

        items = tuple(self.scenes.items())
        print(
            f"INDEX_LOAD event=start scenes={len(items)} workers=16",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(load_arrays, item) for item in items]
            for completed, future in enumerate(as_completed(futures), start=1):
                name, arrays = future.result()
                self._arrays[name] = arrays
                if completed % 250 == 0 or completed == len(items):
                    elapsed = time.perf_counter() - started
                    print(
                        "INDEX_LOAD event=progress "
                        f"completed={completed}/{len(items)} "
                        f"rate={completed / max(elapsed, 1e-9):.1f}_scenes_per_second",
                        flush=True,
                    )

    def validate_source(self, data_root):
        actual = compute_source_fingerprint(data_root, self.scenes)
        if actual != self.source_fingerprint:
            raise ValueError(
                f"MV-Kubric metadata index is stale: source fingerprint {actual} does not "
                f"match indexed fingerprint {self.source_fingerprint}"
            )

    def scene(self, name):
        try:
            entry = self.scenes[name]
        except KeyError as error:
            raise KeyError(f"Scene {name!r} is absent from {self.root / MANIFEST_NAME}") from error
        return entry, self._arrays[name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root")
    parser.add_argument("--index-root")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    print(
        build_kubric_metadata_index(
            args.data_root,
            args.index_root,
            args.overwrite,
            args.workers,
            args.progress_every,
        )
    )


if __name__ == "__main__":
    main()
