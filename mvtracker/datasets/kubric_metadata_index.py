"""One-time metadata index for native MV-Kubric scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def build_kubric_metadata_index(data_root, index_root=None, overwrite=False):
    """Build the compact, relocatable index used by the optimized loader."""
    data_root = Path(data_root)
    index_root = Path(index_root or data_root / INDEX_DIRECTORY_NAME)
    manifest_path = index_root / MANIFEST_NAME
    if manifest_path.exists() and not overwrite:
        return manifest_path

    scenes_root = index_root / "scenes"
    scenes_root.mkdir(parents=True, exist_ok=True)
    scene_entries = {}
    scene_paths = sorted(
        (path for path in data_root.iterdir() if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    for scene_path in scene_paths:
        tracks_path = scene_path / "tracks_3d.npz"
        if not tracks_path.is_file():
            continue
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
        for view_dir in view_dirs:
            metadata = _read_json(view_dir / "metadata.json")
            intr, extr = _camera_matrices(metadata)
            sensor_widths.append(metadata["camera"]["sensor_width"])
            focal_lengths.append(metadata["camera"]["focal_length"])
            rgba = sorted(path.name for path in view_dir.glob("rgba_*"))
            depth = sorted(path.name for path in view_dir.glob("depth_*"))
            if len(rgba) != n_frames or len(depth) != n_frames:
                raise ValueError(
                    f"{view_dir}: expected {n_frames} RGB/depth frames, got "
                    f"{len(rgba)}/{len(depth)}"
                )
            if extr.shape != (n_frames, 3, 4):
                raise ValueError(f"{view_dir}: invalid extrinsics shape {extr.shape}")
            view_names.append(view_dir.name)
            rgba_files.append(rgba)
            depth_files.append(depth)
            intrinsics.append(intr)
            extrinsics.append(extr)

        arrays_name = f"{scene_path.name}.npz"
        np.savez(
            scenes_root / arrays_name,
            intrinsics=np.stack(intrinsics),
            extrinsics=np.stack(extrinsics),
            sensor_widths=np.asarray(sensor_widths),
            focal_lengths=np.asarray(focal_lengths),
        )
        scene_entries[scene_path.name] = {
            "n_frames": n_frames,
            "n_tracks": n_tracks,
            "invalid_frame_indices": _invalid_frames(scene_path),
            "view_names": view_names,
            "rgba_files": rgba_files,
            "depth_files": depth_files,
            "arrays": f"scenes/{arrays_name}",
        }

    manifest = {
        "version": INDEX_VERSION,
        "scenes": scene_entries,
    }
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    temporary.replace(manifest_path)
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

    def scene(self, name):
        try:
            entry = self.scenes[name]
        except KeyError as error:
            raise KeyError(f"Scene {name!r} is absent from {self.root / MANIFEST_NAME}") from error
        arrays_path = self.root / entry["arrays"]
        if not arrays_path.is_file():
            raise FileNotFoundError(f"MV-Kubric indexed arrays are missing: {arrays_path}")
        return entry, arrays_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root")
    parser.add_argument("--index-root")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(build_kubric_metadata_index(args.data_root, args.index_root, args.overwrite))


if __name__ == "__main__":
    main()
