"""Provider-neutral access to precomputed estimated-depth sidecars."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Sequence

import numpy as np


ESTIMATED_DEPTH_FORMAT = "mvtracker_estimated_depth"
ESTIMATED_DEPTH_SCHEMA_VERSION = 2
ESTIMATED_DEPTH_PER_VIEW_SCHEMA_VERSION = 3
ESTIMATED_DEPTH_TYPE_PROBABILITIES = {
    "gt": 0.70,
    "estimated": 0.20,
    "estimated_cleaned": 0.10,
}


def sample_depth_source(
    rng: np.random.RandomState,
    *,
    variable: bool,
    replay_depth_source: str | None = None,
) -> str:
    """Make the native depth draw, optionally replaying a recorded result."""
    sampled = "gt"
    if variable:
        sampled = str(
            rng.choice(
                tuple(ESTIMATED_DEPTH_TYPE_PROBABILITIES),
                p=tuple(ESTIMATED_DEPTH_TYPE_PROBABILITIES.values()),
            )
        )
    if replay_depth_source is None:
        return sampled
    if replay_depth_source not in ESTIMATED_DEPTH_TYPE_PROBABILITIES:
        raise ValueError(f"unknown depth source: {replay_depth_source}")
    return replay_depth_source


class EstimatedDepthStore:
    def __init__(self, root: str | Path | None, provider: str | None):
        if (root is None) != (provider is None):
            raise ValueError("estimated_depth_root and estimated_depth_provider must be set together")
        self.root = Path(root) if root is not None else None
        self.provider = provider
        self._manifests: dict[str, dict] = {}
        self._arrays: dict[Path, np.ndarray] = {}

    @property
    def enabled(self) -> bool:
        return self.root is not None

    def _manifest(self, scene: str) -> dict:
        if scene not in self._manifests:
            scene_root = self.root / scene
            path = scene_root / "manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            schema_version = manifest.get("schema_version")
            supported = schema_version in {
                ESTIMATED_DEPTH_SCHEMA_VERSION,
                ESTIMATED_DEPTH_PER_VIEW_SCHEMA_VERSION,
            }
            if (
                manifest.get("format") != ESTIMATED_DEPTH_FORMAT
                or not supported
                or manifest.get("provider") != self.provider
                or manifest.get("complete") is not True
            ):
                raise ValueError(f"{path}: incompatible estimated-depth manifest")
            self._manifests[scene] = manifest
        return self._manifests[scene]

    def _mmap(self, path: Path) -> np.ndarray:
        if path not in self._arrays:
            self._arrays[path] = np.load(path, mmap_mode="r", allow_pickle=False)
        return self._arrays[path]

    def load(
        self,
        scene: str,
        view_ids: Sequence[int],
        frame_indices: Sequence[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.enabled:
            raise RuntimeError("estimated-depth storage is disabled")
        scene_root = self.root / scene
        manifest = self._manifest(scene)
        manifest_views = [int(view) for view in manifest["view_ids"]]
        manifest_frame_count = int(manifest["frame_count"])
        try:
            selected_views = [manifest_views.index(int(view)) for view in view_ids]
        except ValueError as error:
            raise ValueError(f"{scene_root}: selected view is absent from sidecar") from error
        if manifest.get("layout") == "per_view" or manifest.get("schema_version") == ESTIMATED_DEPTH_PER_VIEW_SCHEMA_VERSION:
            requested_frames = None if frame_indices is None else [int(frame) for frame in frame_indices]
            if requested_frames is not None and any(
                frame < 0 or frame >= manifest_frame_count for frame in requested_frames
            ):
                raise ValueError(f"{scene_root}: requested frame is outside sidecar bounds")
            depths = []
            masks = []
            for view_position in selected_views:
                view = manifest_views[view_position]
                depth = self._mmap(scene_root / str(view) / "depth.npy")
                cleaned_mask = self._mmap(scene_root / str(view) / "cleaned_mask.npy")
                expected_shape = (
                    manifest_frame_count,
                    *tuple(int(value) for value in manifest["resolution_hw"]),
                )
                if depth.shape != expected_shape or depth.dtype != np.float32:
                    raise ValueError(f"{scene_root}/{view}/depth.npy: expected {expected_shape} float32")
                if cleaned_mask.shape != expected_shape or cleaned_mask.dtype != np.bool_:
                    raise ValueError(f"{scene_root}/{view}/cleaned_mask.npy: expected {expected_shape} bool")
                index = slice(None) if requested_frames is None else requested_frames
                depths.append(np.asarray(depth[index]))
                masks.append(np.asarray(cleaned_mask[index]))
            return np.stack(depths), np.stack(masks)

        depth = self._mmap(scene_root / "depth.npy")
        cleaned_mask = self._mmap(scene_root / "cleaned_mask.npy")
        expected_shape = (
            len(manifest_views),
            manifest_frame_count,
            *tuple(int(value) for value in manifest["resolution_hw"]),
        )
        if depth.shape != expected_shape or depth.dtype != np.float32:
            raise ValueError(f"{scene_root / 'depth.npy'}: expected {expected_shape} float32")
        if cleaned_mask.shape != expected_shape or cleaned_mask.dtype != np.bool_:
            raise ValueError(
                f"{scene_root / 'cleaned_mask.npy'}: expected {expected_shape} bool"
            )
        if frame_indices is not None:
            requested_frames = [int(frame) for frame in frame_indices]
            selected_frames = requested_frames
            if any(frame < 0 or frame >= manifest_frame_count for frame in selected_frames):
                raise ValueError(f"{scene_root}: requested frame is outside sidecar bounds")
            index = np.ix_(selected_views, np.asarray(selected_frames, dtype=np.int64))
            selected_depth = depth[index]
            selected_mask = cleaned_mask[index]
        else:
            selected_depth = depth[selected_views]
            selected_mask = cleaned_mask[selected_views]
        return np.asarray(selected_depth), np.asarray(selected_mask)


class RuntimeRecipeDepthStore:
    """Consume recipe-keyed DA3 depth produced on the container's local SSD."""

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def load(self, step: int, logical_index: int) -> tuple[np.ndarray, np.ndarray, float, int]:
        sample_root = self.root / f"step-{int(step):06d}" / f"sample-{int(logical_index):02d}"
        ready = sample_root / "ready"
        failed = self.root / "failed"
        started = time.perf_counter()
        last_log = started
        while not ready.is_file():
            if failed.is_file():
                raise RuntimeError(failed.read_text(encoding="utf-8").strip())
            now = time.perf_counter()
            if now - last_log >= 10:
                print(
                    "RUNTIME_DEPTH event=waiting "
                    f"step={step} logical_index={logical_index} "
                    f"elapsed_seconds={now - started:.1f}",
                    flush=True,
                )
                last_log = now
            time.sleep(0.05)
        depth_path = sample_root / "depth.npy"
        mask_path = sample_root / "cleaned_mask.npy"
        depth = np.load(depth_path, allow_pickle=False).astype(np.float32, copy=True)
        mask = np.load(mask_path, allow_pickle=False).astype(np.bool_, copy=True)
        byte_count = depth.nbytes + mask.nbytes
        shutil.rmtree(sample_root)
        return depth, mask, time.perf_counter() - started, byte_count
