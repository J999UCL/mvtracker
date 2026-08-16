"""Small local-SSD staging helpers for the VGGT-Omega Modal profile.

The Modal Volume is a network filesystem.  The profile therefore downloads the
private DIEGESIS snapshot directly to the container and copies only the four
MV-Kubric scenes needed for the short profile to local SSD before inference.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import time
from typing import Iterable


DIEGESIS_REPO = "j99999/diegesis"
DIEGESIS_REVISION = "81389015a6d713a848a120e34850f360621bcdce"
MVKUBRIC_SCENES = ("900", "901", "902", "903")


def _tree_stats(root: Path) -> dict[str, int]:
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "file_count": len(files),
        "size_bytes": sum(path.stat().st_size for path in files),
    }


def stage_diegesis(local_root: Path, token: str) -> dict:
    """Download the pinned DIEGESIS snapshot directly onto local container SSD."""
    from huggingface_hub import snapshot_download

    local_root = Path(local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    snapshot_download(
        repo_id=DIEGESIS_REPO,
        repo_type="dataset",
        revision=DIEGESIS_REVISION,
        token=token,
        local_dir=local_root,
    )
    scenes_root = local_root / "scenes"
    return {
        "repo_id": DIEGESIS_REPO,
        "revision": DIEGESIS_REVISION,
        "local_root": str(local_root),
        "scenes_root": str(scenes_root),
        "download_seconds": round(time.perf_counter() - started, 2),
        **_tree_stats(local_root),
    }


def _copy_mvkubric_scene(source: Path, destination: Path) -> dict[str, int | str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied_files = 0
    copied_bytes = 0
    for view_root in sorted(source.glob("view_*")):
        if not view_root.is_dir():
            continue
        output_view = destination / view_root.name
        output_view.mkdir(parents=True, exist_ok=True)
        metadata = view_root / "metadata.json"
        if not metadata.is_file():
            raise FileNotFoundError(metadata)
        files: Iterable[Path] = (metadata, *sorted(view_root.glob("rgba_*.png")))
        for path in files:
            target = output_view / path.name
            shutil.copy2(path, target)
            copied_files += 1
            copied_bytes += target.stat().st_size
        if not list(view_root.glob("rgba_*.png")):
            raise FileNotFoundError(f"no rgba_*.png files in {view_root}")
    return {
        "scene": source.name,
        "source": str(source),
        "destination": str(destination),
        "file_count": copied_files,
        "size_bytes": copied_bytes,
    }


def stage_mvkubric(volume_root: Path, local_root: Path) -> dict:
    """Copy only scenes 900--903 RGB PNGs and camera metadata from the Volume."""
    volume_root = Path(volume_root)
    local_root = Path(local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    scenes = []
    for scene_name in MVKUBRIC_SCENES:
        source = volume_root / scene_name
        if not source.is_dir():
            raise FileNotFoundError(source)
        scenes.append(
            _copy_mvkubric_scene(source, local_root / scene_name)
        )
    return {
        "source_root": str(volume_root),
        "local_root": str(local_root),
        "scene_ids": list(MVKUBRIC_SCENES),
        "scenes": scenes,
        **_tree_stats(local_root),
    }


def write_staging_report(path: Path, *, diegesis: dict, mvkubric: dict) -> dict:
    report = {
        "format": "mvtracker_vggt_omega_modal_staging",
        "diegesis": diegesis,
        "mvkubric": mvkubric,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
