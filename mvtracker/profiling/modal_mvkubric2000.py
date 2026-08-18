"""Pinned CPU ingestion for the 2,000-scene MV-Kubric training image."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import time

from huggingface_hub import hf_hub_download, hf_hub_url


MVKUBRIC_REPO = "ethz-vlg/mv3dpt-datasets"
MVKUBRIC_REVISION = "cccb9128fb95d302c662151e65a09377175c2a3a"
ARCHIVE_ROOT_RELATIVE = Path("archives/mvkubric/2000-scenes-v1")
VALIDATION_ROOT_RELATIVE = Path(
    "datasets/kubric-multiview/2000-scenes-v1/validation"
)
VALIDATION_SOURCE_ARCHIVE = "kubric-multiview--train.full.0031-1000.tar.gz"
VALIDATION_SOURCE_SIZE_BYTES = 394_716_348_566
VALIDATION_RANGE_BYTES = 1 << 30
TRAIN_ARCHIVES = (
    {
        "filename": "kubric-multiview--train.full.1001-2000.tar.gz",
        "size_bytes": 404_985_432_432,
        "sha256": "f1dfd394406f620cb567506de79bdad45456996000dd82a9ac322c829aca71dc",
        "scene_start": 1001,
        "scene_end": 2000,
    },
    {
        "filename": "kubric-multiview--train.full.2001-3000.tar.gz",
        "size_bytes": 407_247_025_781,
        "sha256": "4117af637bab3b20f0656a00a6cd7ad2a6c56e689e3de70267acd6209d1c544c",
        "scene_start": 2001,
        "scene_end": 3000,
    },
)
TRAIN_SCENES = tuple(str(scene) for scene in range(1001, 3001))
VALIDATION_SCENES = tuple(str(scene) for scene in range(101, 128))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_training_archives(data_root: Path, token: str) -> list[dict[str, object]]:
    archive_root = data_root / ARCHIVE_ROOT_RELATIVE
    archive_root.mkdir(parents=True, exist_ok=True)
    reports = []
    for spec in TRAIN_ARCHIVES:
        destination = archive_root / spec["filename"]
        if not destination.is_file():
            downloaded = Path(
                hf_hub_download(
                    repo_id=MVKUBRIC_REPO,
                    repo_type="dataset",
                    revision=MVKUBRIC_REVISION,
                    filename=spec["filename"],
                    token=token,
                    local_dir=str(archive_root),
                )
            )
            if downloaded != destination:
                shutil.copyfile(downloaded, destination)
        observed_size = destination.stat().st_size
        if observed_size != spec["size_bytes"]:
            raise RuntimeError(
                f"{destination}: size {observed_size} does not match pinned "
                f"size {spec['size_bytes']}"
            )
        observed_sha256 = sha256(destination)
        if observed_sha256 != spec["sha256"]:
            raise RuntimeError(
                f"{destination}: SHA-256 {observed_sha256} does not match pinned "
                f"hash {spec['sha256']}"
            )
        reports.append(
            {
                **spec,
                "relative_path": str(destination.relative_to(data_root)),
                "sha256": observed_sha256,
            }
        )
    return reports


class _RangeStream(io.RawIOBase):
    """Read a large Xet object through bounded HTTP ranges."""

    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.offset = 0
        self.response = None

    def readable(self) -> bool:
        return True

    def readinto(self, target) -> int:
        if self.offset >= self.size:
            return 0
        if self.response is None:
            end = min(self.size, self.offset + VALIDATION_RANGE_BYTES) - 1
            import requests

            response = requests.get(
                self.url,
                headers={"Range": f"bytes={self.offset}-{end}"},
                stream=True,
                timeout=120,
            )
            response.raise_for_status()
            expected = f"bytes {self.offset}-{end}/{self.size}"
            if response.headers.get("Content-Range") != expected:
                response.close()
                raise RuntimeError(
                    f"unexpected Xet range: {response.headers.get('Content-Range')!r}; "
                    f"expected {expected!r}"
                )
            self.response = response
        block = self.response.raw.read(len(target))
        if not block:
            self.response.close()
            self.response = None
            return self.readinto(target)
        target[: len(block)] = block
        self.offset += len(block)
        if self.offset % VALIDATION_RANGE_BYTES == 0:
            self.response.close()
            self.response = None
        return len(block)

    def close(self):
        if self.response is not None:
            self.response.close()
            self.response = None
        super().close()


def _extract_validation(data_root: Path) -> list[dict[str, object]]:
    staging = Path("/tmp/mvkubric-validation-2000-scenes")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    source_root = staging / "kubric-multiview/train"
    source_root.mkdir(parents=True)
    url = hf_hub_url(
        repo_id=MVKUBRIC_REPO,
        filename=VALIDATION_SOURCE_ARCHIVE,
        repo_type="dataset",
        revision=MVKUBRIC_REVISION,
    )
    seen: set[str] = set()
    stream = io.BufferedReader(
        _RangeStream(url, VALIDATION_SOURCE_SIZE_BYTES),
        buffer_size=8 * 1024 * 1024,
    )
    try:
        with tarfile.open(fileobj=stream, mode="r|gz") as archive:
            for member in archive:
                parts = Path(member.name).parts
                if len(parts) < 3 or parts[:2] != ("kubric-multiview", "train"):
                    continue
                scene_id = parts[2]
                if scene_id not in VALIDATION_SCENES:
                    if seen == set(VALIDATION_SCENES):
                        break
                    continue
                archive.extract(member, staging)
                seen.add(scene_id)
    finally:
        stream.close()
    if seen != set(VALIDATION_SCENES):
        missing = sorted(set(VALIDATION_SCENES) - seen, key=int)
        raise RuntimeError(f"validation archive is missing scenes {missing}")

    destination_root = data_root / VALIDATION_ROOT_RELATIVE
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True)
    reports = []
    for scene_id in VALIDATION_SCENES:
        source_scene = source_root / scene_id
        if not (source_scene / "tracks_3d.npz").is_file():
            raise RuntimeError(f"validation scene {scene_id} is missing tracks_3d.npz")
        views = sorted(path.parent.name for path in source_scene.glob("view_*/metadata.json"))
        if not views:
            raise RuntimeError(f"validation scene {scene_id} has no camera views")
        reports.append(
            {
                "scene_id": scene_id,
                "views": views,
                "files": sum(1 for path in source_scene.rglob("*") if path.is_file()),
                "bytes": sum(path.stat().st_size for path in source_scene.rglob("*") if path.is_file()),
            }
        )
        shutil.copytree(source_scene, destination_root / scene_id)
    shutil.rmtree(staging)
    return reports


def materialize_mvkubric2000(data_root: Path, token: str) -> dict[str, object]:
    """Download immutable archives and materialize only the held-out scenes."""

    data_root = Path(data_root)
    manifest_path = data_root / "mvkubric2000-data-manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("revision") != MVKUBRIC_REVISION
            or manifest.get("train_scene_count") != len(TRAIN_SCENES)
            or manifest.get("validation_scene_count") != len(VALIDATION_SCENES)
        ):
            raise RuntimeError("existing MV-Kubric 2,000-scene manifest is incompatible")
        _download_training_archives(data_root, token)
        validation_root = data_root / VALIDATION_ROOT_RELATIVE
        observed = {path.name for path in validation_root.iterdir() if path.is_dir()}
        if observed != set(VALIDATION_SCENES):
            raise RuntimeError("existing MV-Kubric validation shard is incomplete")
        return manifest

    started = time.perf_counter()
    archives = _download_training_archives(data_root, token)
    validation = _extract_validation(data_root)
    manifest = {
        "schema_version": 1,
        "repo_id": MVKUBRIC_REPO,
        "revision": MVKUBRIC_REVISION,
        "train_scene_count": len(TRAIN_SCENES),
        "train_scene_start": 1001,
        "train_scene_end": 3000,
        "validation_scene_count": len(VALIDATION_SCENES),
        "validation_scene_start": 101,
        "validation_scene_end": 127,
        "archives": archives,
        "validation_archive": {
            "repo_id": MVKUBRIC_REPO,
            "revision": MVKUBRIC_REVISION,
            "filename": VALIDATION_SOURCE_ARCHIVE,
            "size_bytes": VALIDATION_SOURCE_SIZE_BYTES,
            "integrity": "bounded-range-prefix-only",
            "range_bytes": VALIDATION_RANGE_BYTES,
            "relative_path": str(VALIDATION_ROOT_RELATIVE),
        },
        "validation": validation,
        "elapsed_seconds": time.perf_counter() - started,
    }
    temporary = manifest_path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)
    return manifest
